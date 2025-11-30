import os
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, kernel_profiler=False)

# =====================
# Cloth parameters
# =====================
Nx, Ny = 30, 30  # cloth 网格分辨率
n_verts = Nx * Ny
cloth_dx = 0.03  # 顶点间距
cloth_y0 = 0.6  # 初始高度
mass_per_vertex = 1.0
inv_mass_val = 1.0 / mass_per_vertex

gravity = ti.Vector([0.0, -9.8, 0.0])
ground_y = 0.0  # 平面 y = 0 当桌面

# =====================
# Rigid box
# =====================
BOX_CX, BOX_CY, BOX_CZ = 0.0, 0.25, 0.0  # box center
BOX_HX, BOX_HY, BOX_HZ = 0.2, 0.15, 0.2  # half extents

# =====================
# Spring list（结构 + 剪切 + 弯曲）
# =====================
# struct_springs_py 里存所有弹簧：(i, j, rest_length)
struct_springs_py = []


def idx(ix, iy):
    return iy * Nx + ix


# 预计算不同类型弹簧的 rest length
L_struct = cloth_dx
L_shear = cloth_dx * np.sqrt(2.0)
L_bend = cloth_dx * 2.0

for iy in range(Ny):
    for ix in range(Nx):
        i = idx(ix, iy)

        # ---------- structural springs（左右 + 上下） ----------
        if ix + 1 < Nx:
            j = idx(ix + 1, iy)
            struct_springs_py.append((i, j, L_struct))
        if iy + 1 < Ny:
            j = idx(ix, iy + 1)
            struct_springs_py.append((i, j, L_struct))

        # ---------- bending springs（隔一个点） ----------
        if ix + 2 < Nx:
            j = idx(ix + 2, iy)
            struct_springs_py.append((i, j, L_bend))
        if iy + 2 < Ny:
            j = idx(ix, iy + 2)
            struct_springs_py.append((i, j, L_bend))

        # ---------- shear springs（对角线） ----------
        if (ix + 1 < Nx) and (iy + 1 < Ny):
            j = idx(ix + 1, iy + 1)
            struct_springs_py.append((i, j, L_shear))
        if (ix + 1 < Nx) and (iy - 1 >= 0):
            j = idx(ix + 1, iy - 1)
            struct_springs_py.append((i, j, L_shear))

num_springs = len(struct_springs_py)

# =====================
# Taichi fields
# =====================
x = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)  # 位置
x_prev = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)  # 上一帧位置
v = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)  # 速度
inv_mass = ti.field(dtype=ti.f32, shape=n_verts)
pinned = ti.field(dtype=ti.i32, shape=n_verts)

spr_ij = ti.Vector.field(2, dtype=ti.i32, shape=num_springs)
spr_rest = ti.field(dtype=ti.f32, shape=num_springs)

# cloth 三角形 index，用于 mesh 渲染
n_quads = (Nx - 1) * (Ny - 1)
n_cloth_indices = n_quads * 6
cloth_indices = ti.field(dtype=ti.i32, shape=n_cloth_indices)

# 只画 line 方便 debug
line_indices = ti.field(dtype=ti.i32, shape=num_springs * 2)

# Rigid box mesh
box_verts = ti.Vector.field(3, dtype=ti.f32, shape=8)
box_indices = ti.field(dtype=ti.i32, shape=36)


# =====================
# 初始化
# =====================
@ti.kernel
def init_cloth():
    for iy, ix in ti.ndrange(Ny, Nx):
        i = iy * Nx + ix
        x[i] = ti.Vector(
            [
                (ix - 0.5 * (Nx - 1)) * cloth_dx,  # x
                cloth_y0,  # y
                (iy - 0.5 * (Ny - 1)) * cloth_dx,  # z
            ]
        )
        v[i] = ti.Vector([0.0, 0.0, 0.0])
        inv_mass[i] = inv_mass_val
        pinned[i] = 0  # 不再固定顶边，全部自由下落


def init_springs_and_lines():
    # 填 springs
    for s, (i, j, L0) in enumerate(struct_springs_py):
        spr_ij[s] = ti.Vector([i, j])
        spr_rest[s] = L0

    # lines：每条弹簧画一条线
    idx_np = np.zeros(num_springs * 2, dtype=np.int32)
    for s, (i, j, _) in enumerate(struct_springs_py):
        idx_np[2 * s + 0] = i
        idx_np[2 * s + 1] = j
    line_indices.from_numpy(idx_np)


def init_cloth_indices():
    idx_list = []
    for iy in range(Ny - 1):
        for ix in range(Nx - 1):
            i0 = idx(ix, iy)
            i1 = idx(ix + 1, iy)
            i2 = idx(ix + 1, iy + 1)
            i3 = idx(ix, iy + 1)
            idx_list.extend([i0, i1, i2, i0, i2, i3])
    arr = np.array(idx_list, dtype=np.int32)
    cloth_indices.from_numpy(arr)


def init_box_mesh():
    cx, cy, cz = BOX_CX, BOX_CY, BOX_CZ
    hx, hy, hz = BOX_HX, BOX_HY, BOX_HZ

    verts_np = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float32,
    )

    idx_np = np.array(
        [
            0,
            1,
            2,
            0,
            2,
            3,
            4,
            6,
            5,
            4,
            7,
            6,
            0,
            3,
            7,
            0,
            7,
            4,
            1,
            5,
            6,
            1,
            6,
            2,
            0,
            4,
            5,
            0,
            5,
            1,
            3,
            2,
            6,
            3,
            6,
            7,
        ],
        dtype=np.int32,
    )

    box_verts.from_numpy(verts_np)
    box_indices.from_numpy(idx_np)


# =====================
# 物理步骤
# =====================
@ti.kernel
def copy_prev_pos():
    for i in range(n_verts):
        x_prev[i] = x[i]


@ti.kernel
def integrate(dt: ti.f32):
    for i in range(n_verts):
        if pinned[i] == 0:
            v[i] += dt * gravity
            x[i] += dt * v[i]


@ti.kernel
def solve_springs(iter_stiff: ti.f32):
    for s in range(num_springs):
        ij = spr_ij[s]
        i = ij[0]
        j = ij[1]
        xi = x[i]
        xj = x[j]
        rest = spr_rest[s]
        delta = xj - xi
        dist = delta.norm() + 1e-6
        C = dist - rest
        if ti.abs(C) > 1e-6:
            dir = delta / dist
            wi = inv_mass[i]
            wj = inv_mass[j]
            wsum = wi + wj
            if wsum > 0:
                corr = iter_stiff * C / wsum * dir
                if pinned[i] == 0:
                    x[i] += wi * corr
                if pinned[j] == 0:
                    x[j] -= wj * corr


@ti.kernel
def collide_with_ground(offset: ti.f32):
    for i in range(n_verts):
        if pinned[i] == 0:
            p = x[i]
            if p.y < ground_y + offset:
                p.y = ground_y + offset
                x[i] = p
                v[i].y = 0.0


@ti.kernel
def collide_with_box():
    cx, cy, cz = BOX_CX, BOX_CY, BOX_CZ
    hx, hy, hz = BOX_HX, BOX_HY, BOX_HZ

    for i in range(n_verts):
        if pinned[i] == 0:
            p = x[i]
            minb = ti.Vector([cx - hx, cy - hy, cz - hz])
            maxb = ti.Vector([cx + hx, cy + hy, cz + hz])

            inside = (
                (p.x >= minb.x)
                and (p.x <= maxb.x)
                and (p.y >= minb.y)
                and (p.y <= maxb.y)
                and (p.z >= minb.z)
                and (p.z <= maxb.z)
            )

            if inside:
                px_min = p.x - minb.x
                px_max = maxb.x - p.x
                py_min = p.y - minb.y
                py_max = maxb.y - p.y
                pz_min = p.z - minb.z
                pz_max = maxb.z - p.z

                pen = ti.Vector([px_min, px_max, py_min, py_max, pz_min, pz_max])
                min_idx = 0
                min_val = pen[0]
                for k in ti.static(range(1, 6)):
                    if pen[k] < min_val:
                        min_val = pen[k]
                        min_idx = k

                if min_idx == 0:
                    p.x = minb.x
                    v[i].x = 0.0
                elif min_idx == 1:
                    p.x = maxb.x
                    v[i].x = 0.0
                elif min_idx == 2:
                    p.y = minb.y
                    v[i].y = 0.0
                elif min_idx == 3:
                    p.y = maxb.y
                    v[i].y = 0.0
                elif min_idx == 4:
                    p.z = minb.z
                    v[i].z = 0.0
                else:
                    p.z = maxb.z
                    v[i].z = 0.0

                x[i] = p


@ti.kernel
def update_velocity(dt: ti.f32, damping: ti.f32):
    for i in range(n_verts):
        if pinned[i] == 0:
            v[i] = (x[i] - x_prev[i]) / dt
            v[i] *= damping


# =====================
# 主程序
# =====================
def main():
    os.makedirs("frames/cloth_test", exist_ok=True)

    init_cloth()
    init_springs_and_lines()
    init_cloth_indices()
    init_box_mesh()

    window = ti.ui.Window(
        "ClothOnBox",
        (1280, 720),
        vsync=False,
        show_window=False,
    )
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    dt = 1.0 / 240.0
    substeps = 2
    iter_per_step = 10
    n_frames = 240

    for frame in range(n_frames):
        for _ in range(substeps):
            copy_prev_pos()
            integrate(dt)
            for _ in range(iter_per_step):
                solve_springs(0.3)
                collide_with_ground(0.005)
                collide_with_box()
            update_velocity(dt, 0.98)

        camera.position(0.4, 0.7, 1.3)
        camera.lookat(0.0, 0.2, 0.0)
        camera.up(0.0, 1.0, 0.0)
        scene.set_camera(camera)
        scene.ambient_light((0.7, 0.7, 0.7))
        scene.point_light(pos=(1.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))

        canvas.set_background_color((0.1, 0.1, 0.12))

        # ground
        ground_verts = np.array(
            [
                [-0.5, ground_y, -0.5],
                [0.5, ground_y, -0.5],
                [0.5, ground_y, 0.5],
                [-0.5, ground_y, 0.5],
            ],
            dtype=np.float32,
        )
        ground_idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        gv = ti.Vector.field(3, dtype=ti.f32, shape=4)
        gi = ti.field(dtype=ti.i32, shape=6)
        gv.from_numpy(ground_verts)
        gi.from_numpy(ground_idx)

        scene.mesh(
            gv,
            indices=gi,
            color=(0.25, 0.25, 0.25),
            two_sided=True,
            index_count=6,
        )

        # box
        scene.mesh(
            box_verts,
            indices=box_indices,
            color=(0.2, 0.5, 0.9),
            two_sided=True,
            index_count=36,
        )

        # cloth
        scene.mesh(
            x,
            indices=cloth_indices,
            color=(0.9, 0.2, 0.2),
            two_sided=True,
            index_count=n_cloth_indices,
        )

        # springs + particles
        scene.lines(
            x,
            indices=line_indices,
            width=1.2,
            color=(1.0, 0.6, 0.2),
            index_count=num_springs * 2,
        )
        scene.particles(
            x,
            radius=0.005,
            color=(0.1, 1.0, 0.1),
        )

        canvas.scene(scene)
        out_path = f"frames/cloth_test/frame_{frame:04d}.png"
        window.save_image(out_path)
        print("saved", out_path)

    os.system(
        "ffmpeg -framerate 30 -i frames/cloth_test/frame_%04d.png "
        "-c:v libx264 -pix_fmt yuv420p cloth_test.mp4 -y"
    )


if __name__ == "__main__":
    main()
