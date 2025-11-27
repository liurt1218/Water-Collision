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

# 只用 structural springs（左右 + 上下）
struct_springs_py = []  # (i, j, rest_length)


def idx(ix, iy):
    return iy * Nx + ix


for iy in range(Ny):
    for ix in range(Nx):
        i = idx(ix, iy)
        if ix + 1 < Nx:
            j = idx(ix + 1, iy)
            struct_springs_py.append((i, j, cloth_dx))
        if iy + 1 < Ny:
            j = idx(ix, iy + 1)
            struct_springs_py.append((i, j, cloth_dx))

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

# 只画 structural 线段，方便 debug
line_indices = ti.field(dtype=ti.i32, shape=num_springs * 2)


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
        pinned[i] = 0

    # 顶边一整排固定住
    for ix in range(Nx):
        i = (Ny - 1) * Nx + ix
        pinned[i] = 1
        inv_mass[i] = 0.0
        v[i] = ti.Vector([0.0, 0.0, 0.0])


def init_springs_and_lines():
    # 填 springs
    for s, (i, j, L0) in enumerate(struct_springs_py):
        spr_ij[s] = ti.Vector([i, j])
        spr_rest[s] = L0

    # lines 只画 structural springs
    idx_np = np.zeros(num_springs * 2, dtype=np.int32)
    for s, (i, j, _) in enumerate(struct_springs_py):
        idx_np[2 * s + 0] = i
        idx_np[2 * s + 1] = j
    line_indices.from_numpy(idx_np)


def init_cloth_indices():
    # Python 端生成三角形索引
    idx_list = []
    for iy in range(Ny - 1):
        for ix in range(Nx - 1):
            i0 = idx(ix, iy)
            i1 = idx(ix + 1, iy)
            i2 = idx(ix + 1, iy + 1)
            i3 = idx(ix, iy + 1)
            # 两个三角形 (i0,i1,i2) 和 (i0,i2,i3)
            idx_list.extend([i0, i1, i2, i0, i2, i3])
    arr = np.array(idx_list, dtype=np.int32)
    cloth_indices.from_numpy(arr)


# =====================
# 物理步骤（简单稳定版）
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
    # PBD style 约束：只做 structural，数值非常稳定
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
def update_velocity(dt: ti.f32, damping: ti.f32):
    for i in range(n_verts):
        if pinned[i] == 0:
            v[i] = (x[i] - x_prev[i]) / dt
            v[i] *= damping


# =====================
# 主程序：离线渲染 cloth + ground
# =====================
def main():
    os.makedirs("frames/cloth_test", exist_ok=True)

    init_cloth()
    init_springs_and_lines()
    init_cloth_indices()

    window = ti.ui.Window(
        "ClothOnPlaneClean",
        (1280, 720),
        vsync=False,
        show_window=False,
    )
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    dt = 1.0 / 240.0  # 时间步小一点，稳定
    substeps = 2
    iter_per_step = 10  # 每步 spring 迭代次数
    n_frames = 240

    for frame in range(n_frames):
        for _ in range(substeps):
            copy_prev_pos()
            integrate(dt)
            # 多次迭代结构弹簧
            for _ in range(iter_per_step):
                solve_springs(0.3)  # stiffness per-iter
                collide_with_ground(0.005)
            update_velocity(dt, 0.98)  # 一点阻尼

        # === 渲染 ===
        camera.position(0.4, 0.7, 1.3)
        camera.lookat(0.0, 0.2, 0.0)
        camera.up(0.0, 1.0, 0.0)
        scene.set_camera(camera)
        scene.ambient_light((0.7, 0.7, 0.7))
        scene.point_light(pos=(1.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))

        canvas.set_background_color((0.1, 0.1, 0.12))

        # 地面：简单画个灰色方块表示桌面（仅视觉用）
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

        # 布：用三角 mesh 画面（有些版本可能只看到线/黑面，但粒子一定是对的）
        scene.mesh(
            x,
            indices=cloth_indices,
            color=(0.9, 0.2, 0.2),
            two_sided=True,
            index_count=n_cloth_indices,
        )

        # 同时再画一遍结构线网格 + 粒子，保证你“视觉上能确认是整块布”
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
