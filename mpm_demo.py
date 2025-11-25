import os
import taichi as ti

ti.init(arch=ti.gpu, device_memory_fraction=0.8)

# ==========================
# MPM params
# ==========================
dim = 2
n_particles = 10000
n_grid = 128

dx = 1.0 / n_grid
inv_dx = 1.0 / dx

dt = 5e-4
gravity = ti.Vector([0.0, -9.81])

p_vol = (dx * 0.5) ** 2
p_rho = 1.0
p_mass = p_vol * p_rho

E = 5e7
nu = 0.45
mu_0 = E / (2 * (1 + nu))
lam_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

# ==========================
# Fields
# ==========================
x2 = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
v = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)

x3 = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)

grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))


@ti.kernel
def init_particles():
    for i in range(n_particles):
        gx = ti.random() * 0.5 + 0.3
        gy = ti.random() * 0.3 + 0.5
        x2[i] = ti.Vector([gx, gy])
        v[i] = ti.Vector([0.0, 0.0])
        F[i] = ti.Matrix.identity(ti.f32, dim)
        C[i] = ti.Matrix.zero(ti.f32, dim, dim)


@ti.kernel
def clear_grid():
    for i, j in grid_m:
        grid_v[i, j] = ti.Vector.zero(ti.f32, dim)
        grid_m[i, j] = 0.0


@ti.kernel
def p2g():
    for p in x2:
        Xp = x2[p]
        vp = v[p]
        Cp = C[p]
        Fp = F[p]

        Fp = (ti.Matrix.identity(ti.f32, dim) + dt * Cp) @ Fp
        J = Fp.determinant()
        J_min, J_max = 0.8, 1.25
        J_clamped = ti.min(ti.max(J, J_min), J_max)
        scale = ti.sqrt(J_clamped / (J + 1e-8))
        Fp = scale * Fp
        F[p] = Fp
        J = J_clamped

        # Neo-Hookean
        invT = Fp.inverse().transpose()
        mu = mu_0
        lam = lam_0
        P = mu * (Fp - invT) + lam * ti.log(J) * invT

        # MLS-MPM
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * P
        affine = stress + p_mass * Cp

        base = (Xp * inv_dx - 0.5).cast(int)
        fx = Xp * inv_dx - base.cast(ti.f32)

        w = [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2,
        ]

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                idx = base + offset
                if 0 <= idx[0] < n_grid and 0 <= idx[1] < n_grid:
                    weight = w[i][0] * w[j][1]
                    dpos = (offset.cast(ti.f32) - fx) * dx
                    mass_contrib = weight * p_mass
                    momentum = mass_contrib * (vp + affine @ dpos)
                    grid_v[idx] += momentum
                    grid_m[idx] += mass_contrib


@ti.kernel
def grid_op():
    damping = 0.997

    for i, j in grid_m:
        m = grid_m[i, j]
        if m > 0:
            v_ij = grid_v[i, j] / m
            v_ij += dt * gravity
            v_ij *= damping

            eps = 2
            if i < eps and v_ij[0] < 0:
                v_ij[0] = 0
            if i > n_grid - eps and v_ij[0] > 0:
                v_ij[0] = 0
            if j > n_grid - eps and v_ij[1] > 0:
                v_ij[1] = 0

            grid_v[i, j] = v_ij


@ti.kernel
def g2p():
    restitution = 1.0
    floor_y = 0.1

    for p in x2:
        Xp = x2[p]
        base = (Xp * inv_dx - 0.5).cast(int)
        fx = Xp * inv_dx - base.cast(ti.f32)

        w = [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2,
        ]

        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                idx = base + offset
                if 0 <= idx[0] < n_grid and 0 <= idx[1] < n_grid:
                    weight = w[i][0] * w[j][1]
                    dpos = (offset.cast(ti.f32) - fx) * dx
                    g_v = grid_v[idx]
                    new_v += weight * g_v
                    new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        v[p] = new_v
        C[p] = new_C
        Xp = Xp + dt * new_v

        if Xp[1] < floor_y:
            Xp[1] = floor_y
            if v[p][1] < 0:
                v[p][1] = -restitution * v[p][1]

        x2[p] = Xp


@ti.kernel
def update_render_pos():
    for p in x2:
        x3[p] = ti.Vector([x2[p][0], x2[p][1], 0.5])


def substep():
    clear_grid()
    p2g()
    grid_op()
    g2p()
    update_render_pos()


# ==========================
# Main program
# ==========================
def main():
    os.makedirs("frames", exist_ok=True)
    os.makedirs("frames/mpm_demo", exist_ok=True)

    init_particles()
    update_render_pos()

    window = ti.ui.Window("MPM Jelly Fast", (1280, 720), show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()

    n_frames = 400
    substeps_per_frame = 50

    for frame in range(n_frames):
        for _ in range(substeps_per_frame):
            substep()

        camera = ti.ui.Camera()
        camera.position(0.5, 0.8, 2.0)
        camera.lookat(0.5, 0.5, 0.0)
        camera.up(0.0, 1.0, 0.0)

        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light((1.5, 2.5, 2.0), (1.0, 1.0, 1.0))

        scene.particles(x3, radius=0.01, color=(0.2, 0.7, 1.0))

        canvas.scene(scene)

        fname = f"frames/mpm_demo/frame_{frame:04d}.png"
        window.save_image(fname)
        print("saved", fname)

    os.system(
        "ffmpeg -framerate 30 -i frames/mpm_demo/frame_%04d.png "
        "-c:v libx264 -pix_fmt yuv420p mpm_demo.mp4 -y"
    )


if __name__ == "__main__":
    main()
