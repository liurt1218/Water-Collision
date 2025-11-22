import taichi as ti
import math
import os

ti.init(arch=ti.gpu, device_memory_fraction=0.8)

# ==========================
# Global configuration
# ==========================
dim = 3

# Fluid block size
x, y, z = 0.96, 0.96, 0.12
dx = 0.024
nx, ny, nz = int(x / dx), int(y / dx), int(z / dx)
n_fluid = nx * ny * nz
dh = 2 * dx

# Rigidbody particles
rb_nx, rb_ny, rb_nz = 4, 4, 4
n_rigid = rb_nx * rb_ny * rb_nz

n_particles = n_fluid + n_rigid

dt = 1e-3
rho0 = 1000.0
g = ti.Vector([0.0, -9.81, 0.0])

# DFSPH
max_iter_density = 1000
max_iter_div = 1000
max_error = 1e-4
max_error_V = 1e-3
eps = 1e-5

# Kernel radius
h = 0.06

# Domain
domain_min = ti.Vector([0.0, 0.0, 0.0])
domain_max = ti.Vector([1.0, 1.0, 1.0])

# Rigidbody
rigid_half = ti.Vector([0.08, 0.08, 0.08])
rigid_mass = 5.0
rigid_restitution = 0.3

# ==========================
# Fields
# ==========================
x = ti.Vector.field(dim, ti.f32, shape=n_particles)
v = ti.Vector.field(dim, ti.f32, shape=n_particles)
a = ti.Vector.field(dim, ti.f32, shape=n_particles)
is_fluid = ti.field(ti.i32, shape=n_particles)
is_dynamic = ti.field(ti.i32, shape=n_particles)
rest_volume = ti.field(ti.f32, shape=n_particles)
density = ti.field(ti.f32, shape=n_particles)

# DFSPH
alpha = ti.field(ti.f32, shape=n_particles)
density_star = ti.field(ti.f32, shape=n_particles)
density_deriv = ti.field(ti.f32, shape=n_particles)
kappa = ti.field(ti.f32, shape=n_particles)
kappa_v = ti.field(ti.f32, shape=n_particles)

# Translation
rb_pos = ti.Vector.field(dim, ti.f32, shape=())
rb_vel = ti.Vector.field(dim, ti.f32, shape=())
rb_force = ti.Vector.field(dim, ti.f32, shape=())

# Local
rb_local = ti.Vector.field(dim, ti.f32, shape=n_rigid)

# Rendering
cube_vertices = ti.Vector.field(3, ti.f32, shape=8)
cube_indices = ti.field(ti.i32, shape=36)  # 12 triangles * 3


# ==========================
# Kernels
# ==========================
@ti.func
def kernel_W(r):
    h_ = ti.static(h)
    dim_ = ti.static(dim)

    res = 0.0

    if r >= 0 and r <= h_:
        k = 0.0
        if dim_ == 3:
            k = 8.0 / math.pi
        elif dim_ == 2:
            k = 40.0 / (7.0 * math.pi)
        else:
            k = 4.0 / 3.0

        k /= h_**dim_

        q = r / h_
        if q <= 1.0:
            if q <= 0.5:
                res = k * (6 * q * q * q - 6 * q * q + 1)
            else:
                res = k * 2 * (1 - q) ** 3

    return res


@ti.func
def kernel_grad(R):
    dim_ = ti.static(dim)
    h_ = ti.static(h)

    R_mod = R.norm()
    grad = ti.Vector.zero(ti.f32, dim_)

    if 0 < R_mod <= h_:
        k = 0.0
        if dim_ == 3:
            k = 8.0 / math.pi
        elif dim_ == 2:
            k = 40.0 / (7.0 * math.pi)
        else:
            k = 4.0 / 3.0
        k = 6.0 * k / (h_**dim_)

        q = R_mod / h_
        grad_q = R / (R_mod * h_)

        if q <= 0.5:
            grad = k * q * (3.0 * q - 2.0) * grad_q
        else:
            factor = 1.0 - q
            grad = k * (-factor * factor) * grad_q

    return grad


# ==========================
# Initialization
# ==========================
@ti.kernel
def init_fluid_and_rigid():
    # Fluid block
    base = ti.Vector([0.02, 0.02, 0.02])
    for i in range(n_fluid):
        ix = i % nx
        iy = (i // nx) % ny
        iz = i // (nx * ny)
        x[i] = base + ti.Vector([ix + 0.5, iy + 0.5, iz + 0.5]) * dx
        v[i] = ti.Vector.zero(ti.f32, dim)
        a[i] = ti.Vector.zero(ti.f32, dim)
        is_fluid[i] = 1
        is_dynamic[i] = 1

        rest_volume[i] = dx**3

    rb_dx = 2.0 * rigid_half[0] / rb_nx
    lb = -rigid_half + 0.5 * rb_dx
    center = ti.Vector([0.7, rigid_half[1], 0.5])

    for k in range(n_rigid):
        ix = k % rb_nx
        iy = (k // rb_nx) % rb_ny
        iz = k // (rb_nx * rb_ny)
        local = lb + ti.Vector([ix, iy, iz]) * rb_dx
        rb_local[k] = local

        pid = n_fluid + k
        x[pid] = center + local
        v[pid] = ti.Vector.zero(ti.f32, dim)
        a[pid] = ti.Vector.zero(ti.f32, dim)
        is_fluid[pid] = 0
        is_dynamic[pid] = 1
        rest_volume[pid] = (
            2 * rigid_half[0] * 2 * rigid_half[1] * 2 * rigid_half[2]
        ) / n_rigid

    rb_pos[None] = center
    rb_vel[None] = ti.Vector.zero(ti.f32, dim)
    rb_force[None] = ti.Vector.zero(ti.f32, dim)


@ti.kernel
def init_cube_indices():
    cube_indices[0] = 0
    cube_indices[1] = 1
    cube_indices[2] = 2
    cube_indices[3] = 0
    cube_indices[4] = 2
    cube_indices[5] = 3
    # back
    cube_indices[6] = 4
    cube_indices[7] = 5
    cube_indices[8] = 6
    cube_indices[9] = 4
    cube_indices[10] = 6
    cube_indices[11] = 7
    # bottom
    cube_indices[12] = 0
    cube_indices[13] = 1
    cube_indices[14] = 5
    cube_indices[15] = 0
    cube_indices[16] = 5
    cube_indices[17] = 4
    # top
    cube_indices[18] = 3
    cube_indices[19] = 2
    cube_indices[20] = 6
    cube_indices[21] = 3
    cube_indices[22] = 6
    cube_indices[23] = 7
    # left
    cube_indices[24] = 0
    cube_indices[25] = 3
    cube_indices[26] = 7
    cube_indices[27] = 0
    cube_indices[28] = 7
    cube_indices[29] = 4
    # right
    cube_indices[30] = 1
    cube_indices[31] = 2
    cube_indices[32] = 6
    cube_indices[33] = 1
    cube_indices[34] = 6
    cube_indices[35] = 5


@ti.kernel
def update_cube_vertices():
    c = rb_pos[None]
    hx, hy, hz = rigid_half[0], rigid_half[1], rigid_half[2]
    cube_vertices[0] = ti.Vector([c.x - hx, c.y - hy, c.z - hz])
    cube_vertices[1] = ti.Vector([c.x + hx, c.y - hy, c.z - hz])
    cube_vertices[2] = ti.Vector([c.x + hx, c.y + hy, c.z - hz])
    cube_vertices[3] = ti.Vector([c.x - hx, c.y + hy, c.z - hz])
    cube_vertices[4] = ti.Vector([c.x - hx, c.y - hy, c.z + hz])
    cube_vertices[5] = ti.Vector([c.x + hx, c.y - hy, c.z + hz])
    cube_vertices[6] = ti.Vector([c.x + hx, c.y + hy, c.z + hz])
    cube_vertices[7] = ti.Vector([c.x - hx, c.y + hy, c.z + hz])


# ==========================
# DFSPH
# ==========================
@ti.kernel
def compute_density():
    for i in range(n_particles):
        if is_fluid[i] == 1:
            rho = rest_volume[i] * kernel_W(0.0)
            xi = x[i]
            for j in range(n_particles):
                if j != i:
                    xj = x[j]
                    r = (xi - xj).norm()
                    if r < dh:
                        rho += rest_volume[j] * kernel_W(r)
            density[i] = rho * rho0


@ti.kernel
def compute_alpha():
    for i in range(n_particles):
        if is_fluid[i] == 1:
            sum_grad_pk = 0.0
            grad_pi = ti.Vector.zero(ti.f32, dim)
            xi = x[i]
            for j in range(n_particles):
                if j == i:
                    continue
                xj = x[j]
                R = xi - xj
                if R.norm() < dh:
                    grad_pj = -rest_volume[j] * kernel_grad(R)
                    if is_fluid[j] == 1:
                        sum_grad_pk += grad_pj.norm_sqr()
                        grad_pi += grad_pj
                    else:
                        # rigid neighbor
                        grad_pi += grad_pj
            sum_grad_pk += grad_pi.norm_sqr()
            factor = 0.0
            if sum_grad_pk > 1e-5:
                factor = 1.0 / sum_grad_pk
            alpha[i] = factor


@ti.kernel
def compute_density_star():
    for i in range(n_particles):
        if is_fluid[i] == 1:
            delta = 0.0
            xi = x[i]
            vi = v[i]
            for j in range(n_particles):
                if j == i:
                    continue
                xj = x[j]
                vj = v[j]
                R = xi - xj
                if R.norm() < dh:
                    delta += rest_volume[j] * (vi - vj).dot(kernel_grad(R))
            density_adv = density[i] / rho0 + dt * delta
            density_star[i] = max(density_adv, 1.0)


@ti.kernel
def compute_kappa():
    dt_inv = 1 / dt
    for i in range(n_particles):
        if is_fluid[i] == 1:
            kappa[i] = (density_star[i] - 1.0) * alpha[i] * dt_inv


@ti.kernel
def correct_density_error_step():
    for i in range(n_particles):
        if is_fluid[i] == 1:
            ki = kappa[i]
            xi = x[i]
            rhoi = density[i]
            for j in range(n_particles):
                if j == i:
                    continue
                xj = x[j]
                R = xi - xj
                if R.norm() < dh:
                    grad_pj = rest_volume[j] * kernel_grad(R)
                    if is_fluid[j] == 1:
                        kj = kappa[j]
                        ksum = ki + kj
                        if abs(ksum) > eps * dt:
                            rhoj = density[j]
                            v[i] -= grad_pj * (ki / rhoi + kj / rhoj) * rho0
                    else:
                        # rigid neighbor
                        ksum = ki
                        if abs(ksum) > eps * dt:
                            v[i] -= grad_pj * (ki / rhoi) * rho0
                            force_j = (
                                grad_pj
                                * (ki / rhoi)
                                * rho0
                                * (rest_volume[i] * rho0)
                                / dt
                            )
                            ti.atomic_add(rb_force[None], force_j)


@ti.kernel
def compute_density_error() -> ti.f32:
    err = 0.0
    cnt = 0
    for i in range(n_particles):
        if is_fluid[i] == 1:
            err += density_star[i] - 1.0
            cnt += 1
    if cnt > 0:
        err /= cnt
    return err


def correct_density_error():
    compute_density_star()
    num_iter = 0
    avg_err = 0.0
    while num_iter < 1 or num_iter < max_iter_density:
        compute_kappa()
        correct_density_error_step()
        compute_density_star()
        avg_err = compute_density_error()
        if avg_err <= max_error:
            break
        num_iter += 1
    print(f"[DFSPH density] iters={num_iter}, avg_err={avg_err*rho0:.4f}")


@ti.kernel
def compute_density_derivative():
    for i in range(n_particles):
        if is_fluid[i] == 1:
            xi = x[i]
            vi = v[i]
            d_adv = 0.0
            n_nbr = 0
            for j in range(n_particles):
                if j == i:
                    continue
                xj = x[j]
                vj = v[j]
                R = xi - xj
                if R.norm() < dh:
                    d_adv += rest_volume[j] * (vi - vj).dot(kernel_grad(R))
                    n_nbr += 1
            d_adv = max(d_adv, 0.0)
            if dim == 3:
                if n_nbr < 20:
                    d_adv = 0.0
            density_deriv[i] = d_adv


@ti.kernel
def compute_kappa_v():
    max_kappa = 0.0
    for i in range(n_particles):
        if is_fluid[i] == 1:
            kappa_v[i] = density_deriv[i] * alpha[i]
            if kappa_v[i] > max_kappa:
                max_kappa = kappa_v[i]


@ti.kernel
def correct_divergence_step():
    for i in range(n_particles):
        if is_fluid[i] == 1:
            ki = kappa_v[i]
            if abs(ki) < eps:
                continue
            xi = x[i]
            rhoi = density[i]
            dv = ti.Vector.zero(ti.f32, dim)
            for j in range(n_particles):
                if j == i:
                    continue
                xj = x[j]
                R = xi - xj
                if R.norm() < dh:
                    grad_pj = rest_volume[j] * kernel_grad(R)
                    if is_fluid[j] == 1:
                        kj = kappa_v[j]
                        ksum = ki + kj
                        if abs(ksum) > eps * dt:
                            rhoj = density[j]
                            dv -= grad_pj * (ki / rhoi + kj / rhoj) * rho0
                    else:
                        ksum = ki
                        if abs(ksum) > eps * dt:
                            dv -= grad_pj * (ki / rhoi) * rho0
                            force_j = (
                                grad_pj
                                * (ki / rhoi)
                                * rho0
                                * (rest_volume[i] * rho0)
                                / dt
                            )
                            ti.atomic_add(rb_force[None], force_j)
            v[i] += dv


@ti.kernel
def compute_divergence_error() -> ti.f32:
    err = 0.0
    cnt = 0
    for i in range(n_particles):
        if is_fluid[i] == 1:
            err += rho0 * density_deriv[i]
            cnt += 1
    if cnt > 0:
        err /= cnt
    return err


def correct_divergence_error():
    compute_density_derivative()
    num_iter = 0
    avg_err = 0.0
    while num_iter < 1 or num_iter < max_iter_div:
        compute_kappa_v()
        correct_divergence_step()
        compute_density_derivative()
        avg_err = compute_divergence_error()
        eta = max_error_V * rho0 / dt
        if avg_err <= eta:
            break
        num_iter += 1
    print(f"[DFSPH div] iters={num_iter}, avg_err={avg_err:.4f}")


# ==========================
# Non-pressure
# ==========================
@ti.kernel
def compute_non_pressure_acceleration():
    for i in range(n_particles):
        ai = ti.Vector.zero(ti.f32, dim)

        if is_fluid[i] == 1:
            ai = g
            xi = x[i]
            vi = v[i]

            for j in range(n_particles):
                if j == i:
                    continue

                xj = x[j]
                R = xi - xj
                r = R.norm()

                if r < dh:
                    dia2 = dx * dx
                    R2 = ti.math.dot(R, R)
                    if R2 > dia2:
                        ai -= 0.01 / rest_volume[i] * rest_volume[j] * R * kernel_W(r)
                    else:
                        ai -= 0.01 / rest_volume[i] * rest_volume[j] * R * kernel_W(dx)

                    if is_fluid[j] == 1:
                        vj = v[j]
                        vij = vj - vi
                        ai += 5 * rest_volume[j] * vij * kernel_W(r)

        a[i] = ai


@ti.kernel
def update_fluid_velocity():
    for i in range(n_particles):
        if is_fluid[i] == 1:
            v[i] += dt * a[i]


@ti.kernel
def update_fluid_position():
    for i in range(n_particles):
        if is_fluid[i] == 1:
            x[i] += dt * v[i]


@ti.kernel
def rigid_step():
    vel = rb_vel[None]
    pos = rb_pos[None]
    force = rb_force[None]

    vel += dt * (force / rigid_mass + g)
    pos += dt * vel

    for k in ti.static(range(dim)):
        if pos[k] < domain_min[k] + rigid_half[k]:
            pos[k] = domain_min[k] + rigid_half[k]
            vel[k] *= -rigid_restitution
        if pos[k] > domain_max[k] - rigid_half[k]:
            pos[k] = domain_max[k] - rigid_half[k]
            vel[k] *= -rigid_restitution

    rb_vel[None] = vel
    rb_pos[None] = pos
    rb_force[None] = ti.Vector.zero(ti.f32, dim)


@ti.kernel
def renew_rigid_particles():
    c = rb_pos[None]
    for k in range(n_rigid):
        pid = n_fluid + k
        x[pid] = c + rb_local[k]
        v[pid] = rb_vel[None]


@ti.kernel
def enforce_boundary():
    for i in range(n_particles):
        if is_fluid[i] == 1:
            p = x[i]
            vi = v[i]
            for k in ti.static(range(dim)):
                if p[k] < domain_min[k]:
                    p[k] = domain_min[k]
                    vi[k] *= -0.3
                if p[k] > domain_max[k]:
                    p[k] = domain_max[k]
                    vi[k] *= -0.3
            x[i] = p
            v[i] = vi


def compute_rigid_particle_volume():
    pass


# ==========================
# Simulation step
# ==========================
def step():
    compute_non_pressure_acceleration()
    update_fluid_velocity()
    correct_density_error()

    update_fluid_position()

    rigid_step()
    renew_rigid_particles()

    enforce_boundary()
    compute_density()
    compute_alpha()
    correct_divergence_error()
    compute_rigid_particle_volume()


# ==========================
# Main
# ==========================
def main():
    os.makedirs("frames", exist_ok=True)
    os.makedirs("frames/test", exist_ok=True)

    init_fluid_and_rigid()
    init_cube_indices()
    compute_density()
    compute_alpha()

    window = ti.ui.Window("DFSPH + Rigid Demo", (1280, 720), show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()

    n_frames = 600
    substeps = 10

    for frame in range(n_frames):
        for _ in range(substeps):
            step()

        camera = ti.ui.Camera()
        camera.position(1.6, 1.0, 1.6)
        camera.lookat(0.5, 0.4, 0.5)
        camera.up(0.0, 1.0, 0.0)
        scene.set_camera(camera)
        scene.ambient_light((0.4, 0.4, 0.4))
        scene.point_light((2.0, 3.0, 2.0), (1.0, 1.0, 1.0))

        scene.particles(x, radius=0.01, index_count=n_fluid, color=(0.2, 0.6, 1.0))

        update_cube_vertices()
        scene.mesh(cube_vertices, indices=cube_indices, color=(1.0, 0.5, 0.2))

        canvas.scene(scene)

        fname = f"frames/test/frame_{frame:04d}.png"
        window.save_image(fname)
        print("saved", fname)
    os.system(
        "ffmpeg -framerate 30 -i frames/test/frame_%04d.png -c:v libx264 -pix_fmt yuv420p sph_rigid_3d.mp4 -y"
    )


if __name__ == "__main__":
    main()
