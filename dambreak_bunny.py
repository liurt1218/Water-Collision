import taichi as ti
import math
import os
import numpy as np

ti.init(arch=ti.gpu, device_memory_fraction=0.8)


def load_obj(path):
    verts = []
    faces = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                _, xs, ys, zs = line.strip().split()[:4]
                verts.append([float(xs), float(ys), float(zs)])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                idx = []
                for p in parts:
                    idx.append(int(p.split("/")[0]) - 1)
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) == 4:
                    faces.append([idx[0], idx[1], idx[2]])
                    faces.append([idx[0], idx[2], idx[3]])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


# ==========================
# Global configuration
# ==========================
dim = 3

# Fluid block size
x, y, z = 0.96, 0.96, 0.24

# Particle diameter
particle_diameter = 0.015

# Total number of particles
nx, ny, nz = (
    int(x / particle_diameter),
    int(y / particle_diameter),
    int(z / particle_diameter),
)
n_fluid = nx * ny * nz
# Support radius
support_radius = 2 * particle_diameter

# Fluid params
rho0 = 1000.0
surface_tension = 0.04
viscosity = 5

mesh_verts_np, mesh_faces_np = load_obj("bunny.obj")
n_mesh_verts = mesh_verts_np.shape[0]
n_mesh_faces = mesh_faces_np.shape[0]
mesh_indices_np = mesh_faces_np.reshape(-1)

mesh_local = ti.Vector.field(3, ti.f32, shape=n_mesh_verts)
mesh_vertices = ti.Vector.field(3, ti.f32, shape=n_mesh_verts)
mesh_indices = ti.field(ti.i32, shape=3 * n_mesh_faces)

# Rigidbody particles
rb_nx, rb_ny, rb_nz = 4, 4, 4
n_rigid_per_body = rb_nx * rb_ny * rb_nz
n_rigid_bodies = 2
n_rigid_total = n_rigid_per_body * n_rigid_bodies

n_particles = n_fluid + n_rigid_total

# Time step
dt = 1e-3

g = ti.Vector([0.0, -9.81, 0.0])

# DFSPH params
max_iter_density = 1000
max_iter_div = 1000
max_error = 1e-4
max_error_V = 1e-3
eps = 1e-5

# Boundaries
domain_min = ti.Vector([0.0, 0.0, 0.0])
domain_max = ti.Vector([1.0, 1.0, 1.0])

# Rigidbody
rigid_half = ti.Vector([0.12, 0.12, 0.12])
rigid_rho = 300.0
rigid_mass = rigid_rho * rigid_half[0] * rigid_half[1] * rigid_half[2] * 8
rigid_restitution = 0.3
mu_friction = 0.3

# ==========================
# Fields
# ==========================
x = ti.Vector.field(dim, ti.f32, shape=n_particles)  # Pos
v = ti.Vector.field(dim, ti.f32, shape=n_particles)  # Vel
a = ti.Vector.field(dim, ti.f32, shape=n_particles)  # Acc
is_fluid = ti.field(ti.i32, shape=n_particles)  # 1 = fluid, 0 = rigid
is_dynamic = ti.field(ti.i32, shape=n_particles)
rest_volume = ti.field(ti.f32, shape=n_particles)
density = ti.field(ti.f32, shape=n_particles)

# DFSPH
alpha = ti.field(ti.f32, shape=n_particles)
density_star = ti.field(ti.f32, shape=n_particles)
density_deriv = ti.field(ti.f32, shape=n_particles)
kappa = ti.field(ti.f32, shape=n_particles)
kappa_v = ti.field(ti.f32, shape=n_particles)

# Rigidbody translation
rb_pos = ti.Vector.field(dim, ti.f32, shape=n_rigid_bodies)
rb_vel = ti.Vector.field(dim, ti.f32, shape=n_rigid_bodies)
rb_force = ti.Vector.field(dim, ti.f32, shape=n_rigid_bodies)

# Rotation
rb_omega = ti.Vector.field(dim, ti.f32, shape=n_rigid_bodies)
rb_torque = ti.Vector.field(dim, ti.f32, shape=n_rigid_bodies)
rb_rot = ti.Matrix.field(dim, dim, ti.f32, shape=n_rigid_bodies)
I_body = ti.Matrix.field(dim, dim, ti.f32, shape=n_rigid_bodies)
I_body_inv = ti.Matrix.field(dim, dim, ti.f32, shape=n_rigid_bodies)

rb_local = ti.Vector.field(dim, ti.f32, shape=n_rigid_per_body)

# Map particles to bodies
rigid_id = ti.field(ti.i32, shape=n_particles)

# Rendering the cube
cube_local = ti.Vector.field(3, ti.f32, shape=8)
cube_vertices = ti.Vector.field(3, ti.f32, shape=8)
cube_indices = ti.field(ti.i32, shape=36)  # Triangular mesh


# ==========================
# Kernels
# ==========================
@ti.func
def kernel_W(r):
    h_ = ti.static(support_radius)
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
    h_ = ti.static(support_radius)

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
# Initialize
# ==========================
@ti.kernel
def init_mesh():
    for i in range(n_mesh_verts):
        mesh_local[i] = ti.Vector(mesh_verts_np[i])  # local space 顶点
    for f in range(n_mesh_faces):
        mesh_indices[3 * f + 0] = mesh_faces_np[f, 0]
        mesh_indices[3 * f + 1] = mesh_faces_np[f, 1]
        mesh_indices[3 * f + 2] = mesh_faces_np[f, 2]


@ti.kernel
def init_fluid_and_rigid():
    # Fluid block
    base = ti.Vector([0.02, 0.02, 0.02])
    for i in range(n_fluid):
        ix = i % nx
        iy = (i // nx) % ny
        iz = i // (nx * ny)
        x[i] = base + ti.Vector([ix + 0.5, iy + 0.5, iz + 0.5]) * particle_diameter
        v[i] = ti.Vector.zero(ti.f32, dim)
        a[i] = ti.Vector.zero(ti.f32, dim)
        is_fluid[i] = 1
        is_dynamic[i] = 1
        rest_volume[i] = particle_diameter**3
        rigid_id[i] = -1

    # Rigid cube
    rb_dx = 2.0 * rigid_half[0] / rb_nx
    lb = -rigid_half + 0.5 * rb_dx
    for k in range(n_rigid_per_body):
        ix = k % rb_nx
        iy = (k // rb_nx) % rb_ny
        iz = k // (rb_nx * rb_ny)
        rb_local[k] = lb + ti.Vector([ix, iy, iz]) * rb_dx

    # Rigidbodies
    for b in range(n_rigid_bodies):
        center = ti.Vector([0.5, rigid_half[1] + 0.3 * b, 0.5])
        rb_pos[b] = center
        rb_vel[b] = ti.Vector.zero(ti.f32, dim)
        rb_force[b] = ti.Vector.zero(ti.f32, dim)
        rb_omega[b] = ti.Vector.zero(ti.f32, dim)
        rb_torque[b] = ti.Vector.zero(ti.f32, dim)

        box_volume = 2 * rigid_half[0] * 2 * rigid_half[1] * 2 * rigid_half[2]

        for k in range(n_rigid_per_body):
            pid = n_fluid + b * n_rigid_per_body + k
            local = rb_local[k]
            x[pid] = center + local
            v[pid] = ti.Vector.zero(ti.f32, dim)
            a[pid] = ti.Vector.zero(ti.f32, dim)
            is_fluid[pid] = 0
            is_dynamic[pid] = 1
            rest_volume[pid] = box_volume / n_rigid_per_body
            rigid_id[pid] = b


@ti.kernel
def init_rigid_orientation_and_cube():
    a = 2.0 * rigid_half[0]
    b = 2.0 * rigid_half[1]
    c = 2.0 * rigid_half[2]

    Ixx = (1.0 / 12.0) * rigid_mass * (b * b + c * c)
    Iyy = (1.0 / 12.0) * rigid_mass * (a * a + c * c)
    Izz = (1.0 / 12.0) * rigid_mass * (a * a + b * b)

    for body in range(n_rigid_bodies):
        I_body[body] = ti.Matrix([[Ixx, 0.0, 0.0], [0.0, Iyy, 0.0], [0.0, 0.0, Izz]])
        I_body_inv[body] = I_body[body].inverse()

        rb_rot[body] = ti.Matrix.identity(ti.f32, 3)
        rb_omega[body] = ti.Vector.zero(ti.f32, 3)
        rb_torque[body] = ti.Vector.zero(ti.f32, 3)

    hx, hy, hz = rigid_half[0], rigid_half[1], rigid_half[2]
    cube_local[0] = ti.Vector([-hx, -hy, -hz])
    cube_local[1] = ti.Vector([+hx, -hy, -hz])
    cube_local[2] = ti.Vector([+hx, +hy, -hz])
    cube_local[3] = ti.Vector([-hx, +hy, -hz])
    cube_local[4] = ti.Vector([-hx, -hy, +hz])
    cube_local[5] = ti.Vector([+hx, -hy, +hz])
    cube_local[6] = ti.Vector([+hx, +hy, +hz])
    cube_local[7] = ti.Vector([-hx, +hy, +hz])


@ti.kernel
def init_cube_indices():
    # front
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
def update_mesh_vertices(body: ti.i32):
    c = rb_pos[body]
    R = rb_rot[body]
    for i in range(n_mesh_verts):
        mesh_vertices[i] = c + R @ mesh_local[i]


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
                    if r < support_radius:
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
                if R.norm() < support_radius:
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
                if R.norm() < support_radius:
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
                if R.norm() < support_radius:
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
                            rb = rigid_id[j]
                            ti.atomic_add(rb_force[rb], force_j)
                            r = x[j] - rb_pos[rb]
                            torque_j = r.cross(force_j)
                            ti.atomic_add(rb_torque[rb], torque_j)


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
                if R.norm() < support_radius:
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
                if R.norm() < support_radius:
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
                            rb = rigid_id[j]
                            ti.atomic_add(rb_force[rb], force_j)
                            r = x[j] - rb_pos[rb]
                            torque_j = r.cross(force_j)
                            ti.atomic_add(rb_torque[rb], torque_j)
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
# Non-pressure acceleration
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

                if r < support_radius:
                    dia2 = particle_diameter * particle_diameter
                    R2 = ti.math.dot(R, R)
                    if R2 > dia2:
                        ai -= (
                            surface_tension
                            / rest_volume[i]
                            * rest_volume[j]
                            * R
                            * kernel_W(r)
                        )
                    else:
                        ai -= (
                            surface_tension
                            / rest_volume[i]
                            * rest_volume[j]
                            * R
                            * kernel_W(particle_diameter)
                        )
                    if is_fluid[j] == 1:
                        vj = v[j]
                        vij = vj - vi
                        ai += viscosity * rest_volume[j] * vij * kernel_W(r)
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


@ti.func
def skew(v):
    return ti.Matrix([[0.0, -v.z, v.y], [v.z, 0.0, -v.x], [-v.y, v.x, 0.0]])


@ti.func
def orthonormalize(R):
    c0 = R[:, 0]
    c1 = R[:, 1]
    c2 = R[:, 2]

    c0 = c0.normalized()
    c1 = (c1 - c0.dot(c1) * c0).normalized()
    c2 = c0.cross(c1)

    return ti.Matrix.cols([c0, c1, c2])


@ti.func
def support_distance_on_box(R, n_world):
    n_local = R.transpose() @ n_world
    half = rigid_half
    return (
        ti.abs(n_local[0]) * half[0]
        + ti.abs(n_local[1]) * half[1]
        + ti.abs(n_local[2]) * half[2]
    )


@ti.func
def contact_offset_on_box(R, dir_world):
    dir_local = R.transpose() @ dir_world
    half = rigid_half

    t_min = 1e8
    for k in ti.static(range(3)):
        ak = dir_local[k]
        if ti.abs(ak) > 1e-6:
            tk = half[k] / ti.abs(ak)
            if tk < t_min:
                t_min = tk

    if t_min > 1e7:
        t_min = 0.0

    contact_local = dir_local * t_min
    contact_world = R @ contact_local
    return contact_world


@ti.kernel
def handle_rigid_collisions():
    inv_m = 1.0 / rigid_mass
    e = rigid_restitution
    mu = mu_friction

    for b1 in range(n_rigid_bodies):
        for b2 in range(b1 + 1, n_rigid_bodies):
            c1 = rb_pos[b1]
            c2 = rb_pos[b2]
            d = c2 - c1
            dist = d.norm()
            if dist <= 1e-6:
                continue
            n = d / dist

            R1 = rb_rot[b1]
            R2 = rb_rot[b2]

            s1 = support_distance_on_box(R1, n)
            s2 = support_distance_on_box(R2, -n)
            min_dist = s1 + s2

            if dist < min_dist:
                penetration = min_dist - dist
                corr = 0.5 * penetration * n
                rb_pos[b1] -= corr
                rb_pos[b2] += corr

                c1 = rb_pos[b1]
                c2 = rb_pos[b2]

                offset1 = contact_offset_on_box(R1, n)
                offset2 = contact_offset_on_box(R2, -n)

                p1 = c1 + offset1
                p2 = c2 + offset2

                dc = p2 - p1
                dc_len = dc.norm()
                if dc_len > 1e-6:
                    n = dc / dc_len

                r1 = p1 - c1
                r2 = p2 - c2

                v1 = rb_vel[b1]
                v2 = rb_vel[b2]
                w1 = rb_omega[b1]
                w2 = rb_omega[b2]

                v1c = v1 + w1.cross(r1)
                v2c = v2 + w2.cross(r2)

                v_rel = v2c - v1c
                v_rel_n = v_rel.dot(n)

                if v_rel_n < 0.0:
                    I1_inv = I_body_inv[b1]
                    I2_inv = I_body_inv[b2]

                    rn1 = r1.cross(n)
                    rn2 = r2.cross(n)

                    ang_term1 = (I1_inv @ rn1).cross(r1).dot(n)
                    ang_term2 = (I2_inv @ rn2).cross(r2).dot(n)
                    denom_n = 2.0 * inv_m + ang_term1 + ang_term2 + 1e-6

                    j_n = -(1.0 + e) * v_rel_n / denom_n
                    J_n = j_n * n

                    rb_vel[b1] -= J_n * inv_m
                    rb_vel[b2] += J_n * inv_m

                    delta_L1_n = r1.cross(-J_n)
                    delta_L2_n = r2.cross(+J_n)
                    rb_omega[b1] += I1_inv @ delta_L1_n
                    rb_omega[b2] += I2_inv @ delta_L2_n

                    v1 = rb_vel[b1]
                    v2 = rb_vel[b2]
                    w1 = rb_omega[b1]
                    w2 = rb_omega[b2]

                    v1c = v1 + w1.cross(r1)
                    v2c = v2 + w2.cross(r2)

                    v_rel = v2c - v1c
                    v_rel_n2 = v_rel.dot(n)
                    v_rel_t = v_rel - v_rel_n2 * n
                    vt_len = v_rel_t.norm()

                    if vt_len > 1e-6:
                        t_dir = v_rel_t / vt_len

                        rt1 = r1.cross(t_dir)
                        rt2 = r2.cross(t_dir)

                        ang_term1_t = (I1_inv @ rt1).cross(r1).dot(t_dir)
                        ang_term2_t = (I2_inv @ rt2).cross(r2).dot(t_dir)
                        denom_t = 2.0 * inv_m + ang_term1_t + ang_term2_t + 1e-6

                        j_t_raw = -vt_len / denom_t

                        max_j_t = mu * ti.abs(j_n)
                        j_t = j_t_raw
                        if j_t > max_j_t:
                            j_t = max_j_t
                        if j_t < -max_j_t:
                            j_t = -max_j_t

                        J_t = j_t * t_dir

                        rb_vel[b1] -= J_t * inv_m
                        rb_vel[b2] += J_t * inv_m

                        delta_L1_t = r1.cross(-J_t)
                        delta_L2_t = r2.cross(+J_t)
                        rb_omega[b1] += I1_inv @ delta_L1_t
                        rb_omega[b2] += I2_inv @ delta_L2_t


@ti.kernel
def rigid_step():
    for b in range(n_rigid_bodies):
        vel = rb_vel[b]
        pos = rb_pos[b]
        force = rb_force[b]

        omega = rb_omega[b]
        torque = rb_torque[b]
        R = rb_rot[b]

        I_b = I_body[b]
        I_b_inv = I_body_inv[b]

        vel += dt * (force / rigid_mass + g)
        pos += dt * vel

        I_world = R @ I_b @ R.transpose()
        I_world_inv = R @ I_b_inv @ R.transpose()

        ang_acc = I_world_inv @ (torque - omega.cross(I_world @ omega))
        omega += dt * ang_acc

        R += dt * skew(omega) @ R
        R = orthonormalize(R)

        for k in ti.static(range(dim)):
            if pos[k] < domain_min[k] + rigid_half[k]:
                pos[k] = domain_min[k] + rigid_half[k]
                vel[k] *= -rigid_restitution
            if pos[k] > domain_max[k] - rigid_half[k]:
                pos[k] = domain_max[k] - rigid_half[k]
                vel[k] *= -rigid_restitution

        rb_vel[b] = vel
        rb_pos[b] = pos
        rb_omega[b] = omega
        rb_rot[b] = R
        rb_force[b] = ti.Vector.zero(ti.f32, dim)
        rb_torque[b] = ti.Vector.zero(ti.f32, dim)


@ti.kernel
def renew_rigid_particles():
    for b in range(n_rigid_bodies):
        c = rb_pos[b]
        R = rb_rot[b]
        omega = rb_omega[b]

        for k in range(n_rigid_per_body):
            pid = n_fluid + b * n_rigid_per_body + k
            local = rb_local[k]
            offset = R @ local
            x[pid] = c + offset
            v[pid] = rb_vel[b] + omega.cross(offset)


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


# ==========================
# Simulation step
# ==========================
def step():
    compute_non_pressure_acceleration()
    update_fluid_velocity()
    correct_density_error()

    update_fluid_position()

    rigid_step()
    handle_rigid_collisions()
    renew_rigid_particles()

    enforce_boundary()
    compute_density()
    compute_alpha()
    correct_divergence_error()


# ==========================
# Main program
# ==========================
def main():
    os.makedirs("frames", exist_ok=True)
    os.makedirs("frames/dambreak_bunny", exist_ok=True)

    global mesh_verts_np

    bbox_min = mesh_verts_np.min(axis=0)
    bbox_max = mesh_verts_np.max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    extent = bbox_max - bbox_min
    max_extent = extent.max() + 1e-8

    target_diameter = 2.0 * float(rigid_half[0])
    scale = target_diameter / max_extent

    mesh_verts_np_scaled = (mesh_verts_np - center) * scale

    mesh_local.from_numpy(mesh_verts_np_scaled.astype(np.float32))
    mesh_indices.from_numpy(mesh_indices_np.astype(np.int32))

    init_fluid_and_rigid()
    init_rigid_orientation_and_cube()
    # init_cube_indices()
    compute_density()
    compute_alpha()

    window = ti.ui.Window("DFSPH + Rigid Demo", (1280, 720), show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()

    n_frames = 400
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

        for b in range(n_rigid_bodies):
            update_mesh_vertices(b)
            scene.mesh(mesh_vertices, indices=mesh_indices, color=(1.0, 0.5, 0.2))

        canvas.scene(scene)

        fname = f"frames/dambreak_bunny/frame_{frame:04d}.png"
        window.save_image(fname)
        print("saved", fname)
    os.system(
        "ffmpeg -framerate 30 -i frames/dambreak_bunny/frame_%04d.png -c:v libx264 -pix_fmt yuv420p dambreak_bunny.mp4 -y"
    )


if __name__ == "__main__":
    main()
