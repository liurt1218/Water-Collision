# sim/fluid.py
import taichi as ti
from . import config as C
from . import state as S


# ================================================================
# SPH kernels
# ================================================================
@ti.func
def kernel_W(r: ti.f32, h_: ti.f32) -> ti.f32:
    """
    Compact-support smoothing kernel.

    Parameters
    ----------
    r : float
        Distance between particles.
    h_ : float
        Support radius for this particle.
    """
    dim_ = ti.static(C.dim)

    res = 0.0
    if 0.0 <= r <= h_:
        k = 0.0
        if dim_ == 3:
            k = 8.0 / ti.math.pi
        elif dim_ == 2:
            k = 40.0 / (7.0 * ti.math.pi)
        else:
            k = 4.0 / 3.0

        k /= h_**dim_

        q = r / h_
        if q <= 1.0:
            if q <= 0.5:
                res = k * (6.0 * q * q * q - 6.0 * q * q + 1.0)
            else:
                res = k * 2.0 * (1.0 - q) ** 3
    return res


@ti.func
def kernel_grad(R: ti.types.vector(C.dim, ti.f32), h_: ti.f32):
    """
    Gradient of the smoothing kernel.

    Parameters
    ----------
    R : Vec
        Vector from neighbor to center particle.
    h_ : float
        Support radius for this particle.
    """
    dim_ = ti.static(C.dim)
    R_mod = R.norm()
    grad = ti.Vector.zero(ti.f32, dim_)

    if 0 < R_mod <= h_:
        k = 0.0
        if dim_ == 3:
            k = 8.0 / ti.math.pi
        elif dim_ == 2:
            k = 40.0 / (7.0 * ti.math.pi)
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


# ================================================================
# Fluid initialization
# ================================================================
@ti.kernel
def _init_fluid_block_kernel(
    p_offset: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    base_x: ti.f32,
    base_y: ti.f32,
    base_z: ti.f32,
    particle_diameter: ti.f32,
    rho0: ti.f32,
    surface_tension: ti.f32,
    viscosity: ti.f32,
    support_radius: ti.f32,
):
    """
    Initialize a single fluid block as a regular 3D grid of particles.

    Parameters
    ----------
    p_offset : int
        Starting index in the global particle arrays for this block.
    nx, ny, nz : int
        Number of particles along x, y, z directions for this block.
    base_x, base_y, base_z : float
        Minimum corner of the block in world space.
    particle_diameter : float
        Spacing between particles (assumed equal in x, y, z).
    rho0, surface_tension, viscosity : float
        Physical parameters for this block.
    support_radius : float
        Kernel support radius for this block.
    """
    base = ti.Vector([base_x, base_y, base_z])
    n_block = nx * ny * nz

    for local in range(n_block):
        i = p_offset + local
        ix = local % nx
        iy = (local // nx) % ny
        iz = local // (nx * ny)

        # Position on a regular grid inside the block
        S.x[i] = base + ti.Vector(
            [ix + 0.5, iy + 0.5, iz + 0.5]
        ) * particle_diameter

        # Initial kinematics
        S.v[i] = ti.Vector.zero(ti.f32, C.dim)
        S.a[i] = ti.Vector.zero(ti.f32, C.dim)

        # Mark as fluid & dynamic
        S.is_fluid[i] = 1
        S.is_dynamic[i] = 1

        # Rest volume is approximated from particle spacing
        S.rest_volume[i] = particle_diameter ** 3

        # Per-particle fluid properties (copied from the block config)
        S.fluid_rho0[i] = rho0
        S.fluid_surface_tension[i] = surface_tension
        S.fluid_viscosity[i] = viscosity
        S.fluid_support_radius[i] = support_radius
        S.fluid_particle_diameter[i] = particle_diameter


def init_fluid_blocks(fluid_scene: C.FluidSceneConfig):
    """
    Python wrapper to initialize all fluid particles for a FluidSceneConfig.

    Assumes S.allocate_fields() has already been called with
    n_fluid = compute_total_fluid_particles(fluid_scene).
    """
    offset = 0
    for cfg in fluid_scene.blocks:
        if not cfg.enabled:
            continue

        sx, sy, sz = cfg.size
        h = cfg.particle_diameter
        nx = int(sx / h)
        ny = int(sy / h)
        nz = int(sz / h)
        n_block = nx * ny * nz

        support_radius = h * cfg.support_radius_scale

        _init_fluid_block_kernel(
            offset,
            nx,
            ny,
            nz,
            cfg.base[0],
            cfg.base[1],
            cfg.base[2],
            h,
            cfg.rho0,
            cfg.surface_tension,
            cfg.viscosity,
            support_radius,
        )
        offset += n_block

    assert offset == S.n_fluid, f"init_fluid_blocks: offset={offset}, n_fluid={S.n_fluid}"


# ================================================================
# DFSPH: density constraint
# ================================================================
@ti.kernel
def compute_density():
    """SPH density summation using per-particle kernel radius and rest density."""
    for i in range(S.n_particles):
        if S.is_fluid[i] == 1:
            hi = S.fluid_support_radius[i]
            rho0_i = S.fluid_rho0[i]

            rho = S.rest_volume[i] * kernel_W(0.0, hi)
            xi = S.x[i]
            for j in range(S.n_particles):
                if j != i:
                    xj = S.x[j]
                    r = (xi - xj).norm()
                    if r < hi:
                        rho += S.rest_volume[j] * kernel_W(r, hi)
            S.density[i] = rho * rho0_i


@ti.kernel
def compute_alpha():
    """Precompute DFSPH alpha for each fluid particle."""
    for i in range(S.n_particles):
        if S.is_fluid[i] == 1:
            hi = S.fluid_support_radius[i]
            sum_grad_pk = 0.0
            grad_pi = ti.Vector.zero(ti.f32, C.dim)
            xi = S.x[i]

            for j in range(S.n_particles):
                if j == i:
                    continue
                xj = S.x[j]
                R = xi - xj
                if R.norm() < hi:
                    grad_pj = -S.rest_volume[j] * kernel_grad(R, hi)
                    if S.is_fluid[j] == 1:
                        sum_grad_pk += grad_pj.norm_sqr()
                        grad_pi += grad_pj
                    else:
                        grad_pi += grad_pj

            sum_grad_pk += grad_pi.norm_sqr()
            factor = 0.0
            if sum_grad_pk > 1e-5:
                factor = 1.0 / sum_grad_pk
            S.alpha[i] = factor


@ti.kernel
def compute_density_star():
    """Predict normalized density using current velocity."""
    for i in range(S.n_particles):
        if S.is_fluid[i] == 1:
            hi = S.fluid_support_radius[i]
            rho0_i = S.fluid_rho0[i]

            delta = 0.0
            xi = S.x[i]
            vi = S.v[i]
            for j in range(S.n_particles):
                if j == i:
                    continue
                xj = S.x[j]
                vj = S.v[j]
                R = xi - xj
                if R.norm() < hi:
                    delta += S.rest_volume[j] * (vi - vj).dot(kernel_grad(R, hi))
            density_adv = S.density[i] / rho0_i + C.dt * delta
            S.density_star[i] = max(density_adv, 1.0)


@ti.kernel
def compute_kappa():
    """Compute Lagrange multipliers for the density constraint."""
    dt_inv = 1.0 / C.dt
    for i in range(S.n_particles):
        if S.is_fluid[i] == 1:
            S.kappa[i] = (S.density_star[i] - 1.0) * S.alpha[i] * dt_inv


@ti.kernel
def correct_density_error_step():
    """Perform one DFSPH density correction step."""
    for i in range(S.n_particles):
        if S.is_fluid[i] == 1:
            hi = S.fluid_support_radius[i]
            rho0_i = S.fluid_rho0[i]

            ki = S.kappa[i]
            xi = S.x[i]
            rhoi = S.density[i]

            for j in range(S.n_particles):
                if j == i:
                    continue
                xj = S.x[j]
                R = xi - xj
                if R.norm() < hi:
                    grad_pj = S.rest_volume[j] * kernel_grad(R, hi)

                    if S.is_fluid[j] == 1:
                        kj = S.kappa[j]
                        ksum = ki + kj
                        if abs(ksum) > C.eps * C.dt:
                            rhoj = S.density[j]
                            S.v[i] -= grad_pj * (ki / rhoi + kj / rhoj) * rho0_i
                    else:
                        ksum = ki
                        if abs(ksum) > C.eps * C.dt:
                            # Solid neighbor: only fluid particle velocity is updated
                            S.v[i] -= grad_pj * (ki / rhoi) * rho0_i

                            # Apply reaction force and torque onto the rigid body
                            force_j = (
                                grad_pj
                                * (ki / rhoi)
                                * rho0_i
                                * (S.rest_volume[i] * rho0_i)
                                / C.dt
                            )
                            rb = S.rigid_id[j]
                            ti.atomic_add(S.rb_force[rb], force_j)
                            r = S.x[j] - S.rb_pos[rb]
                            torque_j = r.cross(force_j)
                            ti.atomic_add(S.rb_torque[rb], torque_j)


@ti.kernel
def compute_density_error() -> ti.f32:
    """Compute average normalized density error."""
    err = 0.0
    cnt = 0
    for i in range(S.n_particles):
        if S.is_fluid[i] == 1:
            err += S.density_star[i] - 1.0
            cnt += 1
    if cnt > 0:
        err /= cnt
    return err


@ti.kernel
def _compute_average_rho0() -> ti.f32:
    """Compute average rest density across all fluid particles."""
    total = 0.0
    cnt = 0
    for i in range(S.n_particles):
        if S.is_fluid[i] == 1:
            total += S.fluid_rho0[i]
            cnt += 1
    if cnt > 0:
        total /= cnt
    return total


def correct_density_error():
    """
    Iteratively enforce the density constraint using DFSPH.

    The stopping criterion is based on the average normalized density error.
    """
    compute_density_star()
    num_iter = 0
    avg_err = 0.0
    rho0_avg = _compute_average_rho0()

    while num_iter < 1 or num_iter < C.max_iter_density:
        compute_kappa()
        correct_density_error_step()
        compute_density_star()
        avg_err = compute_density_error()
        if avg_err <= C.max_error:
            break
        num_iter += 1

    print(
        f"[DFSPH density] iters={num_iter}, "
        f"avg_err={avg_err * rho0_avg:.4f}"
    )


# ================================================================
# DFSPH: divergence constraint
# ================================================================
@ti.kernel
def compute_density_derivative():
    """Compute density time derivative for divergence correction."""
    for i in range(S.n_particles):
        if S.is_fluid[i] == 1:
            hi = S.fluid_support_radius[i]

            xi = S.x[i]
            vi = S.v[i]
            d_adv = 0.0
            n_nbr = 0

            for j in range(S.n_particles):
                if j == i:
                    continue
                xj = S.x[j]
                vj = S.v[j]
                R = xi - xj
                if R.norm() < hi:
                    d_adv += S.rest_volume[j] * (vi - vj).dot(kernel_grad(R, hi))
                    n_nbr += 1

            # Only positive divergence is corrected
            d_adv = max(d_adv, 0.0)
            # Heuristic to avoid over-correction when there are very few neighbors
            if C.dim == 3 and n_nbr < 20:
                d_adv = 0.0
            S.density_deriv[i] = d_adv


@ti.kernel
def compute_kappa_v():
    """Compute Lagrange multipliers for the divergence constraint."""
    for i in range(S.n_particles):
        if S.is_fluid[i] == 1:
            S.kappa_v[i] = S.density_deriv[i] * S.alpha[i]


@ti.kernel
def correct_divergence_step():
    """Perform one DFSPH divergence correction step."""
    for i in range(S.n_particles):
        if S.is_fluid[i] == 1:
            hi = S.fluid_support_radius[i]
            rho0_i = S.fluid_rho0[i]

            ki = S.kappa_v[i]
            if abs(ki) < C.eps:
                continue

            xi = S.x[i]
            rhoi = S.density[i]
            dv = ti.Vector.zero(ti.f32, C.dim)

            for j in range(S.n_particles):
                if j == i:
                    continue
                xj = S.x[j]
                R = xi - xj
                if R.norm() < hi:
                    grad_pj = S.rest_volume[j] * kernel_grad(R, hi)

                    if S.is_fluid[j] == 1:
                        kj = S.kappa_v[j]
                        ksum = ki + kj
                        if abs(ksum) > C.eps * C.dt:
                            rhoj = S.density[j]
                            dv -= grad_pj * (ki / rhoi + kj / rhoj) * rho0_i
                    else:
                        ksum = ki
                        if abs(ksum) > C.eps * C.dt:
                            dv -= grad_pj * (ki / rhoi) * rho0_i

                            # Reaction force on the rigid body
                            force_j = (
                                grad_pj
                                * (ki / rhoi)
                                * rho0_i
                                * (S.rest_volume[i] * rho0_i)
                                / C.dt
                            )
                            rb = S.rigid_id[j]
                            ti.atomic_add(S.rb_force[rb], force_j)
                            r = S.x[j] - S.rb_pos[rb]
                            torque_j = r.cross(force_j)
                            ti.atomic_add(S.rb_torque[rb], torque_j)

            S.v[i] += dv


@ti.kernel
def compute_divergence_error() -> ti.f32:
    """Compute average divergence error in physical units."""
    err = 0.0
    cnt = 0
    for i in range(S.n_particles):
        if S.is_fluid[i] == 1:
            rho0_i = S.fluid_rho0[i]
            err += rho0_i * S.density_deriv[i]
            cnt += 1
    if cnt > 0:
        err /= cnt
    return err


def correct_divergence_error():
    """
    Iteratively enforce the divergence-free condition using DFSPH.

    The stopping criterion is based on the average divergence error
    compared against a tolerance scaled by the average rest density.
    """
    compute_density_derivative()
    num_iter = 0
    avg_err = 0.0
    rho0_avg = _compute_average_rho0()

    while num_iter < 1 or num_iter < C.max_iter_div:
        compute_kappa_v()
        correct_divergence_step()
        compute_density_derivative()
        avg_err = compute_divergence_error()
        eta = C.max_error_V * rho0_avg / C.dt
        if avg_err <= eta:
            break
        num_iter += 1

    print(f"[DFSPH div] iters={num_iter}, avg_err={avg_err:.4f}")


# ================================================================
# Non-pressure forces and integration
# ================================================================
@ti.kernel
def compute_non_pressure_acceleration():
    """
    Compute non-pressure accelerations (gravity, surface tension, viscosity)
    using per-particle fluid parameters.
    """
    for i in range(S.n_particles):
        ai = ti.Vector.zero(ti.f32, C.dim)

        if S.is_fluid[i] == 1:
            # Gravity
            ai = ti.Vector(C.g)

            xi = S.x[i]
            vi = S.v[i]

            hi = S.fluid_support_radius[i]
            di = S.fluid_particle_diameter[i]
            st_i = S.fluid_surface_tension[i]
            mu_i = S.fluid_viscosity[i]

            for j in range(S.n_particles):
                if j == i:
                    continue

                xj = S.x[j]
                R = xi - xj
                r = R.norm()

                if r < hi:
                    # Surface tension term
                    dia2 = di * di
                    R2 = ti.math.dot(R, R)
                    if R2 > dia2:
                        ai -= (
                            st_i
                            / S.rest_volume[i]
                            * S.rest_volume[j]
                            * R
                            * kernel_W(r, hi)
                        )
                    else:
                        ai -= (
                            st_i
                            / S.rest_volume[i]
                            * S.rest_volume[j]
                            * R
                            * kernel_W(di, hi)
                        )

                    # Viscosity term (fluid-fluid interaction only)
                    if S.is_fluid[j] == 1:
                        vj = S.v[j]
                        vij = vj - vi
                        ai += mu_i * S.rest_volume[j] * vij * kernel_W(r, hi)

        S.a[i] = ai


@ti.kernel
def update_fluid_velocity():
    """Explicit Euler velocity update for fluid particles."""
    for i in range(S.n_particles):
        if S.is_fluid[i] == 1:
            S.v[i] += C.dt * S.a[i]


@ti.kernel
def update_fluid_position():
    """Explicit Euler position update for fluid particles."""
    for i in range(S.n_particles):
        if S.is_fluid[i] == 1:
            S.x[i] += C.dt * S.v[i]


# ================================================================
# Boundary
# ================================================================
@ti.kernel
def enforce_boundary():
    """Simple axis-aligned box boundary for fluid particles."""
    for i in range(S.n_particles):
        if S.is_fluid[i] == 1:
            p = S.x[i]
            vi = S.v[i]
            for k in ti.static(range(C.dim)):
                if p[k] < C.domain_min[k]:
                    p[k] = C.domain_min[k]
                    vi[k] *= -0.3
                if p[k] > C.domain_max[k]:
                    p[k] = C.domain_max[k]
                    vi[k] *= -0.3
            S.x[i] = p
            S.v[i] = vi
