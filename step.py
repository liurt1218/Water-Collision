# step.py
import taichi as ti
import config as C
import state as S
import materials as M
import rigid


@ti.kernel
def substep_mpm(gravity: float):
    """
    One MPM substep:
      1. Clear grid
      2. P2G: particle -> grid
      3. Grid operations (gravity, boundary)
      4. G2P: grid -> particle
    """
    dim = ti.static(C.dim)

    # Per-material tables (Python lists, read via ti.static)
    rho0_list = ti.static(M.rho0_table)
    E_list = ti.static(M.E_table)
    nu_list = ti.static(M.nu_table)
    kind_list = ti.static(M.kind_table)
    n_materials = ti.static(len(M.rho0_table))

    # 1. clear grid
    for I in ti.grouped(S.grid_m):
        S.grid_v[I] = ti.Vector.zero(float, dim)
        S.grid_m[I] = 0.0
        S.grid_pressure[I] = 0.0
        S.grid_pressure_w[I] = 0.0

    # 2. P2G
    for p in S.x:
        if S.is_used[p] == 1:
            Xp = S.x[p]
            Vp = S.v[p]
            Cp = S.C_apic[p]

            base = (Xp * C.inv_dx - 0.5).cast(int)
            fx = Xp * C.inv_dx - base.cast(float)

            # Quadratic B-spline weights (vector form)
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]

            # Update deformation gradient
            S.F[p] = (ti.Matrix.identity(float, dim) + C.dt * Cp) @ S.F[p]

            # Per-material physical parameters: rho0, E, nu -> mu, lambda, p_mass
            mat_id = S.materials[p]

            rho0 = 0.0
            E = 0.0
            nu = 0.0
            kind = 0
            # static loop: safely index Python lists with compile-time mid
            for mid in ti.static(range(n_materials)):
                if mat_id == mid:
                    rho0 = rho0_list[mid]
                    E = E_list[mid]
                    nu = nu_list[mid]
                    kind = kind_list[mid]

            mu = E / (2.0 * (1.0 + nu))
            la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

            # particle mass for this material
            p_mass = rho0 * C.p_vol

            # Additional hardening-like factor h (existing logic)
            h = ti.exp(10.0 * (1.0 - S.Jp[p]))
            if kind == C.JELLY:
                h = 0.3
            mu = mu * h
            la = la * h
            if kind == C.WATER:
                # water: no shear
                mu = 0.0

            # SVD of F
            U, sig, V_svd = ti.svd(S.F[p])
            J = 1.0
            for d in ti.static(range(dim)):
                new_sig = sig[d, d]
                if kind == C.SNOW:
                    new_sig = ti.min(
                        ti.max(sig[d, d], 1.0 - 2.5e-2),
                        1.0 + 4.5e-3,
                    )
                S.Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig

            # Different material behavior
            if kind == C.WATER:
                # liquid: keep only volumetric part
                S.F[p] = ti.Matrix.identity(float, 3) * (J ** (1.0 / 3.0))
            elif kind == C.SNOW:
                S.F[p] = U @ sig @ V_svd.transpose()

            # First Piola-Kirchhoff stress (corotated elasticity + volume)
            P = 2 * mu * (S.F[p] - U @ V_svd.transpose()) @ S.F[
                p
            ].transpose() + ti.Matrix.identity(float, 3) * la * J * (J - 1.0)

            trace_P = P[0, 0] + P[1, 1] + P[2, 2]
            p_particle = -trace_P / dim

            stress = (-C.dt * C.p_vol * 4.0 * C.inv_dx * C.inv_dx) * P
            affine = stress + p_mass * Cp

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                node = base + offset
                # boundary check
                if (
                    0 <= node[0] < C.n_grid
                    and 0 <= node[1] < C.n_grid
                    and 0 <= node[2] < C.n_grid
                ):
                    dpos = (offset.cast(float) - fx) * C.dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    ti.atomic_add(
                        S.grid_v[node],
                        weight * (p_mass * Vp + affine @ dpos),
                    )
                    ti.atomic_add(S.grid_m[node], weight * p_mass)
                    ti.atomic_add(S.grid_pressure[node], weight * p_particle)
                    ti.atomic_add(S.grid_pressure_w[node], weight)

    # 3. Grid operations: normalize, gravity, boundary
    for I in ti.grouped(S.grid_m):
        m = S.grid_m[I]
        if m > 0.0:
            v_new = S.grid_v[I] / m
            # gravity on y
            v_new[1] += C.dt * gravity

            # Grid node position in world space
            x_world = (I.cast(float) + 0.5) * C.dx

            # Simple outer box boundary
            for d in ti.static(range(dim)):
                if I[d] < 3 and v_new[d] < 0:
                    v_new[d] = 0
                if I[d] > C.n_grid - 3 and v_new[d] > 0:
                    v_new[d] = 0

            p_node = 0.0
            if S.grid_pressure_w[I] > 0.0:
                p_node = S.grid_pressure[I] / S.grid_pressure_w[I]
                # Optionally clamp: only compressive pressure
                if p_node < 0.0:
                    p_node = 0.0

            # Rigid body coupling via triangle mesh
            contact_band = 0.75 * C.dx  # how far from surface we start reacting

            for r in ti.static(range(S.N_RIGID)):
                if S.rb_active[r] == 1:
                    hit, n, dist = rigid.query_mesh_contact(x_world, r, contact_band)

                    if hit and dist < contact_band:
                        # Rigid body velocity at contact point
                        p_r = S.rb_pos[r]
                        rel = x_world - p_r
                        v_lin = S.rb_lin_vel[r]
                        w = S.rb_ang_vel[r]
                        v_rb = v_lin + w.cross(rel)

                        area = C.dx * C.dx

                        F_pressure = -p_node * area * n * 5

                        ti.atomic_add(S.rb_force[r], F_pressure)
                        ti.atomic_add(S.rb_torque[r], rel.cross(F_pressure))

                        # Relative velocity (grid node - rigid)
                        v_rel = v_new - v_rb
                        vn = v_rel.dot(n)

                        # Only handle if grid is moving into the rigid body
                        if vn < 0.0:
                            restitution = S.rb_restitution[r]

                            # store old vel
                            v_old = v_new

                            # collision: push grid node away from rigid
                            v_new = v_new - (1.0 + restitution) * vn * n

                            mass_grid = m

                            impulse_grid = mass_grid * (v_new - v_old)

                            impulse = -impulse_grid

                            contact_alpha = 1.0 - dist / contact_band
                            contact_alpha = ti.max(0.0, ti.min(1.0, contact_alpha))

                            couple_strength = 10
                            impulse *= couple_strength * contact_alpha

                            max_impulse = 1.0 * mass_grid
                            imp_norm = impulse.norm()
                            if imp_norm > max_impulse:
                                impulse = impulse * (max_impulse / (imp_norm + 1e-8))

                            ti.atomic_add(S.rb_force[r], impulse / C.dt)
                            ti.atomic_add(S.rb_torque[r], rel.cross(impulse) / C.dt)

            S.grid_v[I] = v_new

    # 4. G2P
    for p in S.x:
        if S.is_used[p] == 1:
            Xp = S.x[p]
            base = (Xp * C.inv_dx - 0.5).cast(int)
            fx = Xp * C.inv_dx - base.cast(float)

            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]

            new_v = ti.Vector.zero(float, dim)
            new_C = ti.Matrix.zero(float, dim, dim)

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                node = base + offset
                if (
                    0 <= node[0] < C.n_grid
                    and 0 <= node[1] < C.n_grid
                    and 0 <= node[2] < C.n_grid
                ):
                    dpos = offset.cast(float) - fx
                    g_v = S.grid_v[node]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    new_v += weight * g_v
                    new_C += 4.0 * C.inv_dx * weight * g_v.outer_product(dpos)

            S.v[p] = new_v
            S.C_apic[p] = new_C
            S.x[p] += C.dt * new_v


def substep(gravity: float):
    # mpm step
    substep_mpm(gravity)

    if S.N_RIGID > 0:
        # integrate
        rigid.integrate_rigid_bodies(gravity)

        rigid.update_all_mesh_vertices()

        rigid.handle_rigid_domain_walls()

        rigid.handle_rigid_collisions()

        rigid.update_all_mesh_vertices()
