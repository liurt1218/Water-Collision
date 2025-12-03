# step.py
import taichi as ti
import config as C
import state as S
import materials as M
import rigid


@ti.kernel
def clear_cdf_grid():
    """
    Reset all CDF-related grid fields.

    grid_dist   = large positive value : no nearby rigid.
    grid_rigid  = -1                   : no rigid associated.
    grid_side   = 0                    : no inside/outside info.
    """
    for I in ti.grouped(S.grid_side):
        S.grid_dist[I] = 1e6
        S.grid_rigid[I] = -1
        S.grid_side[I] = 0
        S.grid_normal[I] = ti.Vector.zero(float, 3)


@ti.kernel
def rasterize_rigid_boundary_cdf(contact_band: float):
    """
    For each grid node, query all rigid meshes and record the nearest one
    using signed distance.

    Writes:
      grid_dist[I]    : signed distance to nearest rigid ( >0 outside, <0 inside )
      grid_rigid[I]   : id of the nearest rigid (or -1 if none)
      grid_side[I]    : -1 (inside), 0 (no rigid), +1 (outside)
      grid_normal[I]  : outward normal of that rigid at this node (solid -> fluid),
                        or (0,0,0) if no rigid.
    """
    for I in ti.grouped(S.grid_side):
        x_world = (I.cast(float) + 0.5) * C.dx

        best_abs = contact_band
        best_signed = 1e6
        best_rigid = -1
        best_side = ti.i8(0)
        best_n = ti.Vector.zero(float, 3)

        for r in range(S.N_RIGID):
            if S.rb_active[r] == 1:
                hit, n, dist = rigid.query_mesh_contact_strict_cdf(
                    x_world, r, contact_band
                )

                if hit and ti.abs(dist) < best_abs:
                    best_abs = ti.abs(dist)
                    best_signed = dist
                    best_rigid = r
                    best_side = ti.i8(1)
                    if dist < 0.0:
                        best_side = ti.i8(-1)

                    if n.norm() > 1e-6:
                        best_n = n / n.norm()
                    else:
                        best_n = ti.Vector([0.0, 1.0, 0.0])

        if best_rigid >= 0:
            S.grid_dist[I] = best_signed
            S.grid_rigid[I] = best_rigid
            S.grid_side[I] = best_side
            S.grid_normal[I] = best_n
        else:
            S.grid_dist[I] = 1e6
            S.grid_rigid[I] = -1
            S.grid_side[I] = 0
            S.grid_normal[I] = ti.Vector.zero(float, 3)


# deprecated
@ti.kernel
def gather_boundary_from_grid(contact_band: float):
    """
    For each particle, gather boundary information from nearby grid nodes.

    For each particle p, we find the closest rigid surface in the grid-based
    signed distance field and compute a contact normal via central differences.

    Writes:
      near_boundary[p]      : 1 if a rigid is within contact_band, else 0
      boundary_rigid_id[p]  : id of the nearest rigid (or -1)
      boundary_dist[p]      : positive distance to that rigid (<= contact_band)
      boundary_normal[p]    : contact normal pointing from rigid to particle
      boundary_side[p]      : -1 (inside), 0 (no rigid), +1 (outside)
    """
    for p in range(C.n_particles):
        if S.is_used[p] == 0:
            S.near_boundary[p] = 0
            S.boundary_rigid_id[p] = -1
            S.boundary_dist[p] = 1e6  # deprecated
            S.boundary_normal[p] = ti.Vector.zero(float, 3)
            S.boundary_side[p] = ti.i8(0)
            S.p_cdf_states[p] = ti.u64(0)  # unused, keep cleared
            continue

        Xp = S.x[p]
        pos_grid = Xp * C.inv_dx
        base = (pos_grid - 0.5).cast(int)

        best_abs = contact_band
        best_signed = 1e6
        best_node = ti.Vector.zero(int, 3)
        best_rigid = -1

        # search in a 3x3x3 grid neighborhood
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            node = base + ti.Vector([i, j, k])
            if (
                0 <= node[0] < C.n_grid
                and 0 <= node[1] < C.n_grid
                and 0 <= node[2] < C.n_grid
            ):
                d = S.grid_dist[node]  # signed distance at node
                rid = S.grid_rigid[node]
                if rid >= 0:
                    abs_d = ti.abs(d)
                    if abs_d < best_abs:
                        best_abs = abs_d
                        best_signed = d
                        best_node = node
                        best_rigid = rid

        if best_rigid != -1 and best_abs < contact_band:
            # Estimate normal via central differences on the signed distance field
            g = ti.Vector.zero(float, 3)
            for axis in ti.static(range(3)):
                e = ti.Vector([0, 0, 0])
                e[axis] = 1

                node_p = best_node + e
                node_m = best_node - e

                for d in ti.static(range(3)):
                    if node_p[d] < 0:
                        node_p[d] = 0
                    if node_p[d] >= C.n_grid:
                        node_p[d] = C.n_grid - 1
                    if node_m[d] < 0:
                        node_m[d] = 0
                    if node_m[d] >= C.n_grid:
                        node_m[d] = C.n_grid - 1

                d_p = S.grid_dist[node_p]
                d_m = S.grid_dist[node_m]
                g[axis] = (d_p - d_m) * (0.5 * C.inv_dx)

            n = ti.Vector([0.0, 1.0, 0.0])
            if g.norm() > 1e-6:
                n = g.normalized()
            else:
                n = ti.Vector([0.0, 1.0, 0.0])

            # Ensure normal points from rigid to particle
            center = S.rb_pos[best_rigid]
            if n.dot(Xp - center) < 0.0:
                n = -n

            S.near_boundary[p] = 1
            S.boundary_rigid_id[p] = best_rigid
            S.boundary_dist[p] = best_abs
            S.boundary_normal[p] = n
            # side: positive signed distance -> outside, negative -> inside
            side = ti.i8(1)
            if best_signed < 0.0:
                side = ti.i8(-1)
            S.boundary_side[p] = side
            S.p_cdf_states[p] = ti.u64(0)  # not used in this scheme
        else:
            S.near_boundary[p] = 0
            S.boundary_rigid_id[p] = -1
            S.boundary_dist[p] = 1e6
            S.boundary_normal[p] = ti.Vector.zero(float, 3)
            S.boundary_side[p] = ti.i8(0)
            S.p_cdf_states[p] = ti.u64(0)


@ti.func
def friction_project(v_rel, n, mu):
    """
    Coulomb friction projection for a relative velocity v_rel along normal n.

    Args:
        v_rel: relative velocity (fluid - rigid).
        n:     contact normal (unit vector).
        mu:    friction coefficient.

    Returns:
        v_rel_after: relative velocity after friction projection.
    """
    vn = v_rel.dot(n)

    # Normal component, clamp to non-penetration: vn_out >= 0
    vn_out = ti.max(vn, 0.0)
    v_n = vn_out * n

    # Tangential component
    v_t = v_rel - vn * n
    vt_norm = v_t.norm() + 1e-8
    max_vt = mu * ti.abs(vn)

    v_t_new = ti.Vector.zero(float, v_rel.n)
    if vt_norm <= max_vt:
        # Sticking: no tangential motion
        v_t_new = ti.Vector.zero(float, v_rel.n)
    else:
        # Sliding: tangential component is limited by Coulomb friction
        v_t_new = v_t * (max_vt / vt_norm)

    return v_n + v_t_new


@ti.kernel
def substep_mpm(gravity: float, contact_band: float):
    """
    One MPM substep:
      1. Clear grid
      2. P2G: pure MPM splat (no rigid coupling here)
      3. Grid operations:
         - normalize
         - gravity
         - outer box boundary
         - rigid-fluid contact projection on grid
      4. G2P: pure MPM gather (uses grid_v after contact projection)
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
        # deprecated
        S.grid_pressure[I] = 0.0
        S.grid_pressure_w[I] = 0.0

    # 2. P2G (standard MPM, no rigid logic)
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

            # Per-material physical parameters
            mat_id = S.materials[p]
            rho0 = 0.0
            E = 0.0
            nu = 0.0
            kind = 0
            for mid in ti.static(range(n_materials)):
                if mat_id == mid:
                    rho0 = rho0_list[mid]
                    E = E_list[mid]
                    nu = nu_list[mid]
                    kind = kind_list[mid]

            mu = E / (2.0 * (1.0 + nu))
            la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

            # Particle mass
            p_mass = rho0 * C.p_vol

            # Hardening factor
            h = ti.exp(10.0 * (1.0 - S.Jp[p]))
            if kind == C.JELLY:
                h = 0.3
            mu *= h
            la *= h
            if kind == C.WATER:
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

            if kind == C.WATER:
                S.F[p] = ti.Matrix.identity(float, 3) * (J ** (1.0 / 3.0))
            elif kind == C.SNOW:
                S.F[p] = U @ sig @ V_svd.transpose()

            # First Piola-Kirchhoff stress
            P = 2 * mu * (S.F[p] - U @ V_svd.transpose()) @ S.F[
                p
            ].transpose() + ti.Matrix.identity(float, 3) * la * J * (J - 1.0)

            # (deprecated) particle pressure
            trace_P = P[0, 0] + P[1, 1] + P[2, 2]
            p_particle = -trace_P / dim

            stress = (-C.dt * C.p_vol * 4.0 * C.inv_dx * C.inv_dx) * P
            affine = stress + p_mass * Cp

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                node = base + offset
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
                    ti.atomic_add(S.grid_pressure[node], weight * stress @ dpos)
                    ti.atomic_add(S.grid_pressure_w[node], weight)

    # 3. Grid operations: normalize, gravity, outer box, THEN rigid-fluid contact on grid
    for I in ti.grouped(S.grid_m):
        m = S.grid_m[I]
        if m > 0.0:
            # Normalize
            v = S.grid_v[I] / m

            # gravity on y
            v[1] += C.dt * gravity

            # Simple outer box boundary
            for d in ti.static(range(dim)):
                if I[d] < 3 and v[d] < 0:
                    v[d] = 0
                if I[d] > C.n_grid - 3 and v[d] > 0:
                    v[d] = 0

            # --- Rigid-fluid contact on grid (conservative) ---
            r_id = S.grid_rigid[I]
            dist = S.grid_dist[I]
            if r_id >= 0 and S.rb_active[r_id] == 1 and ti.abs(dist) < contact_band:
                # Approximate contact normal from SDF
                n_b = S.grid_normal[I]

                # Rigid-body velocity at this node
                x_world = (I.cast(float) + 0.5) * C.dx
                rel = x_world - S.rb_pos[r_id]
                v_rb = S.rb_lin_vel[r_id] + S.rb_ang_vel[r_id].cross(rel)

                # Relative velocity fluid - rigid
                v_rel = v - v_rb
                mu_c = S.rb_friction[r_id]

                # Project relative velocity with Coulomb friction
                v_rel_after = friction_project(v_rel, n_b, mu_c)
                v_proj = v_rb + v_rel_after

                dv = v - v_proj  # change of fluid velocity
                impulse = m * dv + S.grid_pressure[I]  # node impulse

                # Apply equal & opposite impulse to rigid
                ti.atomic_add(S.rb_force[r_id], impulse / C.dt)
                ti.atomic_add(S.rb_torque[r_id], rel.cross(impulse) / C.dt)

                # Update grid velocity (remove impulse from fluid)
                v = v_proj

            S.grid_v[I] = v * m

    # 4. G2P (standard MPM, uses grid_v after contact)
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
                    g_v = S.grid_v[node] / (S.grid_m[node] + 1e-10)
                    weight = w[i][0] * w[j][1] * w[k][2]

                    new_v += weight * g_v
                    new_C += 4.0 * C.inv_dx * weight * g_v.outer_product(dpos)

            S.v[p] = new_v
            S.C_apic[p] = new_C
            S.x[p] += C.dt * new_v


def substep(gravity: float):
    """
    Wrapper for one full substep including:
      - building CDF grid for rigid boundaries
      - MPM substep with rigid-fluid coupling
      - rigid-body integration and collision handling
    """
    contact_band = 0.75 * C.dx

    if S.N_RIGID > 0:
        clear_cdf_grid()
        rasterize_rigid_boundary_cdf(contact_band)
        gather_boundary_from_grid(contact_band)
    else:
        # No rigid bodies: clear particle boundary info
        for p in range(C.n_particles):
            S.near_boundary[p] = 0
            S.boundary_rigid_id[p] = -1
            S.boundary_dist[p] = 1e6
            S.boundary_normal[p] = ti.Vector([0.0, 0.0, 0.0])
            S.boundary_side[p] = 0

    # MPM step with band-aware rigid-fluid coupling
    substep_mpm(gravity, contact_band)

    if S.N_RIGID > 0:
        # Rigid-body integration and collisions
        rigid.integrate_rigid_bodies(gravity)
        rigid.update_all_mesh_vertices()
        rigid.handle_rigid_domain_walls()
        rigid.handle_rigid_collisions()
        rigid.update_all_mesh_vertices()
