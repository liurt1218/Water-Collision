# rigid.py
import taichi as ti
import config as C
import state as S
import materials as M


GRID_RES = 64
MAX_CELL_VERTS = 32

grid_count = ti.field(dtype=ti.i32, shape=(GRID_RES, GRID_RES, GRID_RES))
grid_indices = ti.field(
    dtype=ti.i32,
    shape=(GRID_RES, GRID_RES, GRID_RES, MAX_CELL_VERTS),
)


# Rigid-body collision helpers (deprecated)
@ti.func
def closest_point_on_triangle(p, a, b, c):
    # Compute the closest point on triangle (a,b,c) to point p.
    ab = b - a
    ac = c - a
    ap = p - a

    # Default: assume inside face
    best = a  # will be overwritten

    d1 = ab.dot(ap)
    d2 = ac.dot(ap)

    # Region A
    if d1 <= 0.0 and d2 <= 0.0:
        best = a
    else:
        bp = p - b
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)

        # Region B
        if d3 >= 0.0 and d4 <= d3:
            best = b
        else:
            vc = d1 * d4 - d3 * d2

            # Region AB
            if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                v = d1 / (d1 - d3)
                best = a + v * ab
            else:
                cp = p - c
                d5 = ab.dot(cp)
                d6 = ac.dot(cp)

                # Region C
                if d6 >= 0.0 and d5 <= d6:
                    best = c
                else:
                    vb = d5 * d2 - d1 * d6

                    # Region AC
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        w = d2 / (d2 - d6)
                        best = a + w * ac
                    else:
                        va = d3 * d6 - d5 * d4

                        # Region BC
                        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                            best = b + w * (c - b)
                        else:
                            # Region inside face
                            denom = 1.0 / (va + vb + vc)
                            v = vb * denom
                            w = vc * denom
                            best = a + ab * v + ac * w

    return best


# deprecated
@ti.func
def query_mesh_contact(x_world, r: int, max_dist: float):
    # Exact point-to-triangle distance (unsigned) to mesh of rigid r.
    hit = False
    best_n_world = ti.Vector([0.0, 0.0, 0.0])
    best_d = max_dist

    # Rigid transform
    R = S.rb_rot[r]
    p_r = S.rb_pos[r]

    # world -> local
    rel = x_world - p_r
    x_local = R.transpose() @ rel

    # Coarse AABB culling in local space
    hx = S.rb_half_extents[r] + ti.Vector([max_dist, max_dist, max_dist])
    inside_aabb = (
        ti.abs(x_local[0]) <= hx[0]
        and ti.abs(x_local[1]) <= hx[1]
        and ti.abs(x_local[2]) <= hx[2]
    )
    if inside_aabb:
        start = S.rb_mesh_vert_offset[r]
        count = S.rb_mesh_vert_count[r]
        n_tris = count // 3

        for t in range(n_tris):
            base = start + t * 3
            n_face_local = S.mesh_local_normals[base]

            a = S.mesh_local_vertices[base + 0]
            b = S.mesh_local_vertices[base + 1]
            c = S.mesh_local_vertices[base + 2]

            q = closest_point_on_triangle(x_local, a, b, c)
            diff = x_local - q
            d = diff.norm()

            if d < best_d and diff.dot(n_face_local) >= 0.0:
                best_n_world = R @ n_face_local
                best_d = d
                hit = True

    return hit, best_n_world, best_d


@ti.func
def point_in_triangle(p, a, b, c):
    """
    Check if point p lies inside triangle (a, b, c) using barycentric coordinates.
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)

    denom = d00 * d11 - d01 * d01
    inside = False

    if denom > 1e-12:
        inv_denom = 1.0 / denom
        v = (d11 * d20 - d01 * d21) * inv_denom
        w = (d00 * d21 - d01 * d20) * inv_denom
        u = 1.0 - v - w

        eps = 1e-6
        inside = (u >= -eps) and (v >= -eps) and (w >= -eps)

    return inside


@ti.func
def query_mesh_contact_strict_cdf(
    x_world: ti.types.vector(3, float), rigid_id: int, max_distance: float
):
    """
    Query the signed distance from a point to a rigid mesh surface.

    Args:
        x_world:      query point in world space.
        rigid_id:     id of the rigid body.
        max_distance: maximum distance to consider (search band).

    Returns:
        hit:          True if the closest point is within max_distance.
        best_n:       outward normal at the closest point.
        best_signed:  signed distance ( >0 outside, <0 inside ).
    """
    best_abs = max_distance  # best absolute distance
    best_signed = 0.0  # best signed distance
    best_n = ti.Vector.zero(float, 3)
    hit = False

    center = S.rb_pos[rigid_id]  # rigid body center (used to orient normals)

    # Vertex range for this rigid body
    offset = S.rb_mesh_vert_offset[rigid_id]
    count = S.rb_mesh_vert_count[rigid_id]
    num_tris = count // 3

    for t in range(num_tris):
        i0 = offset + t * 3
        i1 = i0 + 1
        i2 = i0 + 2

        v0 = S.mesh_vertices[i0]
        v1 = S.mesh_vertices[i1]
        v2 = S.mesh_vertices[i2]

        # Triangle normal
        e0 = v1 - v0
        e1 = v2 - v0
        n = e0.cross(e1)
        n_len = n.norm()
        if n_len >= 1e-8:
            n = n / n_len

            # Orient normal to point outward from the rigid center
            # If n points towards the center, flip it.
            if n.dot(v0 - center) < 0.0:
                n = -n

            # Signed distance to the triangle plane (positive = outside)
            signed_d = (x_world - v0).dot(n)

            # Ignore if outside search band
            if ti.abs(signed_d) <= max_distance:
                # Projection of x_world onto the plane
                p = x_world - signed_d * n

                # Strict: projection must lie inside the triangle
                if point_in_triangle(p, v0, v1, v2):
                    abs_d = ti.abs(signed_d)
                    if abs_d < best_abs:
                        best_abs = abs_d
                        best_signed = signed_d
                        best_n = n
                        hit = True

    return hit, best_n, best_signed


# Rigid-body initialization helpers
@ti.kernel
def clear_rigid_bodies():
    # Clear all rigid-body slots (mark them inactive and reset state).
    for r in range(S.N_RIGID):
        S.rb_active[r] = 0

        # Basic properties
        S.rb_mass[r] = 0.0
        S.rb_inv_mass[r] = 0.0
        S.rb_half_extents[r] = ti.Vector.zero(float, 3)

        # Pose
        S.rb_pos[r] = ti.Vector.zero(float, 3)
        S.rb_rot[r] = ti.Matrix.identity(float, 3)

        # Velocities
        S.rb_lin_vel[r] = ti.Vector.zero(float, 3)
        S.rb_ang_vel[r] = ti.Vector.zero(float, 3)

        # Inertia
        S.rb_inertia_body[r] = ti.Matrix.zero(float, 3, 3)
        S.rb_inv_inertia_body[r] = ti.Matrix.zero(float, 3, 3)
        S.rb_inv_inertia_world[r] = ti.Matrix.identity(float, 3)

        # Forces / torques
        S.rb_force[r] = ti.Vector.zero(float, 3)
        S.rb_torque[r] = ti.Vector.zero(float, 3)

        # Material-like parameters
        S.rb_restitution[r] = 0.0
        S.rb_friction[r] = 0.0


@ti.kernel
def init_rigid_from_bbox(
    idx: int,
    cx: float,
    cy: float,
    cz: float,
    hx: float,
    hy: float,
    hz: float,
    mass: float,
    restitution: float,
    friction: float,
):
    # Initialize one rigid body as a box-shaped collider with a given bounding box, mass, and material parameters.
    if 0 <= idx < S.N_RIGID:
        S.rb_active[idx] = 1

        # Pose
        S.rb_pos[idx] = ti.Vector([cx, cy, cz])
        S.rb_rot[idx] = ti.Matrix.identity(float, 3)

        # Velocities
        S.rb_lin_vel[idx] = ti.Vector.zero(float, 3)
        S.rb_ang_vel[idx] = ti.Vector.zero(float, 3)

        # Geometry
        S.rb_half_extents[idx] = ti.Vector([hx, hy, hz])

        # Mass
        S.rb_mass[idx] = mass
        if mass > 0.0:
            S.rb_inv_mass[idx] = 1.0 / mass
        else:
            # Static body: infinite mass -> zero inverse mass
            S.rb_inv_mass[idx] = 0.0

        # Inertia tensor of a box about its center in body space
        I_body = ti.Matrix.zero(float, 3, 3)
        I_body_inv = ti.Matrix.zero(float, 3, 3)

        if mass > 0.0:
            I_xx = (1.0 / 3.0) * mass * (hy * hy + hz * hz)
            I_yy = (1.0 / 3.0) * mass * (hx * hx + hz * hz)
            I_zz = (1.0 / 3.0) * mass * (hx * hx + hy * hy)

            I_body[0, 0] = I_xx
            I_body[1, 1] = I_yy
            I_body[2, 2] = I_zz

            I_body_inv[0, 0] = 1.0 / I_xx
            I_body_inv[1, 1] = 1.0 / I_yy
            I_body_inv[2, 2] = 1.0 / I_zz

        # Store inertia tensors
        S.rb_inertia_body[idx] = I_body
        S.rb_inv_inertia_body[idx] = I_body_inv
        S.rb_inv_inertia_world[idx] = (
            S.rb_rot[idx] @ I_body_inv @ S.rb_rot[idx].transpose()
        )

        # Material parameters
        S.rb_restitution[idx] = restitution
        S.rb_friction[idx] = friction

        # Forces and torques
        S.rb_force[idx] = ti.Vector.zero(float, 3)
        S.rb_torque[idx] = ti.Vector.zero(float, 3)


# Rigid-body integration
@ti.kernel
def integrate_rigid_bodies(gravity: float):
    # Integrate all active rigid bodies for one time step C.dt.
    for r in range(S.N_RIGID):
        if S.rb_active[r] == 1:
            inv_mass = S.rb_inv_mass[r]

            # 1. Linear integration (external forces + gravity)
            acc_lin = ti.Vector.zero(float, 3)
            if inv_mass > 0.0:
                acc_lin = S.rb_force[r] * inv_mass

            # gravity (grid uses v_y += dt * gravity)
            acc_lin[1] += gravity

            if inv_mass > 0.0:
                S.rb_lin_vel[r] += C.dt * acc_lin
                S.rb_pos[r] += C.dt * S.rb_lin_vel[r]

            # 2. Angular integration
            R = S.rb_rot[r]
            I_body_inv = S.rb_inv_inertia_body[r]
            S.rb_inv_inertia_world[r] = R @ I_body_inv @ R.transpose()

            I_inv_world = S.rb_inv_inertia_world[r]

            tau = S.rb_torque[r]
            dw = I_inv_world @ tau
            S.rb_ang_vel[r] += C.dt * dw

            # small-angle rotation update
            w = S.rb_ang_vel[r]
            wx, wy, wz = w[0], w[1], w[2]
            skew = ti.Matrix(
                [
                    [0.0, -wz, wy],
                    [wz, 0.0, -wx],
                    [-wy, wx, 0.0],
                ]
            )
            S.rb_rot[r] = (ti.Matrix.identity(float, 3) + C.dt * skew) @ R

            # 2.1 Re-orthogonalize (Gram-Schmidt)
            R_new = S.rb_rot[r]
            c0 = R_new[:, 0]
            c1 = R_new[:, 1]

            c0 = c0 / (c0.norm() + 1e-8)
            c1 = c1 - c0 * c0.dot(c1)
            c1 = c1 / (c1.norm() + 1e-8)
            c2 = c0.cross(c1)

            S.rb_rot[r] = ti.Matrix.cols([c0, c1, c2])

            # 3. Reset force and torque
            S.rb_force[r] = ti.Vector.zero(float, 3)
            S.rb_torque[r] = ti.Vector.zero(float, 3)


@ti.kernel
def update_all_mesh_vertices():
    # Update world-space vertices and normals for all rigid bodies.
    if S.n_mesh_vertices > 0 and S.n_rigid_bodies > 0:
        for r in range(S.n_rigid_bodies):
            if S.rb_active[r] != 0:
                p = S.rb_pos[r]
                R = S.rb_rot[r]
                start = S.rb_mesh_vert_offset[r]
                count = S.rb_mesh_vert_count[r]

                for i in range(count):
                    idx = start + i
                    x_local = S.mesh_local_vertices[idx]
                    n_local = S.mesh_local_normals[idx]
                    S.mesh_vertices[idx] = p + R @ x_local
                    S.mesh_normals[idx] = R @ n_local


# deprecated
@ti.kernel
def apply_buoyancy_forces():
    g = -C.gravity

    for r in range(S.N_RIGID):
        if S.rb_active[r] == 1:
            R = S.rb_rot[r]
            p_r = S.rb_pos[r]

            start = S.rb_mesh_vert_offset[r]
            count = S.rb_mesh_vert_count[r]
            n_tris = count // 3

            for t in range(n_tris):
                base = start + t * 3

                a_local = S.mesh_local_vertices[base + 0]
                b_local = S.mesh_local_vertices[base + 1]
                c_local = S.mesh_local_vertices[base + 2]

                a = R @ a_local + p_r
                b = R @ b_local + p_r
                c = R @ c_local + p_r

                ab = b - a
                ac = c - a
                n_world = -ab.cross(ac)
                area2 = n_world.norm()

                if area2 > 1e-10:
                    n_world = n_world / area2
                    area = 0.5 * area2

                    center = (a + b + c) / 3.0
                    cy = center[1]

                    p_total = 0.0
                    for mat in ti.static(range(S.N_MATERIALS)):
                        y_surf = S.fluid_surface_y_mat[mat]
                        if cy < y_surf:
                            depth = y_surf - cy
                            rho_k = M.rho0_table[mat]
                            p_total += rho_k * g * depth

                    if p_total > 0.0:
                        dF = 10 * p_total * area * n_world
                        rel = center - p_r
                        ti.atomic_add(S.rb_force[r], dF)
                        ti.atomic_add(S.rb_torque[r], rel.cross(dF))


@ti.kernel
def handle_rigid_collisions():
    thresh = 0.01
    dom_min = ti.Vector([C.domain_min[0], C.domain_min[1], C.domain_min[2]])
    dom_max = ti.Vector([C.domain_max[0], C.domain_max[1], C.domain_max[2]])
    dom_extent = dom_max - dom_min
    for b1 in range(S.n_rigid_bodies):
        for b2 in range(b1 + 1, S.n_rigid_bodies):
            c1 = S.rb_pos[b1]
            c2 = S.rb_pos[b2]
            he1 = S.rb_half_extents[b1]
            he2 = S.rb_half_extents[b2]
            r1b = he1.norm()
            r2b = he2.norm()
            dc = c2 - c1
            if dc.norm() > r1b + r2b + thresh:
                continue

            for gx, gy, gz in ti.ndrange(GRID_RES, GRID_RES, GRID_RES):
                grid_count[gx, gy, gz] = 0

            start2 = S.rb_mesh_vert_offset[b2]
            count2 = S.rb_mesh_vert_count[b2]

            for k2 in range(count2):
                vid2 = start2 + k2
                x2 = S.mesh_vertices[vid2]
                rel2 = (x2 - dom_min) / dom_extent
                gx = int(rel2[0] * GRID_RES)
                gy = int(rel2[1] * GRID_RES)
                gz = int(rel2[2] * GRID_RES)
                if gx < 0:
                    gx = 0
                if gx >= GRID_RES:
                    gx = GRID_RES - 1
                if gy < 0:
                    gy = 0
                if gy >= GRID_RES:
                    gy = GRID_RES - 1
                if gz < 0:
                    gz = 0
                if gz >= GRID_RES:
                    gz = GRID_RES - 1
                idx = grid_count[gx, gy, gz]
                if idx < MAX_CELL_VERTS:
                    grid_indices[gx, gy, gz, idx] = vid2
                    grid_count[gx, gy, gz] = idx + 1

            start1 = S.rb_mesh_vert_offset[b1]
            count1 = S.rb_mesh_vert_count[b1]

            min_dist = 1e8
            p1 = ti.Vector.zero(float, 3)
            p2 = ti.Vector.zero(float, 3)

            for k1 in range(count1):
                vid1 = start1 + k1
                x1 = S.mesh_vertices[vid1]
                rel1 = (x1 - dom_min) / dom_extent
                gx0 = int(rel1[0] * GRID_RES)
                gy0 = int(rel1[1] * GRID_RES)
                gz0 = int(rel1[2] * GRID_RES)
                if gx0 < 0:
                    gx0 = 0
                if gx0 >= GRID_RES:
                    gx0 = GRID_RES - 1
                if gy0 < 0:
                    gy0 = 0
                if gy0 >= GRID_RES:
                    gy0 = GRID_RES - 1
                if gz0 < 0:
                    gz0 = 0
                if gz0 >= GRID_RES:
                    gz0 = GRID_RES - 1

                for dx in ti.static(range(-1, 2)):
                    gx = gx0 + dx
                    if 0 <= gx < GRID_RES:
                        for dy in ti.static(range(-1, 2)):
                            gy = gy0 + dy
                            if 0 <= gy < GRID_RES:
                                for dz in ti.static(range(-1, 2)):
                                    gz = gz0 + dz
                                    if 0 <= gz < GRID_RES:
                                        cnt = grid_count[gx, gy, gz]
                                        for ci in range(cnt):
                                            vid2 = grid_indices[gx, gy, gz, ci]
                                            x2 = S.mesh_vertices[vid2]
                                            d = (x2 - x1).norm()
                                            if d < min_dist:
                                                min_dist = d
                                                p1 = x1
                                                p2 = x2

            if min_dist < thresh and min_dist > 1e-6:
                n = (p2 - p1) / min_dist

                c1 = S.rb_pos[b1]
                c2 = S.rb_pos[b2]

                penetration = thresh - min_dist
                corr = 0.5 * penetration * n
                c1 -= corr
                c2 += corr
                S.rb_pos[b1] = c1
                S.rb_pos[b2] = c2

                r1 = p1 - c1
                r2 = p2 - c2

                v1 = S.rb_lin_vel[b1]
                v2 = S.rb_lin_vel[b2]
                w1 = S.rb_ang_vel[b1]
                w2 = S.rb_ang_vel[b2]

                v1c = v1 + w1.cross(r1)
                v2c = v2 + w2.cross(r2)

                v_rel = v2c - v1c
                v_rel_n = v_rel.dot(n)

                if v_rel_n < 0.0:
                    I1_inv = S.rb_inv_inertia_world[b1]
                    I2_inv = S.rb_inv_inertia_world[b2]

                    inv_m1 = S.rb_inv_mass[b1]
                    inv_m2 = S.rb_inv_mass[b2]

                    e = 0.5 * (S.rb_restitution[b1] + S.rb_restitution[b2])
                    mu = 0.5 * (S.rb_friction[b1] + S.rb_friction[b2])

                    rn1 = r1.cross(n)
                    rn2 = r2.cross(n)

                    ang_term1 = (I1_inv @ rn1).cross(r1).dot(n)
                    ang_term2 = (I2_inv @ rn2).cross(r2).dot(n)
                    denom_n = inv_m1 + inv_m2 + ang_term1 + ang_term2 + 1e-6

                    j_n = -(1.0 + e) * v_rel_n / denom_n
                    J_n = j_n * n

                    v1 -= J_n * inv_m1
                    v2 += J_n * inv_m2
                    w1 += I1_inv @ r1.cross(-J_n)
                    w2 += I2_inv @ r2.cross(+J_n)

                    S.rb_lin_vel[b1] = v1
                    S.rb_lin_vel[b2] = v2
                    S.rb_ang_vel[b1] = w1
                    S.rb_ang_vel[b2] = w2

                    v1c = v1 + w1.cross(r1)
                    v2c = v2 + w2.cross(r2)

                    v_rel = v2c - v1c
                    ncomp = v_rel.dot(n)
                    v_tan = v_rel - ncomp * n
                    vt_len = v_tan.norm()

                    if vt_len > 1e-6:
                        t = v_tan / vt_len

                        rt1 = r1.cross(t)
                        rt2 = r2.cross(t)

                        ang_term1_t = (I1_inv @ rt1).cross(r1).dot(t)
                        ang_term2_t = (I2_inv @ rt2).cross(r2).dot(t)
                        denom_t = inv_m1 + inv_m2 + ang_term1_t + ang_term2_t + 1e-6

                        j_t_raw = -vt_len / denom_t
                        max_j_t = mu * ti.abs(j_n)

                        j_t = j_t_raw
                        if j_t > max_j_t:
                            j_t = max_j_t
                        if j_t < -max_j_t:
                            j_t = -max_j_t

                        J_t = j_t * t

                        v1 -= J_t * inv_m1
                        v2 += J_t * inv_m2
                        w1 += I1_inv @ r1.cross(-J_t)
                        w2 += I2_inv @ r2.cross(+J_t)

                        S.rb_lin_vel[b1] = v1
                        S.rb_lin_vel[b2] = v2
                        S.rb_ang_vel[b1] = w1
                        S.rb_ang_vel[b2] = w2


@ti.kernel
def handle_rigid_domain_walls():
    min_x = C.domain_min[0]
    min_y = C.domain_min[1]
    min_z = C.domain_min[2]
    max_x = C.domain_max[0]
    max_y = C.domain_max[1]
    max_z = C.domain_max[2]

    max_pen = 0.01

    for b in range(S.n_rigid_bodies):
        inv_m = S.rb_inv_mass[b]
        restitution = S.rb_restitution[b]
        mu = S.rb_friction[b]

        c = S.rb_pos[b]
        v_b = S.rb_lin_vel[b]
        w = S.rb_ang_vel[b]
        I_inv = S.rb_inv_inertia_world[b]

        start = S.rb_mesh_vert_offset[b]
        count = S.rb_mesh_vert_count[b]

        # y- wall
        n = ti.Vector([0.0, 1.0, 0.0])
        min_coord = 1e8
        cp = ti.Vector.zero(ti.f32, C.dim)
        found = False
        for k in range(count):
            vid = start + k
            p = S.mesh_vertices[vid]
            if p.y < min_coord:
                min_coord = p.y
                cp = p
                found = True
        if found and (min_coord < min_y):
            penetration = min_y - min_coord
            if penetration > max_pen:
                penetration = max_pen
            c.y += penetration
            cp.y += penetration
            S.rb_pos[b] = c
            r = cp - c
            v_c = v_b + w.cross(r)
            v_rel = v_c
            v_rel_n = v_rel.dot(n)
            j_n = 0.0
            if v_rel_n < 0.0:
                rn = r.cross(n)
                ang_term = (I_inv @ rn).cross(r).dot(n)
                denom_n = inv_m + ang_term + 1e-6
                j_n = -(1.0 + restitution) * v_rel_n / denom_n
                J_n = j_n * n
                v_b += J_n * inv_m
                w += I_inv @ r.cross(J_n)
            v_c = v_b + w.cross(r)
            v_rel = v_c
            ncomp = v_rel.dot(n)
            v_tan = v_rel - ncomp * n
            vt_len = v_tan.norm()
            if vt_len > 1e-6:
                t = v_tan / vt_len
                rt = r.cross(t)
                ang_term_t = (I_inv @ rt).cross(r).dot(t)
                denom_t = inv_m + ang_term_t + 1e-6
                j_t = -vt_len / denom_t
                max_j_t = mu * ti.abs(j_n)
                if j_t > max_j_t:
                    j_t = max_j_t
                if j_t < -max_j_t:
                    j_t = -max_j_t
                J_t = j_t * t
                v_b += J_t * inv_m
                w += I_inv @ r.cross(J_t)

        # y+ wall
        n = ti.Vector([0.0, -1.0, 0.0])
        max_coord = -1e8
        cp = ti.Vector.zero(ti.f32, C.dim)
        found = False
        for k in range(count):
            vid = start + k
            p = S.mesh_vertices[vid]
            if p.y > max_coord:
                max_coord = p.y
                cp = p
                found = True
        if found and (max_coord > max_y):
            penetration = max_coord - max_y
            if penetration > max_pen:
                penetration = max_pen
            c.y -= penetration
            cp.y -= penetration
            S.rb_pos[b] = c
            r = cp - c
            v_c = v_b + w.cross(r)
            v_rel = v_c
            v_rel_n = v_rel.dot(n)
            j_n = 0.0
            if v_rel_n < 0.0:
                rn = r.cross(n)
                ang_term = (I_inv @ rn).cross(r).dot(n)
                denom_n = inv_m + ang_term + 1e-6
                j_n = -(1.0 + restitution) * v_rel_n / denom_n
                J_n = j_n * n
                v_b += J_n * inv_m
                w += I_inv @ r.cross(J_n)
            v_c = v_b + w.cross(r)
            v_rel = v_c
            ncomp = v_rel.dot(n)
            v_tan = v_rel - ncomp * n
            vt_len = v_tan.norm()
            if vt_len > 1e-6:
                t = v_tan / vt_len
                rt = r.cross(t)
                ang_term_t = (I_inv @ rt).cross(r).dot(t)
                denom_t = inv_m + ang_term_t + 1e-6
                j_t = -vt_len / denom_t
                max_j_t = mu * ti.abs(j_n)
                if j_t > max_j_t:
                    j_t = max_j_t
                if j_t < -max_j_t:
                    j_t = -max_j_t
                J_t = j_t * t
                v_b += J_t * inv_m
                w += I_inv @ r.cross(J_t)

        # x- wall
        n = ti.Vector([1.0, 0.0, 0.0])
        min_coord = 1e8
        cp = ti.Vector.zero(ti.f32, C.dim)
        found = False
        for k in range(count):
            vid = start + k
            p = S.mesh_vertices[vid]
            if p.x < min_coord:
                min_coord = p.x
                cp = p
                found = True
        if found and (min_coord < min_x):
            penetration = min_x - min_coord
            if penetration > max_pen:
                penetration = max_pen
            c.x += penetration
            cp.x += penetration
            S.rb_pos[b] = c
            r = cp - c
            v_c = v_b + w.cross(r)
            v_rel = v_c
            v_rel_n = v_rel.dot(n)
            j_n = 0.0
            if v_rel_n < 0.0:
                rn = r.cross(n)
                ang_term = (I_inv @ rn).cross(r).dot(n)
                denom_n = inv_m + ang_term + 1e-6
                j_n = -(1.0 + restitution) * v_rel_n / denom_n
                J_n = j_n * n
                v_b += J_n * inv_m
                w += I_inv @ r.cross(J_n)
            v_c = v_b + w.cross(r)
            v_rel = v_c
            ncomp = v_rel.dot(n)
            v_tan = v_rel - ncomp * n
            vt_len = v_tan.norm()
            if vt_len > 1e-6:
                t = v_tan / vt_len
                rt = r.cross(t)
                ang_term_t = (I_inv @ rt).cross(r).dot(t)
                denom_t = inv_m + ang_term_t + 1e-6
                j_t = -vt_len / denom_t
                max_j_t = mu * ti.abs(j_n)
                if j_t > max_j_t:
                    j_t = max_j_t
                if j_t < -max_j_t:
                    j_t = -max_j_t
                J_t = j_t * t
                v_b += J_t * inv_m
                w += I_inv @ r.cross(J_t)

        # x+ wall
        n = ti.Vector([-1.0, 0.0, 0.0])
        max_coord = -1e8
        cp = ti.Vector.zero(ti.f32, C.dim)
        found = False
        for k in range(count):
            vid = start + k
            p = S.mesh_vertices[vid]
            if p.x > max_coord:
                max_coord = p.x
                cp = p
                found = True
        if found and (max_coord > max_x):
            penetration = max_coord - max_x
            if penetration > max_pen:
                penetration = max_pen
            c.x -= penetration
            cp.x -= penetration
            S.rb_pos[b] = c
            r = cp - c
            v_c = v_b + w.cross(r)
            v_rel = v_c
            v_rel_n = v_rel.dot(n)
            j_n = 0.0
            if v_rel_n < 0.0:
                rn = r.cross(n)
                ang_term = (I_inv @ rn).cross(r).dot(n)
                denom_n = inv_m + ang_term + 1e-6
                j_n = -(1.0 + restitution) * v_rel_n / denom_n
                J_n = j_n * n
                v_b += J_n * inv_m
                w += I_inv @ r.cross(J_n)
            v_c = v_b + w.cross(r)
            v_rel = v_c
            ncomp = v_rel.dot(n)
            v_tan = v_rel - ncomp * n
            vt_len = v_tan.norm()
            if vt_len > 1e-6:
                t = v_tan / vt_len
                rt = r.cross(t)
                ang_term_t = (I_inv @ rt).cross(r).dot(t)
                denom_t = inv_m + ang_term_t + 1e-6
                j_t = -vt_len / denom_t
                max_j_t = mu * ti.abs(j_n)
                if j_t > max_j_t:
                    j_t = max_j_t
                if j_t < -max_j_t:
                    j_t = -max_j_t
                J_t = j_t * t
                v_b += J_t * inv_m
                w += I_inv @ r.cross(J_t)

        # z- wall
        n = ti.Vector([0.0, 0.0, 1.0])
        min_coord = 1e8
        cp = ti.Vector.zero(ti.f32, C.dim)
        found = False
        for k in range(count):
            vid = start + k
            p = S.mesh_vertices[vid]
            if p.z < min_coord:
                min_coord = p.z
                cp = p
                found = True
        if found and (min_coord < min_z):
            penetration = min_z - min_coord
            if penetration > max_pen:
                penetration = max_pen
            c.z += penetration
            cp.z += penetration
            S.rb_pos[b] = c
            r = cp - c
            v_c = v_b + w.cross(r)
            v_rel = v_c
            v_rel_n = v_rel.dot(n)
            j_n = 0.0
            if v_rel_n < 0.0:
                rn = r.cross(n)
                ang_term = (I_inv @ rn).cross(r).dot(n)
                denom_n = inv_m + ang_term + 1e-6
                j_n = -(1.0 + restitution) * v_rel_n / denom_n
                J_n = j_n * n
                v_b += J_n * inv_m
                w += I_inv @ r.cross(J_n)
            v_c = v_b + w.cross(r)
            v_rel = v_c
            ncomp = v_rel.dot(n)
            v_tan = v_rel - ncomp * n
            vt_len = v_tan.norm()
            if vt_len > 1e-6:
                t = v_tan / vt_len
                rt = r.cross(t)
                ang_term_t = (I_inv @ rt).cross(r).dot(t)
                denom_t = inv_m + ang_term_t + 1e-6
                j_t = -vt_len / denom_t
                max_j_t = mu * ti.abs(j_n)
                if j_t > max_j_t:
                    j_t = max_j_t
                if j_t < -max_j_t:
                    j_t = -max_j_t
                J_t = j_t * t
                v_b += J_t * inv_m
                w += I_inv @ r.cross(J_t)

        # z+ wall
        n = ti.Vector([0.0, 0.0, -1.0])
        max_coord = -1e8
        cp = ti.Vector.zero(ti.f32, C.dim)
        found = False
        for k in range(count):
            vid = start + k
            p = S.mesh_vertices[vid]
            if p.z > max_coord:
                max_coord = p.z
                cp = p
                found = True
        if found and (max_coord > max_z):
            penetration = max_coord - max_z
            if penetration > max_pen:
                penetration = max_pen
            c.z -= penetration
            cp.z -= penetration
            S.rb_pos[b] = c
            r = cp - c
            v_c = v_b + w.cross(r)
            v_rel = v_c
            v_rel_n = v_rel.dot(n)
            j_n = 0.0
            if v_rel_n < 0.0:
                rn = r.cross(n)
                ang_term = (I_inv @ rn).cross(r).dot(n)
                denom_n = inv_m + ang_term + 1e-6
                j_n = -(1.0 + restitution) * v_rel_n / denom_n
                J_n = j_n * n
                v_b += J_n * inv_m
                w += I_inv @ r.cross(J_n)
            v_c = v_b + w.cross(r)
            v_rel = v_c
            ncomp = v_rel.dot(n)
            v_tan = v_rel - ncomp * n
            vt_len = v_tan.norm()
            if vt_len > 1e-6:
                t = v_tan / vt_len
                rt = r.cross(t)
                ang_term_t = (I_inv @ rt).cross(r).dot(t)
                denom_t = inv_m + ang_term_t + 1e-6
                j_t = -vt_len / denom_t
                max_j_t = mu * ti.abs(j_n)
                if j_t > max_j_t:
                    j_t = max_j_t
                if j_t < -max_j_t:
                    j_t = -max_j_t
                J_t = j_t * t
                v_b += J_t * inv_m
                w += I_inv @ r.cross(J_t)

        S.rb_lin_vel[b] = v_b
        S.rb_ang_vel[b] = w
