# sim/rigid.py
import taichi as ti
import numpy as np

from .config import RigidSceneConfig
from .io_utils import load_obj
from . import state as S
from . import config as C


# ================================================================
# Utilities
# ================================================================
@ti.func
def skew(w):
    """Skew-symmetric matrix for cross product."""
    return ti.Matrix(
        [
            [0.0, -w.z, w.y],
            [w.z, 0.0, -w.x],
            [-w.y, w.x, 0.0],
        ]
    )


@ti.func
def orthonormalize(R):
    """Re-orthonormalize a 3x3 rotation matrix."""
    c0 = R[:, 0]
    c1 = R[:, 1]
    c2 = R[:, 2]

    c0 = c0.normalized()
    c1 = (c1 - c0.dot(c1) * c0).normalized()
    c2 = c0.cross(c1)

    return ti.Matrix.cols([c0, c1, c2])


@ti.func
def rot_from_euler(rx, ry, rz):
    """Rotation matrix from Euler angles."""
    cx, sx = ti.cos(rx), ti.sin(rx)
    cy, sy = ti.cos(ry), ti.sin(ry)
    cz, sz = ti.cos(rz), ti.sin(rz)

    Rx = ti.Matrix([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = ti.Matrix([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = ti.Matrix([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


# ================================================================
# Initialization
# ================================================================
# sim/rigid.py
import numpy as np
import taichi as ti

from .config import RigidSceneConfig
from .io_utils import load_obj
from . import state as S
from . import config as C
import tqdm


def init_rigid_from_scene(rigid_cfg: RigidSceneConfig):
    """
    Python-side initialization of rigid bodies:

    - Load each body's mesh from OBJ.
    - Center and scale vertices to match the requested half extents.
    - Upload local vertices to rb_local and mesh_local.
    - Upload triangle indices to mesh_indices.
    - Upload per-body physical parameters into Taichi fields.
    """
    assert len(rigid_cfg.bodies) == S.n_rigid_bodies

    vert_offset = 0
    index_offset = 0
    rb_local_temp = None
    mesh_local_temp = None
    mesh_indices_temp = None

    for b, body in enumerate(rigid_cfg.bodies):
        verts_np, faces_np = load_obj(body.mesh_path)
        verts_np = verts_np.astype(np.float32)
        faces_np = faces_np.astype(np.int32)

        bbox_min = verts_np.min(axis=0)
        bbox_max = verts_np.max(axis=0)
        center = (bbox_min + bbox_max) * 0.5
        extent = bbox_max - bbox_min
        max_extent = float(extent.max() + 1e-8)

        hx, hy, hz = body.half_extents
        target = max(hx, hy, hz) * 2.0
        scale = target / max_extent

        verts_local = (verts_np - center) * scale  # (Ni, 3)

        vcnt = verts_local.shape[0]
        icnt = faces_np.size  # number of indices = 3 * num_triangles

        # Sanity check with allocated counts
        assert vcnt == S.mesh_vert_count[b]
        assert icnt == S.mesh_index_count[b]

        # Upload local vertices for physics (rb_local) and rendering (mesh_local)
        if rb_local_temp is None:
            rb_local_temp = verts_local
        else:
            rb_local_temp = np.concatenate([rb_local_temp, verts_local], axis=0)
        if mesh_local_temp is None:
            mesh_local_temp = verts_local
        else:
            mesh_local_temp = np.concatenate([mesh_local_temp, verts_local], axis=0)

        # Upload indices; faces_np is (T, 3) of int32
        flat_indices = faces_np.reshape(-1)
        if mesh_indices_temp is None:
            mesh_indices_temp = flat_indices
        else:
            mesh_indices_temp = np.concatenate([mesh_indices_temp, flat_indices], axis=0)

        # Per-body parameters
        S.rb_pos[b] = ti.Vector(body.center)
        S.rb_half[b] = ti.Vector([hx, hy, hz])

        volume = 8.0 * hx * hy * hz
        S.rb_mass[b] = body.density * volume
        S.rb_restitution[b] = body.restitution
        S.rb_friction[b] = body.friction
        S.rb_contact_threshold_scale[b] = body.contact_threshold_scale
        S.rb_ground_penetration_clamp[b] = body.ground_penetration_clamp
        S.rb_angular_damping[b] = body.angular_damping_ground

        vert_offset += vcnt
        index_offset += icnt

    S.rb_local.from_numpy(rb_local_temp)
    S.mesh_local.from_numpy(mesh_local_temp)
    S.mesh_indices.from_numpy(mesh_indices_temp)

    # Mark fluid particles as not belonging to any rigid body
    for i in tqdm.tqdm(range(S.n_fluid), desc="indexing fluid particles"):
        S.rigid_id[i] = -1


@ti.kernel
def _init_rigid_kernel():
    """Initialize inertia tensors, rigid states, and rigid particles."""
    # Inertia and initial state per body
    for body in range(S.n_rigid_bodies):
        half = S.rb_half[body]
        mass = S.rb_mass[body]

        a = 2.0 * half[0]
        b = 2.0 * half[1]
        c = 2.0 * half[2]

        Ixx = (1.0 / 12.0) * mass * (b * b + c * c)
        Iyy = (1.0 / 12.0) * mass * (a * a + c * c)
        Izz = (1.0 / 12.0) * mass * (a * a + b * b)

        S.I_body[body] = ti.Matrix([[Ixx, 0.0, 0.0], [0.0, Iyy, 0.0], [0.0, 0.0, Izz]])
        S.I_body_inv[body] = S.I_body[body].inverse()

        S.rb_vel[body] = ti.Vector.zero(ti.f32, C.dim)
        S.rb_force[body] = ti.Vector.zero(ti.f32, C.dim)
        S.rb_omega[body] = ti.Vector.zero(ti.f32, C.dim)
        S.rb_torque[body] = ti.Vector.zero(ti.f32, C.dim)
        S.rb_rot[body] = rot_from_euler(ti.random(), ti.random(), ti.random())

    # Rigid particles (ragged layout)
    for b in range(S.n_rigid_bodies):
        c = S.rb_pos[b]
        half = S.rb_half[b]
        start = S.rb_offset[b]
        count = S.rb_count[b]

        volume = 8.0 * half[0] * half[1] * half[2]
        rest_v = volume / ti.max(1, count)

        for k in range(count):
            local_idx = start + k
            pid = S.n_fluid + local_idx

            local = S.rb_local[local_idx]
            S.x[pid] = c + local
            S.v[pid] = ti.Vector.zero(ti.f32, C.dim)
            S.a[pid] = ti.Vector.zero(ti.f32, C.dim)
            S.is_fluid[pid] = 0
            S.is_dynamic[pid] = 1
            S.rest_volume[pid] = rest_v
            S.rigid_id[pid] = b

    # Fluid particles are already set to rigid_id = -1 in Python


def init_rigid(rigid_scene_cfg: RigidSceneConfig):
    """High-level rigid initialization entry point."""
    init_rigid_from_scene(rigid_scene_cfg)
    _init_rigid_kernel()


# ================================================================
# Rigid-body integration
# ================================================================
@ti.kernel
def rigid_step():
    """Integrate rigid-body linear and angular motion."""
    for b in range(S.n_rigid_bodies):
        vel = S.rb_vel[b]
        pos = S.rb_pos[b]
        force = S.rb_force[b]

        omega = S.rb_omega[b]
        torque = S.rb_torque[b]
        R = S.rb_rot[b]

        I_b = S.I_body[b]
        I_b_inv = S.I_body_inv[b]

        mass = S.rb_mass[b]
        half = S.rb_half[b]
        restitution = S.rb_restitution[b]

        # Linear motion
        vel += C.dt * (force / mass + C.g)
        pos += C.dt * vel

        # Angular motion
        I_w = R @ I_b @ R.transpose()
        I_w_inv = R @ I_b_inv @ R.transpose()
        ang_acc = I_w_inv @ (torque - omega.cross(I_w @ omega))
        omega += C.dt * ang_acc

        R += C.dt * skew(omega) @ R
        R = orthonormalize(R)

        # Boundary collisions for rigid body center
        for k in ti.static(range(C.dim)):
            if k != 1:
                if pos[k] < C.domain_min[k] + half[k]:
                    pos[k] = C.domain_min[k] + half[k]
                    vel[k] *= -restitution
                if pos[k] > C.domain_max[k] - half[k]:
                    pos[k] = C.domain_max[k] - half[k]
                    vel[k] *= -restitution

        S.rb_vel[b] = vel
        S.rb_pos[b] = pos
        S.rb_omega[b] = omega
        S.rb_rot[b] = R
        S.rb_force[b] = ti.Vector.zero(ti.f32, C.dim)
        S.rb_torque[b] = ti.Vector.zero(ti.f32, C.dim)


@ti.kernel
def renew_rigid_particles():
    """Update rigid particles from current rigid-body states."""
    for b in range(S.n_rigid_bodies):
        c = S.rb_pos[b]
        R = S.rb_rot[b]
        omega = S.rb_omega[b]

        start = S.rb_offset[b]
        count = S.rb_count[b]

        for k in range(count):
            local_idx = start + k
            pid = S.n_fluid + local_idx

            local = S.rb_local[local_idx]
            offset = R @ local
            S.x[pid] = c + offset
            S.v[pid] = S.rb_vel[b] + omega.cross(offset)


@ti.kernel
def update_mesh_vertices(body: ti.i32):
    """Transform mesh vertices of one rigid body to world space."""
    c = S.rb_pos[body]
    R = S.rb_rot[body]

    start = S.mesh_vert_offset[body]
    count = S.mesh_vert_count[body]

    for k in range(count):
        i = start + k
        S.mesh_vertices[i] = c + R @ S.mesh_local[i]


# ================================================================
# Rigid-rigid collision
# ================================================================
@ti.kernel
def handle_rigid_collisions():
    """Rigid-rigid collision with friction using closest particles."""
    for b1 in range(S.n_rigid_bodies):
        for b2 in range(b1 + 1, S.n_rigid_bodies):
            min_dist = 1e8
            p1 = ti.Vector.zero(ti.f32, C.dim)
            p2 = ti.Vector.zero(ti.f32, C.dim)

            # Per-body particle ranges
            start1 = S.rb_offset[b1]
            count1 = S.rb_count[b1]
            start2 = S.rb_offset[b2]
            count2 = S.rb_count[b2]

            # Find closest particle pair between body b1 and b2
            for k1 in range(count1):
                pid1 = S.n_fluid + start1 + k1
                x1 = S.x[pid1]
                for k2 in range(count2):
                    pid2 = S.n_fluid + start2 + k2
                    x2 = S.x[pid2]
                    d = (x2 - x1).norm()
                    if d < min_dist:
                        min_dist = d
                        p1 = x1
                        p2 = x2

            thresh = C.rigid_particle_diameter * 0.8
            if min_dist < thresh and min_dist > 1e-6:
                n = (p2 - p1) / min_dist

                c1 = S.rb_pos[b1]
                c2 = S.rb_pos[b2]

                # Positional correction (split between two bodies)
                penetration = thresh - min_dist
                corr = 0.5 * penetration * n
                c1 -= corr
                c2 += corr
                S.rb_pos[b1] = c1
                S.rb_pos[b2] = c2

                r1 = p1 - c1
                r2 = p2 - c2

                v1 = S.rb_vel[b1]
                v2 = S.rb_vel[b2]
                w1 = S.rb_omega[b1]
                w2 = S.rb_omega[b2]

                v1c = v1 + w1.cross(r1)
                v2c = v2 + w2.cross(r2)

                v_rel = v2c - v1c
                v_rel_n = v_rel.dot(n)

                if v_rel_n < 0.0:
                    I1_inv = S.I_body_inv[b1]
                    I2_inv = S.I_body_inv[b2]

                    m1 = S.rb_mass[b1]
                    m2 = S.rb_mass[b2]
                    inv_m1 = 1.0 / m1
                    inv_m2 = 1.0 / m2

                    # Use average restitution and friction
                    e = 0.5 * (S.rb_restitution[b1] + S.rb_restitution[b2])
                    mu = 0.5 * (S.rb_friction[b1] + S.rb_friction[b2])

                    # Normal impulse
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

                    S.rb_vel[b1] = v1
                    S.rb_vel[b2] = v2
                    S.rb_omega[b1] = w1
                    S.rb_omega[b2] = w2

                    # Recompute relative velocity after normal impulse
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

                        # Clamp friction impulse
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

                        S.rb_vel[b1] = v1
                        S.rb_vel[b2] = v2
                        S.rb_omega[b1] = w1
                        S.rb_omega[b2] = w2


# ================================================================
# Rigid–ground collision
# ================================================================
@ti.kernel
def handle_rigid_ground_collision():
    """Rigid-ground collision with friction and angular damping."""
    ground_y = C.domain_min[1]
    n = ti.Vector([0.0, 1.0, 0.0])

    for b in range(S.n_rigid_bodies):
        # Per-body parameters
        mass = S.rb_mass[b]
        inv_m = 1.0 / mass
        restitution = S.rb_restitution[b]
        mu = S.rb_friction[b]
        ang_damp = S.rb_angular_damping[b]
        max_pen = S.rb_ground_penetration_clamp[b] * C.rigid_particle_diameter

        # Find lowest particle of this body
        min_y = 1e8
        cp = ti.Vector.zero(ti.f32, C.dim)
        found = False

        start = S.rb_offset[b]
        count = S.rb_count[b]
        for k in range(count):
            pid = S.n_fluid + start + k
            p = S.x[pid]
            if p.y < min_y:
                min_y = p.y
                cp = p
                found = True

        if (not found) or (min_y >= ground_y):
            continue

        c = S.rb_pos[b]
        v_b = S.rb_vel[b]
        w = S.rb_omega[b]
        I_inv = S.I_body_inv[b]

        # Positional correction
        penetration = ground_y - min_y
        if penetration > max_pen:
            penetration = max_pen

        c.y += penetration
        cp.y += penetration
        S.rb_pos[b] = c

        r = cp - c

        # Normal impulse
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

        # Tangential (friction) impulse
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

            # Coulomb friction clamp
            max_j_t = mu * ti.abs(j_n)
            if j_t > max_j_t:
                j_t = max_j_t
            if j_t < -max_j_t:
                j_t = -max_j_t

            J_t = j_t * t

            v_b += J_t * inv_m
            w += I_inv @ r.cross(J_t)

        # Angular damping
        w *= 1.0 - ang_damp

        S.rb_vel[b] = v_b
        S.rb_omega[b] = w
