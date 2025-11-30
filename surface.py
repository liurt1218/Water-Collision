# surface.py
import os
import taichi as ti
import numpy as np
import mcubes  # pip install PyMCubes

import config as C
import state as S
import rigid

SURF_N = min(C.n_grid, 64)

N_X = SURF_N
N_Y = SURF_N
N_Z = SURF_N

DX = (C.domain_max[0] - C.domain_min[0]) / N_X  # surface grid spacing

phi = ti.field(dtype=ti.f32, shape=(N_X, N_Y, N_Z))


@ti.func
def mpm_bspline_weight_1d(f: ti.f32, idx: ti.i32) -> ti.f32:
    # Quadratic B-spline weights used in MPM (per dimension).
    w = 0.0
    if idx == 0:
        w = 0.5 * (1.5 - f) * (1.5 - f)
    elif idx == 1:
        w = 0.75 - (f - 1.0) * (f - 1.0)
    else:  # idx == 2
        w = 0.5 * (f - 0.5) * (f - 0.5)
    return w


@ti.kernel
def build_scalar_field_for_material(mat_id: ti.i32):
    # Build a scalar field using an MPM-style quadratic B-spline kernel, but only from particles with a specific material_id.
    # Clear scalar field
    for I in ti.grouped(phi):
        phi[I] = 0.0

    for p in range(C.n_particles):
        if S.is_used[p] == 1 and S.materials[p] == mat_id:
            xp = S.x[p]

            # Map to surface grid coordinates: [0, N_X) etc.
            Xp = (xp - ti.Vector(C.domain_min)) / DX

            # Standard MPM base index + local coordinate in [0, 3)
            base = (Xp - 0.5).cast(int)
            fx = Xp - base.cast(ti.f32)

            for i, j, k in ti.ndrange(3, 3, 3):
                gx = base[0] + i
                gy = base[1] + j
                gz = base[2] + k

                if 0 <= gx < N_X and 0 <= gy < N_Y and 0 <= gz < N_Z:
                    wx = mpm_bspline_weight_1d(fx[0], i)
                    wy = mpm_bspline_weight_1d(fx[1], j)
                    wz = mpm_bspline_weight_1d(fx[2], k)
                    w = wx * wy * wz

                    ti.atomic_add(phi[gx, gy, gz], w)


def reconstruct_fluid_surface_for_material(mat_id: int, iso_ratio: float = -1.0):
    # Reconstruct the surface mesh for a given material_id using marching cubes.
    # 1. Build scalar field for this material
    build_scalar_field_for_material(mat_id)

    # 2. Copy to NumPy
    phi_np = phi.to_numpy()
    phi_max = float(phi_np.max())
    if phi_max <= 0.0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    positive = phi_np[phi_np > 0.0]
    phi_mean = float(positive.mean()) if positive.size > 0 else phi_max

    # 3. Choose an iso level
    if iso_ratio > 0.0:
        iso = iso_ratio * phi_max
    else:
        # Heuristic: somewhere between mean and a fraction of max
        iso_from_max = 0.2 * phi_max
        iso_from_mean = 0.8 * phi_mean
        iso = max(iso_from_max, iso_from_mean)

    phi_shifted = iso - phi_np  # iso-surface at phi_shifted == 0

    # 4. Marching cubes on CPU
    verts, faces = mcubes.marching_cubes(phi_shifted, 0.0)

    # 5. Map index space -> world space
    domain_min = np.array(C.domain_min, dtype=np.float32)

    verts_world = np.empty_like(verts, dtype=np.float32)
    verts_world[:, 0] = domain_min[0] + (verts[:, 0] + 0.5) * DX
    verts_world[:, 1] = domain_min[1] + (verts[:, 1] + 0.5) * DX
    verts_world[:, 2] = domain_min[2] + (verts[:, 2] + 0.5) * DX

    return verts_world.astype(np.float32), faces.astype(np.int32)


def export_fluid_objs_per_material(
    out_dir: str,
    frame: int,
    iso_ratio: float = -1.0,
    fluid_material_ids=None,
):
    # Export fluid surfaces as OBJ for each material_id.
    os.makedirs(out_dir, exist_ok=True)

    # Determine which material IDs to export
    mats_np = S.materials.to_numpy()
    used_np = S.is_used.to_numpy().astype(bool)

    if fluid_material_ids is None:
        # Use all material IDs that appear in used particles
        mats_used = mats_np[used_np]
        if mats_used.size == 0:
            print(f"[fluid_export] frame {frame}: no active particles, skip.")
            return
        unique_mats = np.unique(mats_used)
    else:
        unique_mats = np.array(fluid_material_ids, dtype=np.int32)

    for mat_id in unique_mats:
        verts, faces = reconstruct_fluid_surface_for_material(
            mat_id, iso_ratio=iso_ratio
        )
        if verts.shape[0] == 0 or faces.shape[0] == 0:
            # No visible surface for this material in this frame
            continue

        obj_path = os.path.join(out_dir, f"fluid_{int(mat_id)}_{frame:04d}.obj")
        with open(obj_path, "w") as f:
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for tri in faces:
                f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")

        print(
            f"[fluid_export] material {int(mat_id)}, frame {frame}: "
            f"{verts.shape[0]} verts, {faces.shape[0]} faces -> {obj_path}"
        )


def export_rigid_objs(out_dir: str, frame: int):
    if S.n_rigid_bodies == 0 or S.n_mesh_vertices == 0:
        return

    os.makedirs(out_dir, exist_ok=True)

    # Update world-space vertices for all rigid meshes
    rigid.update_all_mesh_vertices()

    verts_all = S.mesh_vertices.to_numpy()  # [N, 3]
    offsets = S.rb_mesh_vert_offset.to_numpy()  # [n_rigid]
    counts = S.rb_mesh_vert_count.to_numpy()  # [n_rigid]

    for rid in range(S.n_rigid_bodies):
        off = int(offsets[rid])
        cnt = int(counts[rid])
        if cnt <= 0:
            continue

        tri_verts = verts_all[off : off + cnt]  # [cnt, 3]
        n_tris = cnt // 3

        obj_path = os.path.join(out_dir, f"rigid_{rid}_{frame:04d}.obj")
        with open(obj_path, "w") as f:
            for v in tri_verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for t in range(n_tris):
                i0 = 3 * t + 1
                i1 = 3 * t + 2
                i2 = 3 * t + 3
                f.write(f"f {i0} {i1} {i2}\n")

        print(
            f"[rigid_export] rigid {rid}, frame {frame}: "
            f"{cnt} verts ({n_tris} tris) -> {obj_path}"
        )


def export_all_objs(
    out_dir: str,
    frame: int,
    iso_ratio: float = -1.0,
    fluid_material_ids=None,
):
    export_fluid_objs_per_material(
        out_dir, frame, iso_ratio=iso_ratio, fluid_material_ids=fluid_material_ids
    )
    export_rigid_objs(out_dir, frame)
