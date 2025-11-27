# sim/surface.py
import taichi as ti
import numpy as np
import mcubes  # pip install PyMCubes

from . import config as C
from . import state as S
from .fluid import kernel_W

# ================================================================
# Grid parameters for scalar field (for fluid surface)
# ================================================================
N_X = 80
N_Y = 80
N_Z = 80

# Assume domain is an axis-aligned box and grid spacing is uniform.
DX = (C.domain_max[0] - C.domain_min[0]) / N_X

# Scalar field (will be allocated lazily after ti.init())
phi = None

# Internal flag and kernel holder for lazy initialization
_surface_initialized = False
_build_scalar_field_kernel = None


def init_surface_module():
    """
    Lazily initialize Taichi fields and kernels for surface reconstruction.

    This must be called after ti.init(). We call it automatically inside
    reconstruct_fluid_surface() / export_scene_obj(), so the user does not
    need to call it manually.
    """
    global phi, _surface_initialized, _build_scalar_field_kernel

    if _surface_initialized:
        return

    # Allocate scalar field
    phi = ti.field(dtype=ti.f32, shape=(N_X, N_Y, N_Z))

    @ti.kernel
    def build_scalar_field_kernel():
        """
        Build a 3D scalar field from fluid particles using the SPH kernel.
        The field roughly represents fluid density in a regular grid.
        """
        # Clear scalar field
        for I in ti.grouped(phi):
            phi[I] = 0.0

        # Accumulate contributions from each fluid particle
        for i in range(S.n_particles):
            if S.is_fluid[i] == 1:
                xp = S.x[i]
                hi = S.fluid_support_radius[i]
                vol_i = S.rest_volume[i]

                base = ((xp - ti.Vector(C.domain_min)) / DX).cast(int)
                r = int(hi / DX) + 1

                for off in ti.grouped(
                    ti.ndrange((-r, r + 1), (-r, r + 1), (-r, r + 1))
                ):
                    g = base + off
                    if 0 <= g[0] < N_X and 0 <= g[1] < N_Y and 0 <= g[2] < N_Z:
                        xg = ti.Vector(C.domain_min) + (g.cast(ti.f32) + 0.5) * DX
                        R = xp - xg
                        dist = R.norm()
                        if dist < hi:
                            w = kernel_W(dist, hi)
                            phi[g] += vol_i * w

    _build_scalar_field_kernel = build_scalar_field_kernel
    _surface_initialized = True


def reconstruct_fluid_surface(iso_ratio: float = 0.5):
    """
    Reconstruct the fluid surface as a triangle mesh using marching cubes.

    Parameters
    ----------
    iso_ratio : float
        Ratio in [0, 1]. The isosurface level is chosen as
        c = iso_ratio * max(phi). Larger values make the fluid look "thicker".

    Returns
    -------
    verts_world : (N, 3) float32 array
        World-space vertex positions of the reconstructed surface.
    faces : (M, 3) int32 array
        Triangle indices (0-based) into verts_world.
    """
    # Ensure fields and kernels are initialized (after ti.init())
    init_surface_module()

    # Build scalar field on the GPU
    _build_scalar_field_kernel()

    # Move scalar field to NumPy
    phi_np = phi.to_numpy()

    phi_max = float(phi_np.max())
    if phi_max <= 0.0:
        # No valid fluid contribution
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    c = iso_ratio * phi_max
    phi_shifted = c - phi_np  # isosurface at phi_shifted == 0

    # Marching cubes on CPU
    verts, faces = mcubes.marching_cubes(phi_shifted, 0.0)

    domain_min = np.array(C.domain_min, dtype=np.float32)

    verts_world = np.empty_like(verts, dtype=np.float32)
    verts_world[:, 0] = domain_min[0] + (verts[:, 0] + 0.5) * DX
    verts_world[:, 1] = domain_min[1] + (verts[:, 1] + 0.5) * DX
    verts_world[:, 2] = domain_min[2] + (verts[:, 2] + 0.5) * DX

    return verts_world.astype(np.float32), faces.astype(np.int32)


# ======================================================================
# New: export fluid-only OBJ
# ======================================================================
def export_fluid_obj(path: str, iso_ratio: float = 0.5):
    """
    Export only the fluid surface of the current scene to an OBJ file.

    This uses marching cubes over the scalar field built from all fluid
    particles. Rigid bodies are not included.
    """
    verts, faces = reconstruct_fluid_surface(iso_ratio=iso_ratio)

    if verts.shape[0] == 0 or faces.shape[0] == 0:
        print(f"[fluid_export] no verts or faces, skip exporting {path}")
        return

    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            # faces are 0-based; OBJ uses 1-based indexing
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")

    print(
        f"[fluid_export] exported {verts.shape[0]} verts, "
        f"{faces.shape[0]} faces to {path}"
    )


# ======================================================================
# New: export rigid-only OBJ for a single body
# ======================================================================
def export_rigid_obj(path: str, body_id: int):
    """
    Export a single rigid body mesh (in world space) to an OBJ file.

    Parameters
    ----------
    path : str
        Output OBJ file path.
    body_id : int
        Index of the rigid body (0 <= body_id < S.n_rigid_bodies).

    Notes
    -----
    - This assumes update_mesh_vertices(body_id) has already been called
      for the current frame so that S.mesh_vertices contains world-space
      positions for this body.
    """
    if body_id < 0 or body_id >= S.n_rigid_bodies:
        print(f"[rigid_export] invalid body_id={body_id}, skip exporting {path}")
        return

    verts_np = S.mesh_vertices.to_numpy()
    indices_np = S.mesh_indices.to_numpy()

    v_off_b = int(S.mesh_vert_offset[body_id])
    v_cnt_b = int(S.mesh_vert_count[body_id])
    i_off_b = int(S.mesh_index_offset[body_id])
    i_cnt_b = int(S.mesh_index_count[body_id])

    if v_cnt_b <= 0 or i_cnt_b <= 0:
        print(f"[rigid_export] body {body_id} has no verts or faces, skip {path}")
        return

    # Collect vertices for this body (local list)
    local_verts = []
    for k in range(v_cnt_b):
        v = verts_np[v_off_b + k]
        local_verts.append([float(v[0]), float(v[1]), float(v[2])])

    # Collect faces for this body
    local_faces = []
    for k in range(0, i_cnt_b, 3):
        li0 = int(indices_np[i_off_b + k + 0])
        li1 = int(indices_np[i_off_b + k + 1])
        li2 = int(indices_np[i_off_b + k + 2])

        local_faces.append([li0 + 1, li1 + 1, li2 + 1])

    with open(path, "w") as f:
        for v in local_verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in local_faces:
            f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")

    print(
        f"[rigid_export] body {body_id}: exported {len(local_verts)} verts, "
        f"{len(local_faces)} faces to {path}"
    )


def export_scene_obj(path: str, iso_ratio: float = 0.5):
    """
    Export the whole scene (fluid surface + all rigid meshes) as a single OBJ.

    Notes
    -----
    - This function assumes that S.mesh_vertices already store rigid-body
      vertices in world space for the current frame. In other words,
      update_mesh_vertices(b) must have been called for all rigid bodies
      before calling this function.
    - Rigid mesh indices are stored globally in S.mesh_indices, while
      mesh_vert_offset[b] / mesh_vert_count[b] define each body's vertex
      slice. We convert global indices to local ones when appending.
    """
    # Reconstruct fluid surface first
    fluid_verts, fluid_faces = reconstruct_fluid_surface(iso_ratio=iso_ratio)

    all_verts = []
    all_faces = []

    # Append fluid mesh
    if fluid_verts.shape[0] > 0 and fluid_faces.shape[0] > 0:
        all_verts.extend(fluid_verts.tolist())
        for tri in fluid_faces:
            all_faces.append([int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1])
    v_offset = len(all_verts)

    # Append rigid meshes (world-space)
    verts_np = S.mesh_vertices.to_numpy()
    indices_np = S.mesh_indices.to_numpy()

    for b in range(S.n_rigid_bodies):
        v_off_b = int(S.mesh_vert_offset[b])
        v_cnt_b = int(S.mesh_vert_count[b])
        i_off_b = int(S.mesh_index_offset[b])
        i_cnt_b = int(S.mesh_index_count[b])

        if v_cnt_b <= 0 or i_cnt_b <= 0:
            continue

        # Append vertices of this body
        for k in range(v_cnt_b):
            v = verts_np[v_off_b + k]
            all_verts.append([float(v[0]), float(v[1]), float(v[2])])

        # Append faces of this body
        # indices_np are GLOBAL indices into verts_np.
        # We convert them to LOCAL indices within this body's vertex slice,
        # then shift by v_offset to match the appended vertices in all_verts.
        local_v_offset = v_offset

        for k in range(0, i_cnt_b, 3):
            i0 = int(indices_np[i_off_b + k + 0])
            i1 = int(indices_np[i_off_b + k + 1])
            i2 = int(indices_np[i_off_b + k + 2])

            all_faces.append(
                [
                    local_v_offset + i0 + 1,
                    local_v_offset + i1 + 1,
                    local_v_offset + i2 + 1,
                ]
            )

        v_offset = len(all_verts)

    # Write OBJ file
    if len(all_verts) == 0 or len(all_faces) == 0:
        print(f"[scene_export] no verts or faces, skip exporting {path}")
        return

    with open(path, "w") as f:
        for v in all_verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in all_faces:
            f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")

    print(
        f"[scene_export] exported {len(all_verts)} verts, "
        f"{len(all_faces)} faces to {path}"
    )
