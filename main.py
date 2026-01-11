import os
import taichi as ti
import numpy as np
import json
import tqdm

import config as C
import materials as M

# Taichi initialization
ti.init(arch=ti.gpu)  # or ti.cpu

import state as S
import fluid
import step
import rigid
import render
import argparse


# CLI parsing
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-config",
        type=str,
        default="scene_config.json",
        help="Scene config JSON path",
    )
    return parser.parse_args()


# Material setup
def build_materials(scene_cfg):
    # Define and register all materials used in this scene.
    name_to_id = {}
    fbs = scene_cfg.get("fluids", [])

    for fb in fbs:
        config = M.MaterialConfig(
            rho0=fb["rho"],
            E=fb["E"],
            nu=fb["nu"],
            kind=fb["kind"],
            name=fb["name"],
            eta=fb["eta"],
        )
        mat_id = M.global_registry.register(config)
        name_to_id[fb["name"]] = mat_id

    if len(fbs) > 0:
        # Build kernel tables
        M.build_kernel_tables()

    return name_to_id


# OBJ loading + preprocessing: scale into user bbox
def load_obj_vertices_and_faces(obj_path: str):
    # Minimal OBJ loader: parse 'v' and 'f' lines.
    verts = []
    faces = []

    with open(obj_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    verts.append([x, y, z])
            elif line.startswith("f "):
                parts = line.split()[1:]
                # Each part can be like "i", "i/j", "i/j/k", etc.
                idx = []
                for p in parts:
                    token = p.split("/")[0]
                    if token == "":
                        continue
                    # OBJ is 1-based index
                    idx.append(int(token) - 1)
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) == 4:
                    # Triangulate quad: (0,1,2) and (0,2,3)
                    faces.append([idx[0], idx[1], idx[2]])
                    faces.append([idx[0], idx[2], idx[3]])
                # Ignore polygons with >4 vertices for now

    if not verts or not faces:
        raise RuntimeError(f"[OBJ] Failed to load vertices/faces from: {obj_path}")

    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    return verts, faces


def compute_local_mesh_in_bbox(
    verts_raw: np.ndarray,
    half_extents_ratio: np.ndarray,
):
    # Scale and center the raw OBJ vertices using *per-axis scale ratios*.
    vmin = verts_raw.min(axis=0)
    vmax = verts_raw.max(axis=0)
    ext_raw = vmax - vmin  # original extents (x,y,z)
    center_raw = 0.5 * (vmin + vmax)

    # Original half extents of the mesh bbox
    half_raw = 0.5 * ext_raw  # (ext_x/2, ext_y/2, ext_z/2)

    # Per-axis scale ratios (sx, sy, sz)
    s = np.array(half_extents_ratio, dtype=np.float32)

    # Local vertices: center at origin, then per-axis scaling
    verts_local = (verts_raw - center_raw) * s

    # Physical half extents: original half extents * scale ratios
    phys_half_extents = half_raw * s  # (hx, hy, hz)

    return verts_local, phys_half_extents


def load_scene_config(path: str):
    with open(path, "r") as f:
        return json.load(f)


def build_scene_from_config(name_to_id, cfg):
    blocks = []
    for fb in cfg.get("fluids", []):
        ftype = fb.get("type", "cube")
        if ftype == "cube":
            min_corner = ti.Vector(fb["min_corner"])
            size = ti.Vector(fb["size"])
            mat_name = fb["name"]
            if mat_name not in name_to_id:
                raise ValueError(f"Unknown fluid material name: {mat_name}")
            mat_id = name_to_id[mat_name]
            blocks.append(
                fluid.CubeVolume(
                    min_corner,
                    size,
                    mat_id,
                    fb["kind"],
                    fb.get("color", ti.Vector([0.0, 0.0, 0.0])),
                    int("color" in fb.keys()),
                )
            )
        else:
            raise NotImplementedError(f"Unsupported fluid type: {ftype}")
    if len(blocks) > 0:
        fluid.init_vols(blocks)


# Rigid scene init: user configs + Taichi rigid fields + mesh cache
def init_rigid_scene_from_user_configs(cfg):
    user_rigids_cfg = cfg.get("rigids", [])
    user_rigids = []
    for r in user_rigids_cfg:
        user_rigids.append(
            {
                "name": r["name"],
                "obj_path": r["obj_path"],
                "position": np.array(r["position"], dtype=np.float32),
                "half_extents": np.array(r["half_extents"], dtype=np.float32),
                "density": float(r.get("density", 500.0)),
                "restitution": float(r.get("restitution", 0.2)),
                "friction": float(r.get("friction", 0.3)),
            }
        )

    n_rigid = len(user_rigids)
    if n_rigid == 0:
        print("[INFO] No rigid bodies configured.")
        return

    # 1) Allocate rigid-body physics fields and clear them
    S.init_rigid_fields(n_rigid)
    rigid.clear_rigid_bodies()

    # 2) First pass: build triangle-soup arrays and per-rigid offsets
    tri_vertices_all = []
    tri_normals_all = []
    vert_offsets = []
    vert_counts = []
    phys_half_extents_list = []
    faces_all = []
    faces_offsets = []
    faces_counts = []

    current_offset = 0
    face_offset = 0

    for cfg_r in user_rigids:
        obj_path = cfg_r["obj_path"]
        pos = cfg_r["position"]
        half_extents = cfg_r["half_extents"]
        density = cfg_r["density"]
        restitution = cfg_r["restitution"]
        friction = cfg_r["friction"]

        # Load OBJ vertices/faces
        verts_raw, faces = load_obj_vertices_and_faces(obj_path)  # (V,3) and (F,3)

        # Scale into bbox in local space
        verts_local_full, phys_half_extents = compute_local_mesh_in_bbox(
            verts_raw,
            half_extents,  # now interpreted as per-axis scale ratios
        )
        phys_half_extents_list.append(phys_half_extents.astype(np.float32))

        # Flatten triangles into vertex list (triangle soup), faces: (F,3), tri_verts_local: (F*3,3)
        tri_verts_local = verts_local_full[faces.reshape(-1)]

        # Flat-shaded normals per triangle
        F = faces.shape[0]
        tri_normals_local = np.zeros_like(tri_verts_local, dtype=np.float32)
        for f in range(F):
            i0, i1, i2 = faces[f]
            p0 = verts_local_full[i0]
            p1 = verts_local_full[i1]
            p2 = verts_local_full[i2]
            n = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(n)
            if norm > 1e-8:
                n = n / norm
            tri_normals_local[3 * f + 0] = n
            tri_normals_local[3 * f + 1] = n
            tri_normals_local[3 * f + 2] = n

        n_tri_verts = tri_verts_local.shape[0]

        vert_offsets.append(current_offset)
        vert_counts.append(n_tri_verts)
        faces_offsets.append(face_offset)
        faces_counts.append(F)

        tri_vertices_all.append(tri_verts_local)
        tri_normals_all.append(tri_normals_local)
        faces_all.append(faces)

        current_offset += n_tri_verts
        face_offset += F

    # Concatenate all rigid meshes into one big buffer
    tri_vertices_all = np.concatenate(tri_vertices_all, axis=0).astype(np.float32)
    tri_normals_all = np.concatenate(tri_normals_all, axis=0).astype(np.float32)
    faces_all = np.concatenate(faces_all, axis=0).astype(np.int32)

    total_verts = tri_vertices_all.shape[0]
    total_faces = tri_normals_all.shape[0] // 3

    # 3) Allocate Taichi mesh fields and upload data
    S.init_rigid_mesh_fields(total_verts, n_rigid, total_faces)
    S.mesh_local_vertices.from_numpy(tri_vertices_all)
    S.mesh_local_normals.from_numpy(tri_normals_all)
    S.mesh_local_faces.from_numpy(faces_all)
    S.rb_mesh_vert_offset.from_numpy(np.array(vert_offsets, dtype=np.int32))
    S.rb_mesh_vert_count.from_numpy(np.array(vert_counts, dtype=np.int32))
    S.rb_mesh_face_offset.from_numpy(np.array(faces_offsets, dtype=np.int32))
    S.rb_mesh_face_count.from_numpy(np.array(faces_counts, dtype=np.int32))

    # 4) Initialize rigid-body physics for each config (bbox collider + inertia)
    for idx, cfg_r in enumerate(user_rigids):
        pos = cfg_r["position"]
        density = cfg_r["density"]
        restitution = cfg_r["restitution"]
        friction = cfg_r["friction"]

        # Use *physical* half extents (after scaling), not the raw ratios
        phys_he = phys_half_extents_list[idx]  # (3,)
        hx, hy, hz = phys_he.tolist()

        cx, cy, cz = pos.tolist()

        volume_box = (2.0 * hx) * (2.0 * hy) * (2.0 * hz)
        mass = density * volume_box

        rigid.init_rigid_from_bbox(
            idx,
            cx,
            cy,
            cz,
            hx,
            hy,
            hz,
            mass,
            restitution,
            friction,
        )
    # copy to torch.tensor
    S.init_rigid_torch_storage()

    print(
        f"[INFO] Initialized {n_rigid} rigid bodies, total mesh vertices = {total_verts}."
    )


# Main loop
def main():
    args = parse_args()
    scene_cfg = load_scene_config(args.scene_config)
    output_dir = scene_cfg["out_dir"]

    # Initialize configurations
    C.init_config(scene_cfg["particles"], scene_cfg["grids"])
    S.init_particle_level_fields()
    S.init_grid_level_fields()

    # 1. Define materials (user can edit rho0/E/nu here)
    name_to_id = build_materials(scene_cfg)

    # 2. Initialize fluid scene (multiple blocks)
    build_scene_from_config(name_to_id, scene_cfg)

    # 3. Initialize rigid bodies from user configs (OBJ + bbox + parameters)
    init_rigid_scene_from_user_configs(scene_cfg)

    particle_radii = []
    user_rigids_cfg = scene_cfg.get("rigids", [])
    for r in user_rigids_cfg:
        particle_radii.append(float(r.get("particle_radius", 0.01)))

    # 4. Prepare output directory
    os.makedirs(f"frames/{output_dir}", exist_ok=True)
    os.makedirs(f"renderings", exist_ok=True)

    # 5. Main loop with fixed frame count
    for frame in tqdm.tqdm(range(scene_cfg["sim_steps"])):
        # Physics substeps
        for _ in range(scene_cfg["substeps"]):
            step.substep(C.gravity)

        try:
            render.render_frame(
                frame,
                output_dir,
                particle_radii=particle_radii,
            )
        except Exception as e:
            print(f"Error rendering frame {frame} with error {e}")

    # 6. Encode video with ffmpeg
    os.system(
        f"ffmpeg -framerate 30 -i frames/{output_dir}/frame_%04d.png "
        f"-c:v libx264 -pix_fmt yuv420p renderings/{output_dir}.mp4 -y"
    )
    os.system(
        f'ffmpeg -framerate 10 -i frames/{output_dir}/frame_%04d.png -vf "fps=10,scale=512:-1,palettegen=stats_mode=full" palette.png -y '
        f'&& ffmpeg -framerate 10 -i frames/{output_dir}/frame_%04d.png -i palette.png -lavfi "fps=10,scale=512:-1,paletteuse=dither=bayer:bayer_scale=4" renderings/{output_dir}.gif -y'
    )


if __name__ == "__main__":
    main()
