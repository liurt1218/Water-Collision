# main.py
import os
import taichi as ti

from sim import config as C
from sim.config import (
    FluidBlockConfig,
    FluidSceneConfig,
    RigidBodyConfig,
    RigidSceneConfig,
)
from sim.io_utils import load_obj
import sim.state as S
from sim.fluid import (
    compute_density,
    compute_alpha,
    init_fluid_blocks,
)
from sim.rigid import init_rigid, update_mesh_vertices
from sim.step import step


def compute_fluid_block_ranges(fluid_scene_cfg: FluidSceneConfig):
    """
    Compute (offset, count) for each enabled fluid block in the global
    particle array.

    The order matches the order used in init_fluid_blocks(), so each
    block occupies a contiguous range [offset, offset + count) in S.x.
    """
    ranges = []
    offset = 0
    for cfg in fluid_scene_cfg.blocks:
        if not cfg.enabled:
            continue
        sx, sy, sz = cfg.size
        h = cfg.particle_diameter
        nx = int(sx / h)
        ny = int(sy / h)
        nz = int(sz / h)
        n_block = nx * ny * nz
        ranges.append((offset, n_block))
        offset += n_block
    return ranges


def main(fluid_scene_cfg: FluidSceneConfig, rigid_scene_cfg: RigidSceneConfig):
    """Entry point of the DFSPH + multi-rigid-body + multi-fluid-block simulation."""
    ti.init(arch=ti.gpu, device_memory_fraction=0.8)

    # ------------------------------------------------------------------
    # Determine fluid particle count from all fluid blocks
    # ------------------------------------------------------------------
    n_fluid = C.compute_total_fluid_particles(fluid_scene_cfg)

    # ------------------------------------------------------------------
    # Collect per-body mesh vertex and index counts for rigid bodies
    # ------------------------------------------------------------------
    mesh_vert_counts = []
    mesh_index_counts = []
    for body in rigid_scene_cfg.bodies:
        verts_np, faces_np = load_obj(body.mesh_path)
        mesh_vert_counts.append(int(verts_np.shape[0]))
        mesh_index_counts.append(int(faces_np.reshape(-1).size))

    # ------------------------------------------------------------------
    # Allocate all Taichi fields (fluid + rigid + mesh)
    # ------------------------------------------------------------------
    S.allocate_fields(
        n_fluid_in=n_fluid,
        mesh_vert_counts=mesh_vert_counts,
        mesh_index_counts=mesh_index_counts,
    )

    # ------------------------------------------------------------------
    # Initialize fluid blocks and rigid bodies
    # ------------------------------------------------------------------
    if n_fluid > 0:
        init_fluid_blocks(fluid_scene_cfg)
    init_rigid(rigid_scene_cfg)

    # Precompute DFSPH alpha and initial densities
    if n_fluid > 0:
        compute_density()
        compute_alpha()

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    os.makedirs("frames", exist_ok=True)
    os.makedirs("renderings", exist_ok=True)
    os.makedirs("frames/bunny_teapot", exist_ok=True)

    # ------------------------------------------------------------------
    # Taichi GUI setup
    # ------------------------------------------------------------------
    window = ti.ui.Window(
        "DFSPH + Rigid Demo (Multi-Fluid)", (1280, 720), show_window=False
    )
    canvas = window.get_canvas()
    scene = window.get_scene()

    n_frames = 400
    substeps = 10

    # Precompute fluid block ranges for rendering (optional, for per-block colors)
    fluid_block_ranges = compute_fluid_block_ranges(fluid_scene_cfg)

    for frame in range(n_frames):
        # --------------------------------------------------------------
        # Physics substeps
        # --------------------------------------------------------------
        for _ in range(substeps):
            step()

        # --------------------------------------------------------------
        # Camera and lights
        # --------------------------------------------------------------
        camera = ti.ui.Camera()
        camera.position(1.6, 1.0, 1.6)
        camera.lookat(0.5, 0.4, 0.5)
        camera.up(0.0, 1.0, 0.0)
        scene.set_camera(camera)
        scene.ambient_light((0.4, 0.4, 0.4))
        scene.point_light((2.0, 3.0, 2.0), (1.0, 1.0, 1.0))

        # --------------------------------------------------------------
        # Render fluid particles
        # --------------------------------------------------------------
        if n_fluid > 0 and S.n_fluid > 0:
            # Option 1: render all fluid particles as a single group
            # scene.particles(
            #     S.x,
            #     radius=0.01,
            #     index_offset=0,
            #     index_count=S.n_fluid,
            #     color=(0.2, 0.6, 1.0),
            # )

            # Option 2: render each fluid block with its own color
            for b_id, (offset, count) in enumerate(fluid_block_ranges):
                if count <= 0:
                    continue

                # Simple color palette per block
                if b_id == 0:
                    color = (0.2, 0.6, 1.0)  # blue-ish
                elif b_id == 1:
                    color = (1.0, 0.4, 0.3)  # orange-red
                else:
                    color = (0.4, 1.0, 0.4)  # green-ish

                scene.particles(
                    S.x,
                    radius=0.01,
                    index_offset=offset,
                    index_count=count,
                    color=color,
                )

        # --------------------------------------------------------------
        # Render rigid meshes (each body has its own mesh slice)
        # --------------------------------------------------------------
        for b in range(S.n_rigid_bodies):
            # Update world-space vertices of body b
            update_mesh_vertices(b)
            scene.mesh(
                S.mesh_vertices,
                indices=S.mesh_indices,
                vertex_offset=S.mesh_vert_offset[b],
                vertex_count=S.mesh_vert_count[b],
                index_offset=S.mesh_index_offset[b],
                index_count=S.mesh_index_count[b],
                color=(1.0, 0.5, 0.2),
            )

        canvas.scene(scene)

        fname = f"frames/bunny_teapot/frame_{frame:04d}.png"
        window.save_image(fname)
        print("saved", fname)

    # ------------------------------------------------------------------
    # Encode frames into an MP4 video using ffmpeg
    # ------------------------------------------------------------------
    os.system(
        "ffmpeg -framerate 30 -i frames/bunny_teapot/frame_%04d.png "
        "-c:v libx264 -pix_fmt yuv420p renderings/bunny_teapot.mp4 -y"
    )
    os.system(
        "ffmpeg -framerate 8 "
        "-i frames/bunny_teapot/frame_%04d.png "
        '-vf "scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=sierra2_4a" '
        "renderings/bunny_teapot.gif -y"
    )


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Fluid scene: multiple blocks, each with independent properties
    # ------------------------------------------------------------------
    fluid_scene_cfg = FluidSceneConfig(
        blocks=[
            # Block 0
            FluidBlockConfig(
                enabled=True,
                base=(0.02, 0.02, 0.02),
                size=(0.96, 0.96, 0.24),
                particle_diameter=0.015,
                rho0=1000.0,
                surface_tension=0.04,
                viscosity=5.0,
            ),
            # Block 1 (example: smaller block with different parameters)
            # You can comment this out if you only want one block.
            # FluidBlockConfig(
            #    enabled=True,
            #    base=(0.02, 0.52, 0.02),
            #    size=(0.96, 0.48, 0.24),
            #    particle_diameter=0.02,
            #    rho0=800.0,
            #    surface_tension=0.02,
            #    viscosity=2.0,
            # ),
        ]
    )

    # ------------------------------------------------------------------
    # Rigid scene: each body can have its own mesh and parameters
    # ------------------------------------------------------------------
    rigid_scene_cfg = RigidSceneConfig(
        bodies=[
            RigidBodyConfig(
                mesh_path="obj/teapot.obj",
                center=(0.5, 0.2 + 0.4, 0.5),
                half_extents=(0.18, 0.18, 0.18),
                density=800.0,
                restitution=0.6,
                friction=0.2,
            ),
            RigidBodyConfig(
                mesh_path="obj/bunny.obj",
                center=(0.5, 0.2, 0.5),
                half_extents=(0.2, 0.2, 0.2),
                density=300.0,
                restitution=0.3,
                friction=0.3,
            ),
        ]
    )

    main(fluid_scene_cfg, rigid_scene_cfg)
