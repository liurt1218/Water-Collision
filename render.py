import numpy as np
import pysplashsurf as ps
import bpy
import state as S
import colorsys
import config as C


def rigid_surface_reconstruction(particles, particle_radius):
    mesh_with_data, _ = ps.reconstruction_pipeline(
        particles,
        particle_radius=particle_radius,  # Further reduced particle radius for finer resolution
        rest_density=1000.0,
        smoothing_length=1.5,  # Reduced smoothing length for sharper details
        cube_size=0.1,  # Smaller cube size for capturing intricate details
        iso_surface_threshold=0.4,  # Lower threshold for finer surface details
        mesh_smoothing_weights=True,
        mesh_smoothing_weights_normalization=8.0,  # Adjusted for sharper smoothing
        mesh_smoothing_iters=40,  # Increased iterations for even smoother mesh
        normals_smoothing_iters=20,  # Increased iterations for smoother normals
        mesh_cleanup=True,
        compute_normals=True,
        subdomain_grid=True,
        subdomain_num_cubes_per_dim=256,  # Further increased grid size for maximum detail
        output_mesh_smoothing_weights=True,
    )
    return mesh_with_data


def get_material_for_rigid(i: int) -> bpy.types.Material:
    name = f"RigidMat_{i}"
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True

    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")

    if i == 0:
        r, g, b = 0.6, 0.3, 0.9
    elif i == 1:
        r, g, b = 0.5, 0.05, 0.05
    elif i == 2:
        r, g, b = 0.1, 0.15, 0.4
    else:
        hue = (i * 0.6180339887) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 1.0)

    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)

    bsdf.inputs["Roughness"].default_value = 0.3 + 0.5 * ((i % 3) / 2.0)
    bsdf.inputs["Metallic"].default_value = 0.2 * (i % 2)

    return mat


def add_outer_walls(wall_material, thickness=0.02):
    t = thickness

    def add_wall(location, scale):
        bpy.ops.mesh.primitive_cube_add(size=1, location=location, scale=scale)
        obj = bpy.context.object
        obj.data.materials.append(wall_material)

    add_wall(
        location=(0.5, 0.5, -t / 2),
        scale=(1.0, 1.0, t),
    )

    add_wall(
        location=(0.5, 0.5, 1 + t / 2),
        scale=(1.0, 1.0, t),
    )

    add_wall(
        location=(-t / 2, 0.5, 0.5),
        scale=(t, 1.0, 1.0),
    )

    add_wall(
        location=(1 + t / 2, 0.5, 0.5),
        scale=(t, 1.0, 1.0),
    )

    add_wall(
        location=(0.5, -t / 2, 0.5),
        scale=(1.0, t, 1.0),
    )

    add_wall(
        location=(0.5, 1 + t / 2, 0.5),
        scale=(1.0, t, 1.0),
    )


def render_rigid_body(particle_radii):
    if S.n_mesh_vertices == 0 or S.n_rigid_bodies == 0:
        return

    offsets = S.rb_mesh_vert_offset.to_numpy()
    counts = S.rb_mesh_vert_count.to_numpy()
    # f_offsets = S.rb_mesh_face_offset.to_numpy()
    # f_counts = S.rb_mesh_face_count.to_numpy()
    all_vertices = S.mesh_vertices_t.detach().cpu().numpy()
    # all_faces = S.mesh_local_faces.to_numpy()

    for i in range(S.n_rigid_bodies):
        start = offsets[i]
        end = start + counts[i]
        vertices = all_vertices[start:end]

        mesh_with_data = rigid_surface_reconstruction(vertices, particle_radii[i])
        # start = f_offsets[i]
        # end = start + f_counts[i]
        # triangles = all_faces[start:end] + 1

        mesh = bpy.data.meshes.new(f"RigidBodyMesh_{i}")
        mesh.from_pydata(
            mesh_with_data.mesh.vertices, [], mesh_with_data.mesh.triangles
        )
        mesh.update()

        obj = bpy.data.objects.new(f"RigidBody_{i}", mesh)
        bpy.context.collection.objects.link(obj)

        mat = get_material_for_rigid(i)
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    wall_material = bpy.data.materials.new(name="WallMaterial")
    wall_material.use_nodes = True
    wall_material.node_tree.nodes["Principled BSDF"].inputs[
        "Base Color"
    ].default_value = (1.0, 1.0, 1.0, 1.0)
    wall_material.node_tree.nodes["Principled BSDF"].inputs[
        "Metallic"
    ].default_value = 0.0
    wall_material.node_tree.nodes["Principled BSDF"].inputs[
        "Roughness"
    ].default_value = 0.0
    wall_material.node_tree.nodes["Principled BSDF"].inputs["IOR"].default_value = 1.4
    wall_material.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 1.0
    wall_material.node_tree.nodes["Principled BSDF"].inputs[
        "Transmission Weight"
    ].default_value = 0.97
    add_outer_walls(wall_material, thickness=0.02)


def fluid_surface_reconstruction(particles):
    mesh_with_data, _ = ps.reconstruction_pipeline(
        particles,
        particle_radius=0.015,
        rest_density=1000.0,
        smoothing_length=2.0,
        cube_size=0.5,
        iso_surface_threshold=0.6,
        mesh_smoothing_weights=True,
        mesh_smoothing_weights_normalization=13.0,
        mesh_smoothing_iters=40,
        normals_smoothing_iters=10,
        mesh_cleanup=True,
        compute_normals=True,
        subdomain_grid=True,
        subdomain_num_cubes_per_dim=64,
        output_mesh_smoothing_weights=True,
    )
    return mesh_with_data


def render_fluid():
    particles = S.x.to_numpy()
    color_dict = {}

    for i in range(len(particles)):
        color_tp = tuple(S.color[i].to_list() + [1.0])
        if color_tp in color_dict:
            color_dict[color_tp].append(i)
        else:
            color_dict[color_tp] = [i]

    for i, (color, mat_ids) in enumerate(color_dict.items()):
        print(
            f"Rendering fluid material {i} with color {color} and {len(mat_ids)} particles."
        )
        if len(mat_ids) == 0:
            continue

        selected_particles = particles[mat_ids]

        mesh_with_data = fluid_surface_reconstruction(selected_particles)

        surf_material = bpy.data.materials.new(name=f"SurfMaterial_{i}")
        surf_material.use_nodes = True
        surf_material.node_tree.nodes["Principled BSDF"].inputs[
            "Base Color"
        ].default_value = color
        surf_material.node_tree.nodes["Principled BSDF"].inputs[
            "Metallic"
        ].default_value = 0.0
        surf_material.node_tree.nodes["Principled BSDF"].inputs[
            "Roughness"
        ].default_value = 0.0
        surf_material.node_tree.nodes["Principled BSDF"].inputs[
            "IOR"
        ].default_value = 1.333
        surf_material.node_tree.nodes["Principled BSDF"].inputs[
            "Alpha"
        ].default_value = 1.0
        surf_material.node_tree.nodes["Principled BSDF"].inputs[
            "Transmission Weight"
        ].default_value = 1.0

        mesh = bpy.data.meshes.new(f"FluidMesh_{i}")
        mesh.from_pydata(
            mesh_with_data.mesh.vertices, [], mesh_with_data.mesh.triangles
        )
        mesh.update()

        obj = bpy.data.objects.new(f"Fluid_{i}", mesh)
        bpy.context.collection.objects.link(obj)

        obj.data.materials.append(surf_material)


def render_frame(frame, output_dir, particle_radii):
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Add a camera
    bpy.ops.object.camera_add(
        location=(1.6, 1.0, 1.6), rotation=(-np.pi / 8, np.pi / 4, 0)
    )
    camera = bpy.context.object
    camera.data.lens = 35.0
    bpy.context.scene.camera = camera

    # Add lighting
    bpy.ops.object.light_add(type="SUN", rotation=(-np.pi / 2, 0, 0))
    light1 = bpy.context.object
    light1.data.energy = 2.0

    bpy.ops.object.light_add(type="SUN")
    light2 = bpy.context.object
    light2.data.energy = 2.0

    if S.N_RIGID > 0:
        render_rigid_body(particle_radii)
    if C.n_particles > 0:
        render_fluid()

    hdr_path = "assets/charolettenbrunn_park_4k.hdr"
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    env_texture = nodes.new(type="ShaderNodeTexEnvironment")
    env_texture.image = bpy.data.images.load(hdr_path)
    background = nodes["Background"]
    world.node_tree.links.new(env_texture.outputs["Color"], background.inputs["Color"])

    # Set render settings
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 128
    bpy.context.scene.render.resolution_x = 800
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.filepath = f"frames/{output_dir}/frame_{frame:04d}.png"
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"

    # Render the scene
    bpy.ops.render.render(write_still=True)
    print(f"Rendered frame saved to frames/{output_dir}/frame_{frame:04d}.png")
