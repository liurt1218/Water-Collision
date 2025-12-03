import taichi as ti
import math
import os
import numpy as np
import pysplashsurf as ps
import bpy

import state as S

def rigid_surface_reconstruction(particles):
    mesh_with_data, _ = ps.reconstruction_pipeline(
        particles,
        particle_radius=0.01,  # Further reduced particle radius for finer resolution
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
        output_mesh_smoothing_weights=True
    )
    return mesh_with_data

def render_rigid_body():
    if S.n_mesh_vertices == 0 or S.n_rigid_bodies == 0:
        return
    
    cube_material = bpy.data.materials.new(name="CubeMaterial")
    cube_material.use_nodes = True
    cube_material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1.0, 0.5, 0.2, 1.0)
    cube_material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 1.0

    
    offsets = S.rb_mesh_vert_offset.to_numpy()
    counts = S.rb_mesh_vert_count.to_numpy()

    for i in range(S.n_rigid_bodies):
        start = offsets[i]
        end = start + counts[i]
        vertices = S.mesh_vertices.to_numpy()[start:end]

        mesh_with_data = rigid_surface_reconstruction(vertices)

        mesh = bpy.data.meshes.new(f"RigidBodyMesh_{i}")
        mesh.from_pydata(mesh_with_data.mesh.vertices, [], mesh_with_data.mesh.triangles)
        mesh.update()

        obj = bpy.data.objects.new(f"RigidBody_{i}", mesh)
        bpy.context.collection.objects.link(obj)

        obj.data.materials.append(bpy.data.materials["CubeMaterial"])

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
        output_mesh_smoothing_weights=True
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
        print(f"Rendering fluid material {i} with color {color} and {len(mat_ids)} particles.")
        if len(mat_ids) == 0:
            continue

        selected_particles = particles[mat_ids]
        
        mesh_with_data = fluid_surface_reconstruction(selected_particles)

        surf_material = bpy.data.materials.new(name=f"SurfMaterial_{i}")
        surf_material.use_nodes = True
        surf_material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = color
        surf_material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.5
        surf_material.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.3

        mesh = bpy.data.meshes.new(f"FluidMesh_{i}")
        mesh.from_pydata(mesh_with_data.mesh.vertices, [], mesh_with_data.mesh.triangles)
        mesh.update()

        obj = bpy.data.objects.new(f"Fluid_{i}", mesh)
        bpy.context.collection.objects.link(obj)

        obj.data.materials.append(surf_material)

def render_frame(frame, output_dir):
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Add a camera
    bpy.ops.object.camera_add(location=(1.6, 1.0, 1.6), rotation=(-np.pi/6, np.pi/4, 0))
    camera = bpy.context.object
    camera.data.lens = 20.0
    bpy.context.scene.camera = camera

    # Add lighting
    bpy.ops.object.light_add(type='SUN', rotation=(-np.pi/2, 0, 0))
    light1 = bpy.context.object
    light1.data.energy = 2.0

    bpy.ops.object.light_add(type='SUN')
    light2 = bpy.context.object
    light2.data.energy = 2.0

    render_rigid_body()
    render_fluid()

    # Set render settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 128
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.filepath = f"frames/{output_dir}/frame_{frame:04d}.png"
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # Render the scene
    bpy.ops.render.render(write_still=True)
    print(f"Rendered frame saved to frames/{output_dir}/frame_{frame:04d}.png")