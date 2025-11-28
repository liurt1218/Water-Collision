# render.py
import bpy
import os
import re
import sys
import argparse
from mathutils import Vector


def parse_args():
    """
    Parse command-line arguments passed after '--'.
    This is required when running scripts via Blender.
    """
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="DFSPH + Rigid simulation renderer")
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Project name used as output directory prefix",
    )
    return parser.parse_args(argv)


args = parse_args()

# ==============================================================
# User configuration
# ==============================================================
OBJ_DIR = f"meshes/{args.project}"
# Pattern: scene_<tag>_<frame>.obj, e.g. scene_fluid_0_0000.obj, scene_rigid_1_0000.obj
OBJ_PATTERN = r"scene_([a-zA-Z0-9_]+)_(\d{4})\.obj"
OUT_DIR = f"blender_frames/{args.project}"

FPS = 30
CYCLES_SAMPLES = 96

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("blender_renderings", exist_ok=True)

# ==============================================================
# Clean default scene
# ==============================================================
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

scene = bpy.context.scene

# ==============================================================
# Render settings
# ==============================================================
scene.render.engine = "CYCLES"
scene.cycles.samples = CYCLES_SAMPLES
scene.render.fps = FPS
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = OUT_DIR + "/frame_"
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100

# Enable Cycles denoising for cleaner images
view_layer = bpy.context.view_layer
if hasattr(view_layer, "cycles"):
    view_layer.cycles.use_denoising = True

# Color management
scene.view_settings.view_transform = "Filmic"
scene.view_settings.look = "Medium High Contrast"

# ==============================================================
# Camera
# ==============================================================
cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

cam_data.clip_start = 0.001
cam_data.clip_end = 10000.0

cam.location = (0.0, -5.0, 3.0)
cam.rotation_euler = (1.1, 0.0, 0.0)

# ==============================================================
# Lights
# ==============================================================
light_data = bpy.data.lights.new(name="KeyLight", type="AREA")
light = bpy.data.objects.new("KeyLight", light_data)
scene.collection.objects.link(light)
light.location = (3.0, -4.0, 5.0)
light_data.energy = 1800.0
light_data.use_shadow = True
light_data.shadow_soft_size = 0.6

fill_data = bpy.data.lights.new(name="FillLight", type="POINT")
fill = bpy.data.objects.new("FillLight", fill_data)
scene.collection.objects.link(fill)
fill.location = (-2.0, 3.0, 4.0)
fill_data.energy = 200.0
fill_data.use_shadow = False

# ==============================================================
# World background
# ==============================================================
world = scene.world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg is not None:
    bg.inputs["Color"].default_value = (0.03, 0.03, 0.04, 1.0)
    bg.inputs["Strength"].default_value = 1.5

# ==============================================================
# Ground plane
# ==============================================================
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.5, -0.5, -0.02))
ground_obj = bpy.context.active_object
ground_obj.name = "Ground"
ground_obj.scale = (5.0, 5.0, 1.0)
ground_mesh = ground_obj.data

ground_mat = bpy.data.materials.new(name="Ground")
ground_mat.use_nodes = True
g_nodes = ground_mat.node_tree.nodes
g_links = ground_mat.node_tree.links

g_bsdf = g_nodes.get("Principled BSDF")
g_output = None
for n in g_nodes:
    if n.type == "OUTPUT_MATERIAL":
        g_output = n
        break

if g_bsdf is None:
    g_bsdf = g_nodes.new(type="ShaderNodeBsdfPrincipled")
if g_output is None:
    g_output = g_nodes.new(type="ShaderNodeOutputMaterial")

base = g_bsdf.inputs.get("Base Color")
if base is not None:
    base.default_value = (0.12, 0.12, 0.12, 1.0)

g_rough = g_bsdf.inputs.get("Roughness")
if g_rough is not None:
    g_rough.default_value = 0.8

g_links.new(g_bsdf.outputs["BSDF"], g_output.inputs["Surface"])
ground_mesh.materials.append(ground_mat)

# Smooth shading for ground
for poly in ground_mesh.polygons:
    poly.use_smooth = True
ground_mesh.use_auto_smooth = True

# ==============================================================
# Materials: fluid and rigid
# ==============================================================
# Fluid: glassy water with volume absorption
water_mat = bpy.data.materials.new(name="Water")
water_mat.use_nodes = True
w_nodes = water_mat.node_tree.nodes
w_links = water_mat.node_tree.links

w_bsdf = w_nodes.get("Principled BSDF")
w_output = None
for n in w_nodes:
    if n.type == "OUTPUT_MATERIAL":
        w_output = n
        break

if w_bsdf is None:
    w_bsdf = w_nodes.new(type="ShaderNodeBsdfPrincipled")
if w_output is None:
    w_output = w_nodes.new(type="ShaderNodeOutputMaterial")

w_color = w_bsdf.inputs.get("Base Color")
if w_color is not None:
    w_color.default_value = (0.2, 0.6, 1.0, 1.0)

w_trans = w_bsdf.inputs.get("Transmission")
if w_trans is not None:
    w_trans.default_value = 1.0

w_ior = w_bsdf.inputs.get("IOR")
if w_ior is not None:
    w_ior.default_value = 1.333

w_rough = w_bsdf.inputs.get("Roughness")
if w_rough is not None:
    w_rough.default_value = 0.05

w_spec = w_bsdf.inputs.get("Specular")
if w_spec is not None:
    w_spec.default_value = 0.6

# Surface connection
w_links.new(w_bsdf.outputs["BSDF"], w_output.inputs["Surface"])

# Multiple fluid materials with different colors for different fluid blocks
FLUID_COLORS = [
    (0.2, 0.6, 1.0, 1.0),  # fluid_0: blue-ish
    (1.0, 0.3, 0.3, 1.0),  # fluid_1: red-ish
    (0.3, 0.9, 0.5, 1.0),  # fluid_2: green-ish
    (0.9, 0.8, 0.3, 1.0),  # fluid_3: yellow-ish
]

fluid_mats = []

for i, col in enumerate(FLUID_COLORS):
    if i == 0:
        # Reuse the main water_mat for fluid_0
        mat = water_mat
    else:
        # Copy node setup from water_mat to keep the same look
        mat = water_mat.copy()
        mat.name = f"Water_{i}"
    nodes = mat.node_tree.nodes

    # Set base color
    bsdf = None
    for n in nodes:
        if n.type == "BSDF_PRINCIPLED":
            bsdf = n
            break
    if bsdf is not None:
        base = bsdf.inputs.get("Base Color")
        if base is not None:
            base.default_value = col

    # Optional: slightly tint volume absorption color as well
    abs_node_local = None
    for n in nodes:
        if n.type == "VOLUME_ABSORPTION":
            abs_node_local = n
            break
    if abs_node_local is not None:
        abs_col_input = abs_node_local.inputs.get("Color")
        if abs_col_input is not None:
            # Use a darker version of the base color
            r, g, b, a = col
            abs_col_input.default_value = (r * 0.3, g * 0.3, b * 0.3, a)

    fluid_mats.append(mat)

# Add volume absorption for depth
abs_node = w_nodes.new(type="ShaderNodeVolumeAbsorption")
abs_node.location = (w_bsdf.location.x + 200, w_bsdf.location.y - 200)

abs_color = abs_node.inputs.get("Color")
if abs_color is not None:
    abs_color.default_value = (0.05, 0.15, 0.35, 1.0)

abs_density = abs_node.inputs.get("Density")
if abs_density is not None:
    abs_density.default_value = 0.4

w_links.new(abs_node.outputs["Volume"], w_output.inputs["Volume"])

# Rigid materials
RIGID_COLORS = [
    (0.9, 0.5, 0.2, 1.0),
    (0.8, 0.8, 0.8, 1.0),
    (0.6, 0.9, 0.4, 1.0),
    (0.9, 0.4, 0.6, 1.0),
]

rigid_mats = []
for i, col in enumerate(RIGID_COLORS):
    mat = bpy.data.materials.new(name=f"Rigid_{i}")
    mat.use_nodes = True
    r_nodes = mat.node_tree.nodes
    r_links = mat.node_tree.links

    r_bsdf = r_nodes.get("Principled BSDF")
    r_output = None
    for n in r_nodes:
        if n.type == "OUTPUT_MATERIAL":
            r_output = n
            break

    if r_bsdf is None:
        r_bsdf = r_nodes.new(type="ShaderNodeBsdfPrincipled")
    if r_output is None:
        r_output = r_nodes.new(type="ShaderNodeOutputMaterial")

    base = r_bsdf.inputs.get("Base Color")
    if base is not None:
        base.default_value = col

    r_rough = r_bsdf.inputs.get("Roughness")
    if r_rough is not None:
        r_rough.default_value = 0.25

    metal = r_bsdf.inputs.get("Metallic")
    if metal is not None:
        metal.default_value = 0.4

    r_links.new(r_bsdf.outputs["BSDF"], r_output.inputs["Surface"])
    rigid_mats.append(mat)

# ==============================================================
# Collect OBJ files (multiple objs per frame)
# ==============================================================
frame_to_files = {}  # frame_id -> list of (tag, filename)

for fname in os.listdir(OBJ_DIR):
    m = re.match(OBJ_PATTERN, fname)
    if not m:
        continue
    tag = m.group(1)  # e.g. "fluid_0" or "rigid_1"
    frame_id = int(m.group(2))
    frame_to_files.setdefault(frame_id, []).append((tag, fname))

sorted_frames = sorted(frame_to_files.keys())
if not sorted_frames:
    raise RuntimeError(
        f"No OBJ files found in {OBJ_DIR} matching pattern {OBJ_PATTERN}"
    )

n_frames = len(sorted_frames)
scene.frame_start = 1
scene.frame_end = n_frames

print(f"Found {n_frames} frames, with multiple OBJ files per frame.")

# ==============================================================
# Import OBJs for each frame and keyframe visibility
# ==============================================================
for idx, frame_id in enumerate(sorted_frames):
    files = frame_to_files[frame_id]
    print(
        f"\n=== Frame {idx + 1} (id={frame_id}), importing {len(files)} OBJ files ==="
    )

    frame_imported_objs = []

    total_verts = 0
    total_faces = 0
    xmin = ymin = zmin = float("inf")
    xmax = ymax = zmax = float("-inf")

    for tag, fname in files:
        filepath = os.path.join(OBJ_DIR, fname)
        print(f"  Importing {filepath} (tag={tag})")

        before_objs = set(scene.objects)
        res = bpy.ops.wm.obj_import(filepath=filepath)
        if "FINISHED" not in res:
            raise RuntimeError(f"Failed to import OBJ file: {filepath}")

        after_objs = set(scene.objects)
        imported_objs = list(after_objs - before_objs)
        if not imported_objs:
            print(f"  Warning: no new objects detected after importing {filepath}")
            continue

        bpy.context.view_layer.update()

        for obj in imported_objs:
            frame_imported_objs.append(obj)

            mesh = obj.data
            vcount = len(mesh.vertices)
            fcount = len(mesh.polygons)
            total_verts += vcount
            total_faces += fcount

            print(f"    Object '{obj.name}': Verts={vcount}, Faces={fcount}")

            if vcount > 0:
                bbox = [obj.matrix_world @ v.co for v in mesh.vertices]
                xs = [v.x for v in bbox]
                ys = [v.y for v in bbox]
                zs = [v.z for v in bbox]
                xmin = min(xmin, min(xs))
                xmax = max(xmax, max(xs))
                ymin = min(ymin, min(ys))
                ymax = max(ymax, max(ys))
                zmin = min(zmin, min(zs))
                zmax = max(zmax, max(zs))

            tag_lower = tag.lower()
            if tag_lower.startswith("fluid"):
                # Choose fluid material based on fluid index in tag, e.g. "fluid_0", "fluid_1", ...
                mat = fluid_mats[0]
                try:
                    parts = tag_lower.split("_")
                    if len(parts) > 1:
                        idx_int = int(parts[1])
                        mat = fluid_mats[idx_int % len(fluid_mats)]
                except Exception:
                    pass
            elif tag_lower.startswith("rigid"):
                mat = rigid_mats[0]
                try:
                    parts = tag_lower.split("_")
                    if len(parts) > 1:
                        idx_int = int(parts[1])
                        mat = rigid_mats[idx_int % len(rigid_mats)]
                except Exception:
                    pass
            else:
                mat = rigid_mats[0]

            if len(mesh.materials) == 0:
                mesh.materials.append(mat)
            else:
                mesh.materials[0] = mat

    print(f"Total imported verts={total_verts}, faces={total_faces}")
    if total_verts > 0:
        print(
            "Union bounds: "
            f"x[{xmin:.4f}, {xmax:.4f}] "
            f"y[{ymin:.4f}, {ymax:.4f}] "
            f"z[{zmin:.4f}, {zmax:.4f}]"
        )

    # Aim camera at first frame's union center
    if idx == 0 and total_verts > 0:
        center = Vector(((xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5))
        print(f"First frame union center: {center}")

        cam_offset = Vector((2.0, -3.0, 1.8))
        cam.location = center + cam_offset

        direction = center - cam.location
        rot_quat = direction.to_track_quat("-Z", "Y")
        cam.rotation_euler = rot_quat.to_euler()

        light.location = cam.location + Vector((2.0, 1.0, 2.0))
        print(f"Camera location: {cam.location}")
        print(f"Light location: {light.location}")

    # Smooth shading for imported meshes in this frame
    for obj in frame_imported_objs:
        mesh = getattr(obj, "data", None)
        if mesh is not None and hasattr(mesh, "polygons"):
            for poly in mesh.polygons:
                poly.use_smooth = True
            mesh.use_auto_smooth = True

    # Visibility keyframes
    frame = idx + 1
    for obj in frame_imported_objs:
        obj.hide_render = True
        obj.hide_viewport = True
        obj.keyframe_insert(data_path="hide_render", frame=1)
        obj.keyframe_insert(data_path="hide_viewport", frame=1)

        obj.hide_render = False
        obj.hide_viewport = False
        obj.keyframe_insert(data_path="hide_render", frame=frame)
        obj.keyframe_insert(data_path="hide_viewport", frame=frame)

        obj.hide_render = True
        obj.hide_viewport = True
        obj.keyframe_insert(data_path="hide_render", frame=frame + 1)
        obj.keyframe_insert(data_path="hide_viewport", frame=frame + 1)

print("Finished importing and keyframing all OBJ frames.")

# ==============================================================
# Render animation
# ==============================================================
print("\nStart rendering animation...")
bpy.ops.render.render(animation=True)
print("Rendering finished. Frames saved to:", OUT_DIR)
os.system(
    f"ffmpeg -framerate 30 -i blender_frames/{args.project}/frame_%04d.png "
    f"-c:v libx264 -pix_fmt yuv420p blender_renderings/{args.project}.mp4 -y"
)
os.system(
    "ffmpeg -framerate 8 "
    f"-i blender_frames/{args.project}/frame_%04d.png "
    '-vf "scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=full[p];[s1][p]paletteuse=dither=floyd_steinberg" '
    f"blender_renderings/{args.project}.gif -y "
)
