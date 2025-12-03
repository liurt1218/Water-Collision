# state.py
import taichi as ti
import config as C

# Number of materials
N_MATERIALS = C.N_MATERIALS

# Particle-level fields
x = ti.Vector.field(C.dim, float, C.n_particles)  # positions
v = ti.Vector.field(C.dim, float, C.n_particles)  # velocities
C_apic = ti.Matrix.field(C.dim, C.dim, float, C.n_particles)  # APIC affine C
F = ti.Matrix.field(3, 3, dtype=float, shape=C.n_particles)  # deformation gradient
Jp = ti.field(float, C.n_particles)  # plastic J (snow/liquid)

# Particle-level boundary info (for rigid coupling, deprecated)
p_cdf_states = ti.field(dtype=ti.u64, shape=C.n_particles)
boundary_dist = ti.field(dtype=float, shape=C.n_particles)
boundary_normal = ti.Vector.field(3, dtype=float, shape=C.n_particles)
near_boundary = ti.field(dtype=ti.i32, shape=C.n_particles)
boundary_rigid_id = ti.field(dtype=ti.i32, shape=C.n_particles)
boundary_side = ti.field(dtype=ti.i8, shape=C.n_particles)

materials = ti.field(int, C.n_particles)  # material_id
is_used = ti.field(int, C.n_particles)  # 0=unused, 1=used
color = ti.Vector.field(3, float, C.n_particles)

fluid_surface_y_mat = ti.field(dtype=ti.f32, shape=N_MATERIALS)

# Grid fields
grid_v = ti.Vector.field(C.dim, float, (C.n_grid,) * C.dim)
grid_m = ti.field(float, (C.n_grid,) * C.dim)

# Grid distances
grid_side = ti.field(dtype=ti.u64, shape=(C.n_grid, C.n_grid, C.n_grid))
grid_dist = ti.field(dtype=ti.f32, shape=(C.n_grid, C.n_grid, C.n_grid))
grid_rigid = ti.field(dtype=ti.i32, shape=(C.n_grid, C.n_grid, C.n_grid))
grid_normal = ti.Vector.field(3, dtype=ti.f32, shape=(C.n_grid, C.n_grid, C.n_grid))

# deprecated
grid_pressure = ti.Vector.field(3, dtype=ti.f32, shape=(C.n_grid, C.n_grid, C.n_grid))
grid_pressure_w = ti.field(dtype=ti.f32, shape=(C.n_grid, C.n_grid, C.n_grid))

# Rigid body state (runtime-sized, depends on user input)
N_RIGID: int = 0

# Positions and orientations
rb_pos = None  # center position, Vector(3)
rb_rot = None  # rotation matrix (world from local), Matrix(3x3)

# Linear and angular velocities (world space)
rb_lin_vel = None  # Vector(3)
rb_ang_vel = None  # Vector(3)

# Box geometry: half extents in x/y/z
rb_half_extents = None  # Vector(3)

# Mass and inertia
rb_mass = None  # scalar
rb_inv_mass = None  # scalar

# Inertia tensors: body space and world space inverse inertia
rb_inertia_body = None  # Matrix(3x3)
rb_inv_inertia_body = None  # Matrix(3x3)
rb_inv_inertia_world = None  # Matrix(3x3)

# Accumulated forces and torques (world space)
rb_force = None  # Vector(3)
rb_torque = None  # Vector(3)

# Per-body material-like properties
rb_restitution = None  # scalar, bounciness
rb_friction = None  # scalar, friction coefficient

# Optional: mark which slots are used (1 = active, 0 = inactive)
rb_active = None  # int32


def init_rigid_fields(n_rigid: int):
    # Allocate Taichi fields for rigid bodies based on user input.
    global N_RIGID
    global rb_pos, rb_rot
    global rb_lin_vel, rb_ang_vel
    global rb_half_extents
    global rb_mass, rb_inv_mass
    global rb_inertia_body, rb_inv_inertia_body, rb_inv_inertia_world
    global rb_force, rb_torque
    global rb_restitution, rb_friction, rb_active

    N_RIGID = n_rigid

    if N_RIGID == 0:
        # No rigid bodies: leave all fields as None.
        return

    # Positions and orientations
    rb_pos = ti.Vector.field(3, dtype=float, shape=N_RIGID)
    rb_rot = ti.Matrix.field(3, 3, dtype=float, shape=N_RIGID)

    # Linear and angular velocities
    rb_lin_vel = ti.Vector.field(3, dtype=float, shape=N_RIGID)
    rb_ang_vel = ti.Vector.field(3, dtype=float, shape=N_RIGID)

    # Geometry
    rb_half_extents = ti.Vector.field(3, dtype=float, shape=N_RIGID)

    # Mass and inertia
    rb_mass = ti.field(dtype=float, shape=N_RIGID)
    rb_inv_mass = ti.field(dtype=float, shape=N_RIGID)

    rb_inertia_body = ti.Matrix.field(3, 3, dtype=float, shape=N_RIGID)
    rb_inv_inertia_body = ti.Matrix.field(3, 3, dtype=float, shape=N_RIGID)
    rb_inv_inertia_world = ti.Matrix.field(3, 3, dtype=float, shape=N_RIGID)

    # Forces and torques
    rb_force = ti.Vector.field(3, dtype=float, shape=N_RIGID)
    rb_torque = ti.Vector.field(3, dtype=float, shape=N_RIGID)

    # Material-like properties
    rb_restitution = ti.field(dtype=float, shape=N_RIGID)
    rb_friction = ti.field(dtype=float, shape=N_RIGID)

    # Active flags
    rb_active = ti.field(dtype=ti.i32, shape=N_RIGID)


# Rigid-body mesh visualization (many rigid bodies)
n_rigid_bodies: int = 0  # number of rigid bodies (for mesh)
n_mesh_vertices: int = 0  # total number of triangle vertices

mesh_local_vertices = None  # Vector(3), shape = n_mesh_vertices
mesh_local_normals = None  # Vector(3), shape = n_mesh_vertices

mesh_vertices = None  # Vector(3), shape = n_mesh_vertices
mesh_normals = None  # Vector(3), shape = n_mesh_vertices

# Per-rigid vertex range in the shared buffer
rb_mesh_vert_offset = None  # int field, shape = n_rigid_bodies
rb_mesh_vert_count = None  # int field, shape = n_rigid_bodies


def init_rigid_mesh_fields(total_verts: int, n_rigid: int):
    # Allocate Taichi fields for rigid-body mesh visualization.
    global n_mesh_vertices, n_rigid_bodies
    global mesh_local_vertices, mesh_local_normals
    global mesh_vertices, mesh_normals
    global rb_mesh_vert_offset, rb_mesh_vert_count

    n_mesh_vertices = total_verts
    n_rigid_bodies = n_rigid

    if n_mesh_vertices == 0 or n_rigid_bodies == 0:
        return

    # Shared vertex buffers (triangle soup)
    mesh_local_vertices = ti.Vector.field(3, dtype=float, shape=n_mesh_vertices)
    mesh_local_normals = ti.Vector.field(3, dtype=float, shape=n_mesh_vertices)

    mesh_vertices = ti.Vector.field(3, dtype=float, shape=n_mesh_vertices)
    mesh_normals = ti.Vector.field(3, dtype=float, shape=n_mesh_vertices)

    # Per-rigid vertex ranges
    rb_mesh_vert_offset = ti.field(dtype=int, shape=n_rigid_bodies)
    rb_mesh_vert_count = ti.field(dtype=int, shape=n_rigid_bodies)
