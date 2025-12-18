# state.py
import taichi as ti
import config as C
import torch

# Number of materials
N_MATERIALS = C.N_MATERIALS
GRID_RES = 64
MAX_CELL_VERTS = 32

# Particle-level Fields

x = None  # positions
v = None  # velocities
C_apic = None  # APIC affine C
F = None  # deformation gradient
Jp = None  # plastic J (snow/liquid)
materials = None  # material_id
is_used = None  # 0=unused, 1=used
color = None
# p_cdf_states = ti.field(dtype=ti.u64, shape=C.n_particles)
# boundary_dist = ti.field(dtype=float, shape=C.n_particles)
# boundary_normal = ti.Vector.field(3, dtype=float, shape=C.n_particles)
# near_boundary = ti.field(dtype=ti.i32, shape=C.n_particles)
# boundary_rigid_id = ti.field(dtype=ti.i32, shape=C.n_particles)
# boundary_side = ti.field(dtype=ti.i8, shape=C.n_particles)
# fluid_surface_y_mat = ti.field(dtype=ti.f32, shape=N_MATERIALS)


def init_particle_level_fields():
    global x, v, C_apic, F, Jp, materials, is_used, color
    if C.n_particles > 0:
        x = ti.Vector.field(C.dim, float, C.n_particles)
        v = ti.Vector.field(C.dim, float, C.n_particles)
        C_apic = ti.Matrix.field(C.dim, C.dim, float, C.n_particles)
        F = ti.Matrix.field(3, 3, dtype=float, shape=C.n_particles)
        Jp = ti.field(float, C.n_particles)
        materials = ti.field(int, C.n_particles)
        is_used = ti.field(int, C.n_particles)
        color = ti.Vector.field(3, float, C.n_particles)


# Grid-level Fields

grid_v = None
grid_m = None
grid_side = None
grid_dist = None
grid_rigid = None
grid_normal = None
grid_pressure = None
# grid_pressure_w = ti.field(dtype=ti.f32, shape=(C.n_grid, C.n_grid, C.n_grid))


def init_grid_level_fields():
    global grid_v, grid_m, grid_side, grid_dist, grid_rigid, grid_normal, grid_pressure
    grid_v = ti.Vector.field(C.dim, float, (C.n_grid,) * C.dim)
    grid_m = ti.field(float, (C.n_grid,) * C.dim)
    grid_side = ti.field(dtype=ti.u64, shape=(C.n_grid, C.n_grid, C.n_grid))
    grid_dist = ti.field(dtype=ti.f32, shape=(C.n_grid, C.n_grid, C.n_grid))
    grid_rigid = ti.field(dtype=ti.i32, shape=(C.n_grid, C.n_grid, C.n_grid))
    grid_normal = ti.Vector.field(3, dtype=ti.f32, shape=(C.n_grid, C.n_grid, C.n_grid))
    grid_pressure = ti.Vector.field(
        3, dtype=ti.f32, shape=(C.n_grid, C.n_grid, C.n_grid)
    )


# Rigid States

N_RIGID = 0
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


# PyTorch tensors

mesh_local_vertices_t = None
mesh_local_normals_t = None
mesh_vertices_t = None
mesh_normals_t = None

rb_pos_t = None
rb_rot_t = None

rb_mesh_vert_offset_t = None
rb_mesh_vert_count_t = None

vertex_owner_t = None

rb_lin_vel_t = None
rb_ang_vel_t = None

rb_inv_mass_t = None
rb_restitution_t = None
rb_friction_t = None
rb_active_t = None

rb_inv_inertia_world_t = None

domain_min_t = torch.tensor(
    C.domain_min, device="cuda", dtype=torch.float32
).contiguous()
domain_max_t = torch.tensor(
    C.domain_max, device="cuda", dtype=torch.float32
).contiguous()

rb_half_extents_t = None
grid_count_t = None
grid_indices_t = None
dpos_t = None
dlv_t = None
dav_t = None


def init_rigid_torch_storage(device="cuda"):
    """Create torch tensors by copying from Taichi rigid fields (one-time init)."""
    global mesh_local_vertices_t, mesh_local_normals_t, mesh_vertices_t, mesh_normals_t
    global rb_pos_t, rb_rot_t
    global rb_mesh_vert_offset_t, rb_mesh_vert_count_t
    global vertex_owner_t

    if n_mesh_vertices == 0 or n_rigid_bodies == 0:
        return

    # Taichi to_torch gives shape (V, 3) for Vector.field
    mesh_local_vertices_t = mesh_local_vertices.to_torch(device=device).contiguous()
    mesh_local_normals_t = mesh_local_normals.to_torch(device=device).contiguous()

    # outputs (CUDA will write)
    mesh_vertices_t = torch.empty(
        (n_mesh_vertices, 3), device=device, dtype=torch.float32
    )
    mesh_normals_t = torch.empty(
        (n_mesh_vertices, 3), device=device, dtype=torch.float32
    )

    # --- Rigid pose (copy from Taichi) ---
    B = n_rigid_bodies

    # --- Rigid state buffers (ALLOC ONCE) ---
    global rb_lin_vel_t, rb_ang_vel_t
    global rb_inv_mass_t, rb_restitution_t, rb_friction_t, rb_active_t
    global rb_inv_inertia_world_t
    global rb_half_extents_t, grid_count_t, grid_indices_t, dpos_t, dlv_t, dav_t

    rb_pos_t = torch.empty((B, 3), device=device, dtype=torch.float32)
    rb_rot_t = torch.empty((B, 3, 3), device=device, dtype=torch.float32)

    rb_lin_vel_t = torch.empty((B, 3), device=device, dtype=torch.float32)
    rb_ang_vel_t = torch.empty((B, 3), device=device, dtype=torch.float32)

    rb_inv_mass_t = torch.empty((B,), device=device, dtype=torch.float32)
    rb_restitution_t = torch.empty((B,), device=device, dtype=torch.float32)
    rb_friction_t = torch.empty((B,), device=device, dtype=torch.float32)
    rb_active_t = torch.empty((B,), device=device, dtype=torch.int32)

    rb_inv_inertia_world_t = torch.empty((B, 3, 3), device=device, dtype=torch.float32)

    # initial copy (IN-PLACE)
    rb_pos_t.copy_(rb_pos.to_torch(device=device))
    rb_rot_t.copy_(rb_rot.to_torch(device=device))

    rb_lin_vel_t.copy_(rb_lin_vel.to_torch(device=device))
    rb_ang_vel_t.copy_(rb_ang_vel.to_torch(device=device))

    rb_inv_mass_t.copy_(rb_inv_mass.to_torch(device=device))
    rb_restitution_t.copy_(rb_restitution.to_torch(device=device))
    rb_friction_t.copy_(rb_friction.to_torch(device=device))
    rb_active_t.copy_(rb_active.to_torch(device=device).to(torch.int32))

    rb_inv_inertia_world_t.copy_(rb_inv_inertia_world.to_torch(device=device))

    rb_half_extents_t = torch.empty((B, 3), device=device, dtype=torch.float32)
    rb_half_extents_t.copy_(rb_half_extents.to_torch(device=device))

    # --- grid buffers ---
    CELLS = GRID_RES * GRID_RES * GRID_RES
    grid_count_t = torch.zeros((B, CELLS), device=device, dtype=torch.int32)
    grid_indices_t = torch.empty(
        (B, CELLS, MAX_CELL_VERTS), device=device, dtype=torch.int32
    )

    # --- delta buffers (per-frame accum) ---
    dpos_t = torch.zeros((B, 3), device=device, dtype=torch.float32)
    dlv_t = torch.zeros((B, 3), device=device, dtype=torch.float32)
    dav_t = torch.zeros((B, 3), device=device, dtype=torch.float32)

    # offsets/counts are tiny; keep on CUDA for convenience
    rb_mesh_vert_offset_t = (
        rb_mesh_vert_offset.to_torch(device=device).to(torch.int32).contiguous()
    )
    rb_mesh_vert_count_t = (
        rb_mesh_vert_count.to_torch(device=device).to(torch.int32).contiguous()
    )

    # --- owner (one-time build) ---
    owner_cpu = torch.empty((n_mesh_vertices,), device="cpu", dtype=torch.int32)
    off = rb_mesh_vert_offset.to_torch(device="cpu").to(torch.int32)
    cnt = rb_mesh_vert_count.to_torch(device="cpu").to(torch.int32)
    for b in range(n_rigid_bodies):
        s = int(off[b].item())
        c = int(cnt[b].item())
        owner_cpu[s : s + c] = b
    vertex_owner_t = owner_cpu.to(device=device, non_blocking=True).contiguous()


def sync_rigid_pose_torch_from_taichi(device="cuda"):
    """Refresh torch copies IN-PLACE (no re-allocation)."""
    rb_pos_t.copy_(rb_pos.to_torch(device=device))
    rb_rot_t.copy_(rb_rot.to_torch(device=device))

    rb_lin_vel_t.copy_(rb_lin_vel.to_torch(device=device))
    rb_ang_vel_t.copy_(rb_ang_vel.to_torch(device=device))

    rb_inv_mass_t.copy_(rb_inv_mass.to_torch(device=device))
    rb_restitution_t.copy_(rb_restitution.to_torch(device=device))
    rb_friction_t.copy_(rb_friction.to_torch(device=device))
    rb_active_t.copy_(rb_active.to_torch(device=device).to(torch.int32))

    rb_inv_inertia_world_t.copy_(rb_inv_inertia_world.to_torch(device=device))


def sync_rigid_state_torch_to_taichi():
    rb_pos.from_torch(rb_pos_t)
    rb_rot.from_torch(rb_rot_t)

    rb_lin_vel.from_torch(rb_lin_vel_t)
    rb_ang_vel.from_torch(rb_ang_vel_t)

    rb_inv_inertia_world.from_torch(rb_inv_inertia_world_t)


def sync_mesh_torch_to_taichi():
    """After CUDA updates mesh_vertices_t/mesh_normals_t, copy back to Taichi fields for existing Taichi kernels."""
    mesh_vertices.from_torch(mesh_vertices_t)
    mesh_normals.from_torch(mesh_normals_t)
