# sim/state.py
import taichi as ti
from . import config as C

# Global counts
n_fluid: int = 0
n_rigid_bodies: int = 0
n_rigid_total: int = 0
n_particles: int = 0

# Particle-level fields
x = None  # positions
v = None  # velocities
a = None  # accelerations

is_fluid = None  # 1 if this particle is fluid, 0 otherwise
is_dynamic = None  # 1 if this particle is dynamic, 0 otherwise

rest_volume = None
density = None

alpha = None
density_star = None
density_deriv = None
kappa = None
kappa_v = None

# Per-particle fluid properties (new)
fluid_rho0 = None  # rest density for this particle
fluid_surface_tension = None  # surface tension coefficient
fluid_viscosity = None  # viscosity coefficient
fluid_support_radius = None  # SPH support radius h
fluid_particle_diameter = None  # particle spacing for this particle

# Rigid body kinematics
rb_pos = None
rb_vel = None
rb_force = None
rb_omega = None
rb_torque = None
rb_rot = None

# Inertia tensors
I_body = None
I_body_inv = None

# Rigid particle layout (ragged)
rb_offset = None  # offset of first particle of body b in rigid block
rb_count = None  # number of rigid particles for body b
rigid_id = None  # for each particle i, which body it belongs to, or -1 for fluid

# Local coordinates of rigid particles (flattened across all bodies)
rb_local = None  # shape = (n_rigid_total, )

# Per-body rigid parameters (mass, friction, etc.)
rb_half = None
rb_mass = None
rb_restitution = None
rb_friction = None
rb_contact_threshold_scale = None
rb_ground_penetration_clamp = None
rb_angular_damping = None

# Mesh fields (flattened for all rigid bodies)
mesh_vert_total = 0
mesh_index_total = 0

mesh_local = None  # local mesh coordinates, per vertex
mesh_vertices = None  # world-space mesh vertices
mesh_indices = None  # triangle indices

mesh_vert_offset = None  # per-body vertex range
mesh_vert_count = None
mesh_index_offset = None  # per-body index range
mesh_index_count = None


def allocate_fields(
    n_fluid_in: int,
    mesh_vert_counts: list[int],
    mesh_index_counts: list[int],
):
    """
    Allocate all Taichi fields based on:
      - fluid particle count
      - per-body mesh vertex counts
      - per-body mesh index counts

    We assume one rigid particle per mesh vertex.
    """
    global n_fluid, n_rigid_bodies, n_rigid_total, n_particles
    global x, v, a, is_fluid, is_dynamic, rest_volume, density
    global alpha, density_star, density_deriv, kappa, kappa_v
    global fluid_rho0, fluid_surface_tension, fluid_viscosity
    global fluid_support_radius, fluid_particle_diameter
    global rb_pos, rb_vel, rb_force, rb_omega, rb_torque, rb_rot
    global rb_half, rb_mass, rb_restitution, rb_friction
    global rb_contact_threshold_scale, rb_ground_penetration_clamp
    global rb_angular_damping
    global rb_local, rb_offset, rb_count, rigid_id
    global I_body, I_body_inv
    global mesh_vert_total, mesh_index_total
    global mesh_local, mesh_vertices, mesh_indices
    global mesh_vert_offset, mesh_vert_count
    global mesh_index_offset, mesh_index_count

    # Counts
    n_fluid = n_fluid_in
    n_rigid_bodies = len(mesh_vert_counts)
    n_rigid_total = int(sum(mesh_vert_counts))
    n_particles = n_fluid + n_rigid_total
    mesh_vert_total = int(sum(mesh_vert_counts))
    mesh_index_total = int(sum(mesh_index_counts))

    # --------------------------------------------------------------
    # Particle-level fields
    # --------------------------------------------------------------
    x = ti.Vector.field(C.dim, ti.f32, shape=n_particles)
    v = ti.Vector.field(C.dim, ti.f32, shape=n_particles)
    a = ti.Vector.field(C.dim, ti.f32, shape=n_particles)

    is_fluid = ti.field(ti.i32, shape=n_particles)
    is_dynamic = ti.field(ti.i32, shape=n_particles)
    rest_volume = ti.field(ti.f32, shape=n_particles)
    density = ti.field(ti.f32, shape=n_particles)

    alpha = ti.field(ti.f32, shape=n_particles)
    density_star = ti.field(ti.f32, shape=n_particles)
    density_deriv = ti.field(ti.f32, shape=n_particles)
    kappa = ti.field(ti.f32, shape=n_particles)
    kappa_v = ti.field(ti.f32, shape=n_particles)

    rigid_id = ti.field(ti.i32, shape=n_particles)

    # Per-particle fluid properties (filled during fluid initialization)
    fluid_rho0 = ti.field(ti.f32, shape=n_particles)
    fluid_surface_tension = ti.field(ti.f32, shape=n_particles)
    fluid_viscosity = ti.field(ti.f32, shape=n_particles)
    fluid_support_radius = ti.field(ti.f32, shape=n_particles)
    fluid_particle_diameter = ti.field(ti.f32, shape=n_particles)

    # --------------------------------------------------------------
    # Rigid body fields
    # --------------------------------------------------------------
    rb_pos = ti.Vector.field(C.dim, ti.f32, shape=n_rigid_bodies)
    rb_vel = ti.Vector.field(C.dim, ti.f32, shape=n_rigid_bodies)
    rb_force = ti.Vector.field(C.dim, ti.f32, shape=n_rigid_bodies)

    rb_omega = ti.Vector.field(C.dim, ti.f32, shape=n_rigid_bodies)
    rb_torque = ti.Vector.field(C.dim, ti.f32, shape=n_rigid_bodies)
    rb_rot = ti.Matrix.field(C.dim, C.dim, ti.f32, shape=n_rigid_bodies)

    rb_half = ti.Vector.field(C.dim, ti.f32, shape=n_rigid_bodies)
    rb_mass = ti.field(ti.f32, shape=n_rigid_bodies)
    rb_restitution = ti.field(ti.f32, shape=n_rigid_bodies)
    rb_friction = ti.field(ti.f32, shape=n_rigid_bodies)
    rb_contact_threshold_scale = ti.field(ti.f32, shape=n_rigid_bodies)
    rb_ground_penetration_clamp = ti.field(ti.f32, shape=n_rigid_bodies)
    rb_angular_damping = ti.field(ti.f32, shape=n_rigid_bodies)

    rb_local = ti.Vector.field(C.dim, ti.f32, shape=n_rigid_total)
    rb_offset = ti.field(ti.i32, shape=n_rigid_bodies)
    rb_count = ti.field(ti.i32, shape=n_rigid_bodies)

    I_body = ti.Matrix.field(C.dim, C.dim, ti.f32, shape=n_rigid_bodies)
    I_body_inv = ti.Matrix.field(C.dim, C.dim, ti.f32, shape=n_rigid_bodies)

    # --------------------------------------------------------------
    # Mesh fields (flattened across all rigid bodies)
    # --------------------------------------------------------------
    mesh_local = ti.Vector.field(3, ti.f32, shape=mesh_vert_total)
    mesh_vertices = ti.Vector.field(3, ti.f32, shape=mesh_vert_total)
    mesh_indices = ti.field(ti.i32, shape=mesh_index_total)

    mesh_vert_offset = ti.field(ti.i32, shape=n_rigid_bodies)
    mesh_vert_count = ti.field(ti.i32, shape=n_rigid_bodies)
    mesh_index_offset = ti.field(ti.i32, shape=n_rigid_bodies)
    mesh_index_count = ti.field(ti.i32, shape=n_rigid_bodies)

    # Fill vertex and index offsets per body, and rigid-particle layout
    vert_offset = 0
    index_offset = 0
    for b, vcnt in enumerate(mesh_vert_counts):
        mesh_vert_offset[b] = vert_offset
        mesh_vert_count[b] = vcnt
        rb_offset[b] = vert_offset
        rb_count[b] = vcnt  # one particle per vertex
        vert_offset += vcnt

    for b, icnt in enumerate(mesh_index_counts):
        mesh_index_offset[b] = index_offset
        mesh_index_count[b] = icnt
        index_offset += icnt
