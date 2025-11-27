# sim/config.py
from dataclasses import dataclass, field
from typing import Sequence, Tuple

Vec3 = Tuple[float, float, float]

# ----------------------------------------------------------------------
# Global simulation scalars (shared by the whole scene)
# ----------------------------------------------------------------------
dim = 3
dt = 1e-3

domain_min: Vec3 = (0.0, 0.0, 0.0)
domain_max: Vec3 = (1.0, 1.0, 1.0)
domain_min_cropped: Vec3 = (0.02, 0.02, 0.02)
domain_max_cropped: Vec3 = (0.98, 0.98, 0.98)
g: Vec3 = (0.0, -9.81, 0.0)

# Default values used as convenient fallbacks for fluid blocks
_default_x_len, _default_y_len, _default_z_len = 0.96, 0.96, 0.24
_default_particle_diameter = 0.02

_default_rho0 = 1000.0
_default_surface_tension = 0.04
_default_viscosity = 5.0

# DFSPH solver parameters (global for now; can be made block-dependent later)
max_iter_density = 1000
max_iter_div = 1000
max_error = 1e-4
max_error_V = 1e-3
eps = 1e-5

# Default support radius scale: h = support_radius_scale * particle_diameter
_default_support_radius_scale = 2.0

rigid_particle_diameter = 0.02


# ----------------------------------------------------------------------
# Fluid configs
# ----------------------------------------------------------------------
@dataclass
class FluidBlockConfig:
    """
    Configuration for a single fluid block.

    Each block can have its own geometry and physical parameters.
    Particles generated from different blocks do not share any fluid
    property; parameters are baked per particle during initialization.
    """

    # Whether this block is active
    enabled: bool = True

    # Block placement in world space
    base: Vec3 = (0.02, 0.02, 0.02)  # minimum corner
    size: Vec3 = (_default_x_len, _default_y_len, _default_z_len)

    # Particle spacing
    particle_diameter: float = _default_particle_diameter

    # Physical properties
    rho0: float = _default_rho0  # rest density
    surface_tension: float = _default_surface_tension
    viscosity: float = _default_viscosity

    # Kernel support radius = support_radius_scale * particle_diameter
    support_radius_scale: float = _default_support_radius_scale


@dataclass
class FluidSceneConfig:
    """Collection of multiple fluid blocks in the scene."""

    blocks: Sequence[FluidBlockConfig] = field(default_factory=list)


# ----------------------------------------------------------------------
# Rigid configs (unchanged, just English comments)
# ----------------------------------------------------------------------
@dataclass
class RigidBodyConfig:
    """Configuration for a single rigid body."""

    center: Vec3
    mesh_path: str

    # Geometry scale (half extents of the approximated bounding box)
    half_extents: Vec3 = (0.12, 0.12, 0.12)

    # Physical properties
    density: float = 300.0
    restitution: float = 0.3
    friction: float = 0.3

    # Contact and damping parameters
    contact_threshold_scale: float = 0.8
    ground_penetration_clamp: float = 1.0
    angular_damping_ground: float = 0.2


@dataclass
class RigidSceneConfig:
    """
    Rigid-body scene configuration.

    Each rigid body can have its own OBJ mesh and physical parameters.
    """

    bodies: Sequence[RigidBodyConfig] = field(default_factory=list)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def compute_total_fluid_particles(fluid_scene: FluidSceneConfig) -> int:
    """
    Compute total number of fluid particles for a given FluidSceneConfig.

    Each block is discretized into a regular grid of particles of spacing
    particle_diameter, filling the block's size in x/y/z.
    """
    total = 0
    for cfg in fluid_scene.blocks:
        if not cfg.enabled:
            continue
        sx, sy, sz = cfg.size
        h = cfg.particle_diameter
        nx = int(sx / h)
        ny = int(sy / h)
        nz = int(sz / h)
        total += nx * ny * nz
    return total
