# sim/step.py
from .fluid import (
    compute_non_pressure_acceleration,
    update_fluid_velocity,
    update_fluid_position,
    enforce_boundary,
    compute_density,
    compute_alpha,
    correct_density_error,
    correct_divergence_error,
)
from .rigid import (
    rigid_step,
    handle_rigid_collisions,
    renew_rigid_particles,
    handle_rigid_ground_collision,
)


def step():
    compute_non_pressure_acceleration()
    update_fluid_velocity()
    correct_density_error()

    update_fluid_position()

    rigid_step()
    handle_rigid_collisions()
    renew_rigid_particles()
    handle_rigid_ground_collision()
    renew_rigid_particles()

    enforce_boundary()
    compute_density()
    compute_alpha()
    correct_divergence_error()
