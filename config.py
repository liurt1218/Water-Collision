# config.py

# Simulation dimension
dim = 3

# Simulation domain
domain_min = (0.0, 0.0, 0.0)
domain_max = (1.0, 1.0, 1.0)

# Gravity
gravity = -9.81

# Output directory
output_dir_default = "mpm_demo"

# Simulation timestep
dt = 1e-4
sim_steps_default = 400
substeps_default = 200

# Window
window_res = (1280, 720)

# Rendering
particle_radius = 0.012
particle_color = (0.2, 0.6, 1.0)

# Material kind IDs
WATER = 0
JELLY = 1
SNOW = 2
N_MATERIALS = 3

# Particle and grid
n_particles = 500000
n_grid = 64
dx = 1.0 / n_grid
inv_dx = 1.0 / dx

# Volume related
p_vol = (dx * 0.5) ** dim


# Initialize particle and grid
def init_config(num_particles, num_grid):
    global n_particles, n_grid, dx, inv_dx
    global p_vol

    n_particles = num_particles
    n_grid = num_grid
    dx = 1.0 / n_grid
    inv_dx = 1.0 / dx
    p_vol = (dx * 0.5) ** dim
