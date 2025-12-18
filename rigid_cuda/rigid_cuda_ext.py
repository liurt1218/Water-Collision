import os
from torch.utils.cpp_extension import load

_this_dir = os.path.dirname(os.path.abspath(__file__))

rigid_cuda = load(
    name="rigid_cuda_ext_mod",
    sources=[
        os.path.join(_this_dir, "rigid_cuda.cpp"),
        os.path.join(_this_dir, "rigid_cuda_kernel.cu"),
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


# Update mesh vertices
def update_all_mesh_vertices(
    mesh_vertices,
    mesh_normals,
    mesh_local_vertices,
    mesh_local_normals,
    rb_pos,
    rb_rot,
    vertex_owner,
):
    rigid_cuda.update_all_mesh_vertices(
        mesh_vertices,
        mesh_normals,
        mesh_local_vertices,
        mesh_local_normals,
        rb_pos,
        rb_rot,
        vertex_owner,
    )


# Collision with domain walls
def handle_domain_walls(
    mesh_vertices,
    rb_pos,
    rb_lin_vel,
    rb_ang_vel,
    rb_inv_inertia_world,
    rb_inv_mass,
    rb_restitution,
    rb_friction,
    rb_active,
    rb_mesh_vert_offset,
    rb_mesh_vert_count,
    domain_min,
    domain_max,
    max_pen=0.01,
):
    rigid_cuda.handle_domain_walls(
        mesh_vertices,
        rb_pos,
        rb_lin_vel,
        rb_ang_vel,
        rb_inv_inertia_world,
        rb_inv_mass,
        rb_restitution,
        rb_friction,
        rb_active,
        rb_mesh_vert_offset,
        rb_mesh_vert_count,
        domain_min,
        domain_max,
        float(max_pen),
    )


# Build collision grid
def build_vertex_grid(
    mesh_vertices,
    vertex_owner,
    rb_active,
    grid_count,
    grid_indices,
    domain_min,
    domain_max,
):
    return rigid_cuda.build_vertex_grid(
        mesh_vertices,
        vertex_owner,
        rb_active,
        grid_count,
        grid_indices,
        domain_min,
        domain_max,
    )


# Collision
def rigid_rigid_collisions(
    mesh_vertices,
    rb_pos,
    rb_lin_vel,
    rb_ang_vel,
    rb_inv_inertia_world,
    rb_inv_mass,
    rb_restitution,
    rb_friction,
    rb_half_extents,
    rb_active,
    rb_mesh_vert_offset,
    rb_mesh_vert_count,
    grid_count,
    grid_indices,
    domain_min,
    domain_max,
    dpos,
    dlv,
    dav,
    thresh: float = 0.01,
):
    return rigid_cuda.rigid_rigid_collisions(
        mesh_vertices,
        rb_pos,
        rb_lin_vel,
        rb_ang_vel,
        rb_inv_inertia_world,
        rb_inv_mass,
        rb_restitution,
        rb_friction,
        rb_half_extents,
        rb_active,
        rb_mesh_vert_offset,
        rb_mesh_vert_count,
        grid_count,
        grid_indices,
        domain_min,
        domain_max,
        dpos,
        dlv,
        dav,
        float(thresh),
    )


# Add the forces and torques
def apply_rigid_deltas(rb_pos, rb_lin_vel, rb_ang_vel, dpos, dlv, dav):
    return rigid_cuda.apply_rigid_deltas(rb_pos, rb_lin_vel, rb_ang_vel, dpos, dlv, dav)
