#include <stdexcept>
#include <string>
#include <torch/extension.h>

// Update mesh CUDA implementation
void update_all_mesh_vertices_cuda(torch::Tensor mesh_vertices,
                                   torch::Tensor mesh_normals,
                                   torch::Tensor mesh_local_vertices,
                                   torch::Tensor mesh_local_normals,
                                   torch::Tensor rb_pos, torch::Tensor rb_rot,
                                   torch::Tensor vertex_owner);

// Domain wall collision CUDA implementation
void handle_domain_walls_cuda(
    torch::Tensor mesh_vertices, torch::Tensor rb_pos, torch::Tensor rb_lin_vel,
    torch::Tensor rb_ang_vel, torch::Tensor rb_inv_inertia_world,
    torch::Tensor rb_inv_mass, torch::Tensor rb_restitution,
    torch::Tensor rb_friction, torch::Tensor rb_active,
    torch::Tensor rb_mesh_vert_offset, torch::Tensor rb_mesh_vert_count,
    torch::Tensor domain_min, torch::Tensor domain_max, float max_pen);

// Collision grid
void build_vertex_grid_cuda(torch::Tensor mesh_vertices,
                            torch::Tensor vertex_owner, torch::Tensor rb_active,
                            torch::Tensor grid_count,
                            torch::Tensor grid_indices,
                            torch::Tensor domain_min, torch::Tensor domain_max);

// Collision
void rigid_rigid_collisions_cuda(
    torch::Tensor mesh_vertices, torch::Tensor rb_pos, torch::Tensor rb_lin_vel,
    torch::Tensor rb_ang_vel, torch::Tensor rb_inv_inertia_world,
    torch::Tensor rb_inv_mass, torch::Tensor rb_restitution,
    torch::Tensor rb_friction, torch::Tensor rb_half_extents,
    torch::Tensor rb_active, torch::Tensor rb_mesh_vert_offset,
    torch::Tensor rb_mesh_vert_count, torch::Tensor grid_count,
    torch::Tensor grid_indices, torch::Tensor domain_min,
    torch::Tensor domain_max, torch::Tensor dpos, torch::Tensor dlv,
    torch::Tensor dav, float thresh);

// Accumulate force
void apply_rigid_deltas_cuda(torch::Tensor rb_pos, torch::Tensor rb_lin_vel,
                             torch::Tensor rb_ang_vel, torch::Tensor dpos,
                             torch::Tensor dlv, torch::Tensor dav);

// Checks
static void check_cuda_contig(const torch::Tensor &t, const char *name) {
  if (!t.defined())
    throw std::runtime_error(std::string(name) + " is undefined");
  if (!t.is_cuda())
    throw std::runtime_error(std::string(name) + " must be CUDA");
  if (!t.is_contiguous())
    throw std::runtime_error(std::string(name) + " must be contiguous");
}

static void check_dim(const torch::Tensor &t, int dim, const char *name) {
  if (t.dim() != dim)
    throw std::runtime_error(std::string(name) +
                             " must have dim=" + std::to_string(dim));
}

static void check_dtype(const torch::Tensor &t, c10::ScalarType ty,
                        const char *name) {
  if (t.scalar_type() != ty)
    throw std::runtime_error(std::string(name) + " wrong dtype");
}

// ----------------- update mesh wrapper -----------------
void update_all_mesh_vertices(torch::Tensor mesh_vertices,
                              torch::Tensor mesh_normals,
                              torch::Tensor mesh_local_vertices,
                              torch::Tensor mesh_local_normals,
                              torch::Tensor rb_pos, torch::Tensor rb_rot,
                              torch::Tensor vertex_owner) {
  check_cuda_contig(mesh_vertices, "mesh_vertices");
  check_cuda_contig(mesh_normals, "mesh_normals");
  check_cuda_contig(mesh_local_vertices, "mesh_local_vertices");
  check_cuda_contig(mesh_local_normals, "mesh_local_normals");
  check_cuda_contig(rb_pos, "rb_pos");
  check_cuda_contig(rb_rot, "rb_rot");
  check_cuda_contig(vertex_owner, "vertex_owner");

  check_dtype(mesh_vertices, torch::kFloat32, "mesh_vertices");
  check_dtype(mesh_normals, torch::kFloat32, "mesh_normals");
  check_dtype(mesh_local_vertices, torch::kFloat32, "mesh_local_vertices");
  check_dtype(mesh_local_normals, torch::kFloat32, "mesh_local_normals");
  check_dtype(rb_pos, torch::kFloat32, "rb_pos");
  check_dtype(rb_rot, torch::kFloat32, "rb_rot");
  check_dtype(vertex_owner, torch::kInt32, "vertex_owner");

  check_dim(mesh_vertices, 2, "mesh_vertices");
  check_dim(mesh_normals, 2, "mesh_normals");
  check_dim(mesh_local_vertices, 2, "mesh_local_vertices");
  check_dim(mesh_local_normals, 2, "mesh_local_normals");
  check_dim(rb_pos, 2, "rb_pos");
  check_dim(rb_rot, 3, "rb_rot");
  check_dim(vertex_owner, 1, "vertex_owner");

  update_all_mesh_vertices_cuda(mesh_vertices, mesh_normals,
                                mesh_local_vertices, mesh_local_normals, rb_pos,
                                rb_rot, vertex_owner);
}

// ----------------- domain walls wrapper -----------------
void handle_domain_walls(
    torch::Tensor mesh_vertices, torch::Tensor rb_pos, torch::Tensor rb_lin_vel,
    torch::Tensor rb_ang_vel, torch::Tensor rb_inv_inertia_world,
    torch::Tensor rb_inv_mass, torch::Tensor rb_restitution,
    torch::Tensor rb_friction, torch::Tensor rb_active,
    torch::Tensor rb_mesh_vert_offset, torch::Tensor rb_mesh_vert_count,
    torch::Tensor domain_min, torch::Tensor domain_max, float max_pen) {
  check_cuda_contig(mesh_vertices, "mesh_vertices");
  check_cuda_contig(rb_pos, "rb_pos");
  check_cuda_contig(rb_lin_vel, "rb_lin_vel");
  check_cuda_contig(rb_ang_vel, "rb_ang_vel");
  check_cuda_contig(rb_inv_inertia_world, "rb_inv_inertia_world");
  check_cuda_contig(rb_inv_mass, "rb_inv_mass");
  check_cuda_contig(rb_restitution, "rb_restitution");
  check_cuda_contig(rb_friction, "rb_friction");
  check_cuda_contig(rb_active, "rb_active");
  check_cuda_contig(rb_mesh_vert_offset, "rb_mesh_vert_offset");
  check_cuda_contig(rb_mesh_vert_count, "rb_mesh_vert_count");
  check_cuda_contig(domain_min, "domain_min");
  check_cuda_contig(domain_max, "domain_max");

  check_dtype(mesh_vertices, torch::kFloat32, "mesh_vertices");
  check_dtype(rb_pos, torch::kFloat32, "rb_pos");
  check_dtype(rb_lin_vel, torch::kFloat32, "rb_lin_vel");
  check_dtype(rb_ang_vel, torch::kFloat32, "rb_ang_vel");
  check_dtype(rb_inv_inertia_world, torch::kFloat32, "rb_inv_inertia_world");
  check_dtype(rb_inv_mass, torch::kFloat32, "rb_inv_mass");
  check_dtype(rb_restitution, torch::kFloat32, "rb_restitution");
  check_dtype(rb_friction, torch::kFloat32, "rb_friction");
  check_dtype(rb_active, torch::kInt32, "rb_active");
  check_dtype(rb_mesh_vert_offset, torch::kInt32, "rb_mesh_vert_offset");
  check_dtype(rb_mesh_vert_count, torch::kInt32, "rb_mesh_vert_count");
  check_dtype(domain_min, torch::kFloat32, "domain_min");
  check_dtype(domain_max, torch::kFloat32, "domain_max");

  handle_domain_walls_cuda(mesh_vertices, rb_pos, rb_lin_vel, rb_ang_vel,
                           rb_inv_inertia_world, rb_inv_mass, rb_restitution,
                           rb_friction, rb_active, rb_mesh_vert_offset,
                           rb_mesh_vert_count, domain_min, domain_max, max_pen);
}

// ----------------- build grid wrapper -----------------
void build_vertex_grid(torch::Tensor mesh_vertices, torch::Tensor vertex_owner,
                       torch::Tensor rb_active, torch::Tensor grid_count,
                       torch::Tensor grid_indices, torch::Tensor domain_min,
                       torch::Tensor domain_max) {
  check_cuda_contig(mesh_vertices, "mesh_vertices");
  check_cuda_contig(vertex_owner, "vertex_owner");
  check_cuda_contig(rb_active, "rb_active");
  check_cuda_contig(grid_count, "grid_count");
  check_cuda_contig(grid_indices, "grid_indices");
  check_cuda_contig(domain_min, "domain_min");
  check_cuda_contig(domain_max, "domain_max");

  check_dtype(mesh_vertices, torch::kFloat32, "mesh_vertices");
  check_dtype(vertex_owner, torch::kInt32, "vertex_owner");
  check_dtype(rb_active, torch::kInt32, "rb_active");
  check_dtype(grid_count, torch::kInt32, "grid_count");
  check_dtype(grid_indices, torch::kInt32, "grid_indices");
  check_dtype(domain_min, torch::kFloat32, "domain_min");
  check_dtype(domain_max, torch::kFloat32, "domain_max");

  build_vertex_grid_cuda(mesh_vertices, vertex_owner, rb_active, grid_count,
                         grid_indices, domain_min, domain_max);
}

// ----------------- collisions wrapper -----------------
void rigid_rigid_collisions(
    torch::Tensor mesh_vertices, torch::Tensor rb_pos, torch::Tensor rb_lin_vel,
    torch::Tensor rb_ang_vel, torch::Tensor rb_inv_inertia_world,
    torch::Tensor rb_inv_mass, torch::Tensor rb_restitution,
    torch::Tensor rb_friction, torch::Tensor rb_half_extents,
    torch::Tensor rb_active, torch::Tensor rb_mesh_vert_offset,
    torch::Tensor rb_mesh_vert_count, torch::Tensor grid_count,
    torch::Tensor grid_indices, torch::Tensor domain_min,
    torch::Tensor domain_max, torch::Tensor dpos, torch::Tensor dlv,
    torch::Tensor dav, float thresh) {
  check_cuda_contig(mesh_vertices, "mesh_vertices");
  check_cuda_contig(rb_pos, "rb_pos");
  check_cuda_contig(rb_lin_vel, "rb_lin_vel");
  check_cuda_contig(rb_ang_vel, "rb_ang_vel");
  check_cuda_contig(rb_inv_inertia_world, "rb_inv_inertia_world");
  check_cuda_contig(rb_inv_mass, "rb_inv_mass");
  check_cuda_contig(rb_restitution, "rb_restitution");
  check_cuda_contig(rb_friction, "rb_friction");
  check_cuda_contig(rb_half_extents, "rb_half_extents");
  check_cuda_contig(rb_active, "rb_active");
  check_cuda_contig(rb_mesh_vert_offset, "rb_mesh_vert_offset");
  check_cuda_contig(rb_mesh_vert_count, "rb_mesh_vert_count");
  check_cuda_contig(grid_count, "grid_count");
  check_cuda_contig(grid_indices, "grid_indices");
  check_cuda_contig(domain_min, "domain_min");
  check_cuda_contig(domain_max, "domain_max");
  check_cuda_contig(dpos, "dpos");
  check_cuda_contig(dlv, "dlv");
  check_cuda_contig(dav, "dav");

  check_dtype(mesh_vertices, torch::kFloat32, "mesh_vertices");
  check_dtype(rb_pos, torch::kFloat32, "rb_pos");
  check_dtype(rb_lin_vel, torch::kFloat32, "rb_lin_vel");
  check_dtype(rb_ang_vel, torch::kFloat32, "rb_ang_vel");
  check_dtype(rb_inv_inertia_world, torch::kFloat32, "rb_inv_inertia_world");
  check_dtype(rb_inv_mass, torch::kFloat32, "rb_inv_mass");
  check_dtype(rb_restitution, torch::kFloat32, "rb_restitution");
  check_dtype(rb_friction, torch::kFloat32, "rb_friction");
  check_dtype(rb_half_extents, torch::kFloat32, "rb_half_extents");
  check_dtype(rb_active, torch::kInt32, "rb_active");
  check_dtype(rb_mesh_vert_offset, torch::kInt32, "rb_mesh_vert_offset");
  check_dtype(rb_mesh_vert_count, torch::kInt32, "rb_mesh_vert_count");
  check_dtype(grid_count, torch::kInt32, "grid_count");
  check_dtype(grid_indices, torch::kInt32, "grid_indices");
  check_dtype(domain_min, torch::kFloat32, "domain_min");
  check_dtype(domain_max, torch::kFloat32, "domain_max");
  check_dtype(dpos, torch::kFloat32, "dpos");
  check_dtype(dlv, torch::kFloat32, "dlv");
  check_dtype(dav, torch::kFloat32, "dav");

  rigid_rigid_collisions_cuda(
      mesh_vertices, rb_pos, rb_lin_vel, rb_ang_vel, rb_inv_inertia_world,
      rb_inv_mass, rb_restitution, rb_friction, rb_half_extents, rb_active,
      rb_mesh_vert_offset, rb_mesh_vert_count, grid_count, grid_indices,
      domain_min, domain_max, dpos, dlv, dav, thresh);
}

// ----------------- apply deltas wrapper -----------------
void apply_rigid_deltas(torch::Tensor rb_pos, torch::Tensor rb_lin_vel,
                        torch::Tensor rb_ang_vel, torch::Tensor dpos,
                        torch::Tensor dlv, torch::Tensor dav) {
  check_cuda_contig(rb_pos, "rb_pos");
  check_cuda_contig(rb_lin_vel, "rb_lin_vel");
  check_cuda_contig(rb_ang_vel, "rb_ang_vel");
  check_cuda_contig(dpos, "dpos");
  check_cuda_contig(dlv, "dlv");
  check_cuda_contig(dav, "dav");

  check_dtype(rb_pos, torch::kFloat32, "rb_pos");
  check_dtype(rb_lin_vel, torch::kFloat32, "rb_lin_vel");
  check_dtype(rb_ang_vel, torch::kFloat32, "rb_ang_vel");
  check_dtype(dpos, torch::kFloat32, "dpos");
  check_dtype(dlv, torch::kFloat32, "dlv");
  check_dtype(dav, torch::kFloat32, "dav");

  apply_rigid_deltas_cuda(rb_pos, rb_lin_vel, rb_ang_vel, dpos, dlv, dav);
}

// ----------------- pybind -----------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update_all_mesh_vertices", &update_all_mesh_vertices,
        "update_all_mesh_vertices (CUDA)");
  m.def("handle_domain_walls", &handle_domain_walls,
        "handle_domain_walls (CUDA)");
  m.def("build_vertex_grid", &build_vertex_grid, "build_vertex_grid (CUDA)");
  m.def("rigid_rigid_collisions", &rigid_rigid_collisions,
        "rigid_rigid_collisions (CUDA)");
  m.def("apply_rigid_deltas", &apply_rigid_deltas, "apply_rigid_deltas (CUDA)");
}
