#include "StaggeredGrid.h"

#include <cassert>

#include "NeighborDirection.h"

// To disable assert*() calls, uncomment this line:
// #define NDEBUG

namespace {

// This represents a triplet of indices as a column vector of nonnegative
// integers:
// [ i ]
// [ j ]
// [ k ]
typedef Eigen::Matrix<std::size_t, 3, 1> GridIndices;

const double kFloatZero = 1.0e-6;
const double kClampCushion = 1.0e-4;
const double kGravAccMetersPerSecond = 9.80665;

Eigen::Vector3d HalfShiftYZ(double dx) {
  Eigen::Vector3d half_shift;
  double dx_2 = dx / 2.0;
  half_shift << 0.0, dx_2, dx_2;
  return half_shift;
}

Eigen::Vector3d HalfShiftXZ(double dx) {
  Eigen::Vector3d half_shift;
  double dx_2 = dx / 2.0;
  half_shift << dx_2, 0.0, dx_2;
  return half_shift;
}

Eigen::Vector3d HalfShiftXY(double dx) {
  Eigen::Vector3d half_shift;
  double dx_2 = dx / 2.0;
  half_shift << dx_2, dx_2, 0.0;
  return half_shift;
}

// Returns the indices of the grid cell you end up in if you start at (0, 0, 0)
// and shift by the real-number amount of grid cells in |p_lc_over_dx|.
//
// If |p_lc_over_dx| contains any negative values and assertions are on, then
// an assertion failure will crash this program.
inline GridIndices floor(const Eigen::Vector3d& p_lc_over_dx) {
  // Ensure we won't end up with negative indices.
  assert(p_lc_over_dx[0] >= 0.0);
  assert(p_lc_over_dx[1] >= 0.0);
  assert(p_lc_over_dx[2] >= 0.0);

  // Indices are valid. Construct and return them.
  // This casts the elements of the vector above as nonnegative integers.
  return p_lc_over_dx.cast<std::size_t>();
}

// Returns the indices of the grid cell containing the point |p_lc| in a grid
// with grid cell width (spacing) |dx|. It is assumed |p_lc| is *relative to*
// the lower corner of the grid, not the absolute position of a point in |R^3.
//
// If |dx| <= 0 or |p_lc|'s location would result in negative indices being
// returned and assertions are on, then an assertion failure will crash this
// program.
inline GridIndices floor(const Eigen::Vector3d& p_lc, double dx) {
  // Ensure grid spacings are positive.
  assert(dx > 0.0);

  // Dividing by |dx| yields a 3D vector indicating the number of grid
  // cells (including fractions of grid cells, as the vector elements are
  // floating-point values) away from |lc| that |p| is located.
  return floor(p_lc / dx);
}

// Returns the barycentric weights of the particle relative to the grid cell you
// end up in if you start at (0, 0, 0) and shift by the real-number amount of
// grid cells in |p_lc_over_dx|.
//
// It is assumed |indices| were obtained by calling the floor function above,
// which would have already checked for any negative elemtns in |p_lc_over_dx|.
inline Eigen::Vector3d GetWeights(const Eigen::Vector3d& p_lc_over_dx,
                                  const GridIndices& indices) {
  return p_lc_over_dx - indices.cast<double>();
}

// Computes a velocity, via trilinear interpolation, for a particle whose
// position has been shifted negatively in the dimensions other than the
// dimension of the velocities to be interpolated.
double InterpolateGridVelocities(
    const Eigen::Vector3d& shifted_particle_position_lc,
    const Array3D<double>& grid_vels, double dx) {
  Eigen::Vector3d p_shift_lc_over_dx = shifted_particle_position_lc / dx;

  // Determine the grid cell containing the shifted particle position.
  GridIndices ijk = floor(p_shift_lc_over_dx);

  // Determine the barycentric weights of the shifted particle position inside
  // that grid cell.
  Eigen::Vector3d weights = GetWeights(p_shift_lc_over_dx, ijk);

  double w0 = weights[0];
  double om_w0 = 1.0 - w0;
  double w1 = weights[1];
  double om_w1 = 1.0 - w1;
  double w2 = weights[2];
  double om_w2 = 1.0 - w2;
  std::size_t i = ijk[0];
  std::size_t j = ijk[1];
  std::size_t k = ijk[2];

  // Trilinearly interpolate grid velocities to get a velocity for the particle.
  return om_w0 * om_w1 * om_w2 * grid_vels(i, j, k) +
         om_w0 * om_w1 * w2 * grid_vels(i, j, k + 1) +
         om_w0 * w1 * om_w2 * grid_vels(i, j + 1, k) +
         om_w0 * w1 * w2 * grid_vels(i, j + 1, k + 1) +
         w0 * om_w1 * om_w2 * grid_vels(i + 1, j, k) +
         w0 * om_w1 * w2 * grid_vels(i + 1, j, k + 1) +
         w0 * w1 * om_w2 * grid_vels(i + 1, j + 1, k) +
         w0 * w1 * w2 * grid_vels(i + 1, j + 1, k + 1);
}

void Contribute(double weight, double particle_velocity,
                Array3D<double>* grid_vels, Array3D<double>* grid_vel_weights,
                std::size_t i, std::size_t j, std::size_t k) {
  (*grid_vels)(i, j, k) += weight * particle_velocity;
  (*grid_vel_weights)(i, j, k) += weight;
}

void Splat(const Eigen::Vector3d& shifted_particle_position_lc, double dx,
           double particle_velocity, Array3D<double>* grid_vels,
           Array3D<double>* grid_vel_weights) {
  Eigen::Vector3d p_shift_lc_over_dx = shifted_particle_position_lc / dx;

  // Determine the grid cell containing the shifted particle position.
  GridIndices ijk = floor(p_shift_lc_over_dx);

  // Determine the barycentric weights of the shifted particle position inside
  // that grid cell.
  Eigen::Vector3d weights = GetWeights(p_shift_lc_over_dx, ijk);

  double w0 = weights[0];
  double om_w0 = 1.0 - w0;
  double w1 = weights[1];
  double om_w1 = 1.0 - w1;
  double w2 = weights[2];
  double om_w2 = 1.0 - w2;
  std::size_t i = ijk[0];
  std::size_t j = ijk[1];
  std::size_t k = ijk[2];

  Contribute(om_w0 * om_w1 * om_w2, particle_velocity, grid_vels,
             grid_vel_weights, i, j, k);
  Contribute(om_w0 * om_w1 * w2, particle_velocity, grid_vels, grid_vel_weights,
             i, j, k + 1);
  Contribute(om_w0 * w1 * om_w2, particle_velocity, grid_vels, grid_vel_weights,
             i, j + 1, k);
  Contribute(om_w0 * w1 * w2, particle_velocity, grid_vels, grid_vel_weights, i,
             j + 1, k + 1);
  Contribute(w0 * om_w1 * om_w2, particle_velocity, grid_vels, grid_vel_weights,
             i + 1, j, k);
  Contribute(w0 * om_w1 * w2, particle_velocity, grid_vels, grid_vel_weights,
             i + 1, j, k + 1);
  Contribute(w0 * w1 * om_w2, particle_velocity, grid_vels, grid_vel_weights,
             i + 1, j + 1, k);
  Contribute(w0 * w1 * w2, particle_velocity, grid_vels, grid_vel_weights,
             i + 1, j + 1, k + 1);
}

MaterialType GetNeighborMaterial(const Array3D<MaterialType>& cell_labels,
                                 std::size_t i, std::size_t j, std::size_t k,
                                 NeighborDirection dir) {
  switch (dir) {
    case LEFT:
      return cell_labels(i - 1, j, k);
    case DOWN:
      return cell_labels(i, j - 1, k);
    case BACK:
      return cell_labels(i, j, k - 1);
    case RIGHT:
      return cell_labels(i + 1, j, k);
    case UP:
      return cell_labels(i, j + 1, k);
    case FORWARD:
      return cell_labels(i, j, k + 1);
  }
  // No default case: switch cases should cover all possibilities
  assert(false);
  return SOLID;  // for compiler happiness, should never get executed
}

unsigned short UpdateFromNeighbor(unsigned short nbr_info,
                                  MaterialType nbr_material,
                                  NeighborDirection dir) {
  unsigned short new_nbr_info = nbr_info;

  if (nbr_material != SOLID) {
    new_nbr_info++;
  }

  if (nbr_material != FLUID) {
    return new_nbr_info;
  }

  return new_nbr_info | dir;
}

void MakeNeighborMaterialInfo(const Array3D<MaterialType>& cell_labels,
                              Array3D<unsigned short>* neighbors) {
  (*neighbors) = 0u;

  for (std::size_t i = 1; i < cell_labels.nx() - 1; i++) {
    for (std::size_t j = 1; j < cell_labels.ny() - 1; j++) {
      for (std::size_t k = 1; k < cell_labels.nz() - 1; k++) {
        if (cell_labels(i, j, k) != FLUID) {
          continue;
        }

        unsigned short nbr_info = 0u;
        for (NeighborDirection dir : kNeighborDirections) {
          MaterialType nbr_material =
              GetNeighborMaterial(cell_labels, i, j, k, dir);
          nbr_info = UpdateFromNeighbor(nbr_info, nbr_material, dir);
        }

        (*neighbors)(i, j, k) = nbr_info;
      }
    }
  }
}

}  // namespace

StaggeredGrid::StaggeredGrid(std::size_t nx, std::size_t ny, std::size_t nz,
                             const Eigen::Vector3d& lc, double dx)
    : nx_(nx),
      ny_(ny),
      nz_(nz),
      ny_nz_(ny * nz),
      lc_(lc),
      uc_(lc + Eigen::Vector3d(nx, ny, nz) * dx),
      dx_(dx),
      half_shift_yz_(HalfShiftYZ(dx)),
      half_shift_xz_(HalfShiftXZ(dx)),
      half_shift_xy_(HalfShiftXY(dx)),
      p_(nx, ny, nz),
      u_(nx + 1, ny, nz),
      v_(nx, ny + 1, nz),
      w_(nx, ny, nz + 1),
      fu_(nx + 1, ny, nz),
      fv_(nx, ny + 1, nz),
      fw_(nx, ny, nz + 1),
      cell_labels_(nx, ny, nz),
      neighbors_(nx, ny, nz),
      pressure_solver_(nx, ny, nz) {}

StaggeredGrid::~StaggeredGrid() {}

Eigen::Vector3d StaggeredGrid::Advect(const Eigen::Vector3d& pos,
                                      double dt) const {
  Eigen::Vector3d p_lc(pos - lc_);
  double u_p = InterpolateGridVelocities(p_lc - half_shift_yz_, u_, dx_);
  double v_p = InterpolateGridVelocities(p_lc - half_shift_xz_, v_, dx_);
  double w_p = InterpolateGridVelocities(p_lc - half_shift_xy_, w_, dx_);
  return ClampToNonSolidCells(pos + dt * Eigen::Vector3d(u_p, v_p, w_p));
}

inline Eigen::Vector3d StaggeredGrid::ClampToNonSolidCells(
    const Eigen::Vector3d& pos) const {
  Eigen::Vector3d clamped_pos = pos;
  const double cell_plus_cushion = dx_ + kClampCushion;

  for (std::size_t i = 0; i < 3; i++) {
    double min = lc_[i] + cell_plus_cushion;
    if (pos[i] <= min) {
      clamped_pos[i] = min;
      continue;
    }

    double max = uc_[i] - cell_plus_cushion;
    if (pos[i] >= max) {
      clamped_pos[i] = max;
    }
  }

  return clamped_pos;
}

void StaggeredGrid::ParticlesToGrid(const std::vector<Particle>& particles) {
  ZeroOutVelocities();
  ClearCellLabels();

  for (std::vector<Particle>::const_iterator p = particles.begin();
       p != particles.end(); p++) {
    Eigen::Vector3d p_lc(p->pos - lc_);
    SetParticlesCellToFluid(p_lc);

    Splat(p_lc - half_shift_yz_, dx_, p->vel[0], &u_, &fu_);
    Splat(p_lc - half_shift_xz_, dx_, p->vel[1], &v_, &fv_);
    Splat(p_lc - half_shift_xy_, dx_, p->vel[2], &w_, &fw_);
  }

  NormalizeHorizontalVelocities();
  NormalizeVerticalVelocities();
  NormalizeDepthVelocities();

  SetBoundaryVelocities();
}

void StaggeredGrid::ZeroOutVelocities() {
  u_ = 0.0;
  fu_ = 0.0;
  v_ = 0.0;
  fv_ = 0.0;
  w_ = 0.0;
  fw_ = 0.0;
}

void StaggeredGrid::ClearCellLabels() {
  SetOuterCellLabelsToSolid();
  SetInnerCellLabelsToEmpty();
}

void StaggeredGrid::SetOuterCellLabelsToSolid() {
  // There's some duplicate assignment of grid cells that are on the
  // corners of the grid. Plus, these solid settings could be set once on
  // construction of |this| StaggeredGrid, but just in case something gets
  // changed we'll make sure to reset it here in every time step of the
  // simulation.

  // All grid cells on the left and right faces of the grid are solid.
  for (std::size_t j = 0; j < ny_; j++) {
    for (std::size_t k = 0; k < nz_; k++) {
      cell_labels_(0, j, k) = MaterialType::SOLID;
      cell_labels_(nx_ - 1, j, k) = MaterialType::SOLID;
    }
  }

  // All grid cells on the bottom and top faces of the grid are solid.
  for (std::size_t i = 0; i < nx_; i++) {
    for (std::size_t k = 0; k < nz_; k++) {
      cell_labels_(i, 0, k) = MaterialType::SOLID;
      cell_labels_(i, ny_ - 1, k) = MaterialType::SOLID;
    }
  }

  // All grid cells on the back and front faces of the grid are solid.
  for (std::size_t i = 0; i < nx_; i++) {
    for (std::size_t j = 0; j < ny_; j++) {
      cell_labels_(i, j, 0) = MaterialType::SOLID;
      cell_labels_(i, j, nz_ - 1) = MaterialType::SOLID;
    }
  }
}

void StaggeredGrid::SetInnerCellLabelsToEmpty() {
  for (std::size_t i = 1; i < nx_ - 1; i++) {
    for (std::size_t j = 1; j < ny_ - 1; j++) {
      for (std::size_t k = 1; k < nz_ - 1; k++) {
        cell_labels_(i, j, k) = MaterialType::EMPTY;
      }
    }
  }
}

void StaggeredGrid::SetParticlesCellToFluid(const Eigen::Vector3d& p_lc) {
  GridIndices ijk = floor(p_lc, dx_);
  cell_labels_(ijk[0], ijk[1], ijk[2]) = MaterialType::FLUID;
}

void StaggeredGrid::NormalizeHorizontalVelocities() {
  // Set boundary velocities to zero.
  for (std::size_t j = 0; j < ny_; j++) {
    for (std::size_t k = 0; k < nz_; k++) {
      u_(0, j, k) = 0.0;
      u_(1, j, k) = 0.0;
      u_(nx_ - 1, j, k) = 0.0;
      u_(nx_, j, k) = 0.0;
    }
  }

  // Normalize the non-boundary velocities unless the corresponding
  // velocity-weight is small.
  for (std::size_t i = 2; i < nx_ - 1; i++) {
    for (std::size_t j = 0; j < ny_; j++) {
      for (std::size_t k = 0; k < nz_; k++) {
        if (fu_(i, j, k) < kFloatZero) {
          u_(i, j, k) = 0.0;
          continue;
        }
        u_(i, j, k) /= fu_(i, j, k);
      }
    }
  }
}

void StaggeredGrid::NormalizeVerticalVelocities() {
  // Set boundary velocities to zero.
  for (std::size_t i = 0; i < nx_; i++) {
    for (std::size_t k = 0; k < nz_; k++) {
      v_(i, 0, k) = 0.0;
      v_(i, 1, k) = 0.0;
      v_(i, ny_ - 1, k) = 0.0;
      v_(i, ny_, k) = 0.0;
    }
  }

  // Normalize the non-boundary velocities unless the corresponding
  // velocity-weight is small.
  for (std::size_t i = 0; i < nx_; i++) {
    for (std::size_t j = 2; j < ny_ - 1; j++) {
      for (std::size_t k = 0; k < nz_; k++) {
        if (fv_(i, j, k) < kFloatZero) {
          v_(i, j, k) = 0.0;
          continue;
        }
        v_(i, j, k) /= fv_(i, j, k);
      }
    }
  }
}

void StaggeredGrid::NormalizeDepthVelocities() {
  // Set boundary velocities to zero.
  for (std::size_t i = 0; i < nx_; i++) {
    for (std::size_t j = 0; j < ny_; j++) {
      w_(i, j, 0) = 0.0;
      w_(i, j, 1) = 0.0;
      w_(i, j, nz_ - 1) = 0.0;
      w_(i, j, nz_) = 0.0;
    }
  }

  // Normalize the non-boundary velocities unless the corresponding
  // velocity-weight is small.
  for (std::size_t i = 0; i < nx_; i++) {
    for (std::size_t j = 0; j < ny_; j++) {
      for (std::size_t k = 2; k < nz_ - 1; k++) {
        if (fw_(i, j, k) < kFloatZero) {
          w_(i, j, k) = 0.0;
          continue;
        }
        w_(i, j, k) /= fw_(i, j, k);
      }
    }
  }
}

void StaggeredGrid::SetBoundaryVelocities() {
  // These are the "boundary conditions."

  for (std::size_t j = 0; j < ny_; j++) {
    for (std::size_t k = 0; k < nz_; k++) {
      // Zero out horizontal velocities on either side of each grid cell on the
      // left and right boundary walls of the grid.
      u_(0, j, k) = 0.0;
      u_(1, j, k) = 0.0;
      u_(nx_ - 1, j, k) = 0.0;
      u_(nx_, j, k) = 0.0;

      // Copy boundary-adjacent velocities in other directions
      // to the boundary velocities.
      v_(0, j, k) = v_(1, j, k);
      v_(nx_ - 1, j, k) = v_(nx_ - 2, j, k);

      w_(0, j, k) = w_(1, j, k);
      w_(nx_ - 1, j, k) = w_(nx_ - 2, j, k);
    }
  }

  for (std::size_t i = 0; i < nx_; i++) {
    for (std::size_t k = 0; k < nz_; k++) {
      // Zero out vertical velocities on either side of each grid cell on the
      // bottom and top boundary walls of the grid.
      v_(i, 0, k) = 0.0;
      v_(i, 1, k) = 0.0;
      v_(i, ny_ - 1, k) = 0.0;
      v_(i, ny_, k) = 0.0;

      // Copy boundary-adjacent velocities in other directions
      // to the boundary velocities.
      u_(i, 0, k) = u_(i, 1, k);
      u_(i, ny_ - 1, k) = u_(i, ny_ - 2, k);

      w_(i, 0, k) = w_(i, 1, k);
      w_(i, ny_ - 1, k) = w_(i, ny_ - 2, k);
    }
  }

  for (std::size_t i = 0; i < nx_; i++) {
    for (std::size_t j = 0; j < ny_; j++) {
      // Zero out depth velocities on either side of each grid cell on the back
      // and front boundary walls of the grid.
      w_(i, j, 0) = 0.0;
      w_(i, j, 1) = 0.0;
      w_(i, j, nz_ - 1) = 0.0;
      w_(i, j, nz_) = 0.0;

      // Copy boundary-adjacent velocities in other directions
      // to the boundary velocities.
      u_(i, j, 0) = u_(i, j, 1);
      u_(i, j, nz_ - 1) = u_(i, j, nz_ - 2);

      v_(i, j, 0) = v_(i, j, 1);
      v_(i, j, nz_ - 1) = v_(i, j, nz_ - 2);
    }
  }
}

void StaggeredGrid::ApplyGravity(double dt) {
  double vertical_velocity_change = -dt * kGravAccMetersPerSecond;
  for (std::size_t i = 0; i < nx_; i++) {
    for (std::size_t j = 0; j < ny_ + 1; j++) {
      for (std::size_t k = 0; k < nz_; k++) {
        v_(i, j, k) += vertical_velocity_change;
      }
    }
  }

  // Make sure we fix the boundary velocities that we just changed!
  SetBoundaryVelocities();
}

void StaggeredGrid::ProjectPressure() {
  // Cache which neighbors are non-SOLID and which ones are FLUID.
  MakeNeighborMaterialInfo(cell_labels_, &neighbors_);

  // Determine fluid pressures that make fluid velocity as divergence-free as
  // we reasonably can.
  pressure_solver_.ProjectPressure(cell_labels_, neighbors_, u_, v_, w_, &p_);

  // Update grid fluid velocity values based on the fluid pressure gradient.
  SubtractPressureGradientFromVelocity();
}

void StaggeredGrid::SubtractPressureGradientFromVelocity() {
  for (std::size_t i = 1; i < nx_ - 1; i++) {
    for (std::size_t j = 1; j < ny_ - 1; j++) {
      for (std::size_t k = 1; k < nz_ - 1; k++) {
        if (cell_labels_(i, j, k) == SOLID) {
          continue;
        }
        double pijk = p_(i, j, k);
        if (cell_labels_(i - 1, j, k) != SOLID) {
          u_(i, j, k) -= pijk - p_(i - 1, j, k);
        }
        if (cell_labels_(i, j - 1, k) != SOLID) {
          v_(i, j, k) -= pijk - p_(i, j - 1, k);
        }
        if (cell_labels_(i, j, k - 1) != SOLID) {
          w_(i, j, k) -= pijk - p_(i, j, k - 1);
        }
      }
    }
  }
}
