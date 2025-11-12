#include "StaggeredGrid.h"

#include <cassert>

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
  GridIndices ijk = floor(p_shift_lc_over_dx);
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

}  // namespace

StaggeredGrid::StaggeredGrid(std::size_t nx, std::size_t ny, std::size_t nz,
                             const Eigen::Vector3d& lc, double dx)
    : nx_(nx),
      ny_(ny),
      nz_(nz),
      ny_nz_(ny * nz),
      lc_(lc),
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
      cell_labels_(nx, ny, nz) {}

StaggeredGrid::~StaggeredGrid() {}

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
