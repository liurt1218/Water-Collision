#ifndef STAGGERED_GRID_H_
#define STAGGERED_GRID_H_

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

#include "Array3D.h"
#include "Particle.h"

// A data type representing a grid with velocity components defined at grid cell
// boundaries and cell-specific values, including pressure, defined at grid cell
// centers
class StaggeredGrid {
 public:
  // Used to label cells
  enum MaterialType { EMPTY, SOLID, FLUID };

  // Allocates a 3D staggered grid storing the following quantities as
  // a simulation's time proceeds:
  // - |nx| x |ny| x |nz| array of fluid pressures
  // - |nx + 1| x |ny| x |nz| array of horizontal fluid velocities
  // - |nx| x |ny + 1| + |nz| array of vertical fluid velocities
  // - |nx| x |ny| x |nz + 1| array of depth fluid velocities
  // - |lc| is the lower corner (min x, y, z) position of the grid
  // - |dx| is the grid cell width (side length)
  StaggeredGrid(std::size_t nx, std::size_t ny, std::size_t nz,
                const Eigen::Vector3d& lc, double dx);

  // Deallocates the data this grid stores.
  ~StaggeredGrid();

  const Array3D<double>& p() const { return p_; }
  const Array3D<double>& u() const { return u_; }
  const Array3D<double>& v() const { return v_; }
  const Array3D<double>& w() const { return w_; }
  const Array3D<MaterialType>& cell_labels() const { return cell_labels_; }

  // Advects velocity for a particle located at |pos|.
  Eigen::Vector3d Advect(const Eigen::Vector3d& pos, double dt) const;

  // Transfers particle velocities to this grid.
  void ParticlesToGrid(const std::vector<Particle>& particles);

  // Subtracts |dt| times acceleration due to gravity to all vertical velocities
  // in this grid.
  void ApplyGravity(double dt);

 private:
  // Don't allow copy constructor to be called.
  StaggeredGrid(const StaggeredGrid& other);

  // Don't allow copy-assignment operator to be called.
  StaggeredGrid& operator=(const StaggeredGrid& other);

  // Returns the result of clamping |pos| to stay within the non-SOLID cells
  // with a small floating-point buffer.
  inline Eigen::Vector3d ClampToNonSolidCells(const Eigen::Vector3d& pos) const;

  void ZeroOutVelocities();

  void ClearCellLabels();
  void SetOuterCellLabelsToSolid();
  void SetInnerCellLabelsToEmpty();

  // Sets the label of the cell containing the particle with position |p_lc|
  // relative to the grid's lower corner to |FLUID|.
  void SetParticlesCellToFluid(const Eigen::Vector3d& p_lc);

  void NormalizeHorizontalVelocities();
  void NormalizeVerticalVelocities();
  void NormalizeDepthVelocities();

  // Sets boundary conditions on the grid velocities.
  void SetBoundaryVelocities();

  // Number of rows of data this array stores (x or i direction)
  const std::size_t nx_;

  // Number of columns of data this array stores (y or j direction)
  const std::size_t ny_;

  // Depth of data this array stores (z or k direction)
  const std::size_t nz_;

  // Size of a single stack of this array's data, for convenience
  const std::size_t ny_nz_;

  // Lower corner position (min x, y, z) of the grid
  const Eigen::Vector3d lc_;

  // Upper corner position (max x, y, z) of the grid
  const Eigen::Vector3d uc_;

  // Grid cell width (side length)
  const double dx_;

  // Half-grid-cell-width shifts for splatting particle values onto the grid
  const Eigen::Vector3d half_shift_yz_;  // (0, dx_/2, dx_/2)
  const Eigen::Vector3d half_shift_xz_;  // (dx_/2, 0, dx_/2)
  const Eigen::Vector3d half_shift_xy_;  // (dx_/2, dx_/2, 0)

  // 3D array of fluid pressures
  Array3D<double> p_;

  // 3D array of horizontal velocity components
  Array3D<double> u_;

  // 3D array of vertical velocity components
  Array3D<double> v_;

  // 3D array of depth (z direction) velocity components
  Array3D<double> w_;

  // Accumulated particle velocity-weights for each grid velocity component
  Array3D<double> fu_;  // horizontal
  Array3D<double> fv_;  // vertical
  Array3D<double> fw_;  // depth

  // Material type of each grid cell
  Array3D<MaterialType> cell_labels_;
};

#endif  // STAGGERED_GRID_H_
