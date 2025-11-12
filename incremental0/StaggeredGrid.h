#ifndef STAGGERED_GRID_H_
#define STAGGERED_GRID_H_

#include <cstddef>

#include "Array3D.h"

// A data type representing a grid with velocity components defined at grid cell
// boundaries and cell-specific values, including pressure, defined at grid cell
// centers
class StaggeredGrid {
 public:
  // Allocates a 3D staggered grid storing the following quantities as
  // a simulation's time proceeds:
  // - |nx| x |ny| x |nz| array of fluid pressures
  // - |nx + 1| x |ny| x |nz| array of horizontal fluid velocities
  // - |nx| x |ny + 1| + |nz| array of vertical fluid velocities
  // - |nx| x |ny| x |nz + 1| array of depth fluid velocities
  StaggeredGrid(std::size_t nx, std::size_t ny, std::size_t nz);

  // Deallocates the data this grid stores.
  ~StaggeredGrid();

  const Array3D<double>& p() const { return p_; }
  const Array3D<double>& u() const { return u_; }
  const Array3D<double>& v() const { return v_; }
  const Array3D<double>& w() const { return w_; }

 private:
  // Don't allow copy constructor to be called.
  StaggeredGrid(const StaggeredGrid& other);

  // Don't allow copy-assignment operator to be called.
  StaggeredGrid& operator=(const StaggeredGrid& other);

  // Number of rows of data this array stores (x or i direction)
  const std::size_t nx_;

  // Number of columns of data this array stores (y or j direction)
  const std::size_t ny_;

  // Depth of data this array stores (z or k direction)
  const std::size_t nz_;

  // Size of a single stack of this array's data, for convenience
  const std::size_t ny_nz_;

  // 3D array of fluid pressures
  Array3D<double> p_;

  // 3D array of horizontal velocity components
  Array3D<double> u_;

  // 3D array of vertical velocity components
  Array3D<double> v_;

  // 3D array of depth (z direction) velocity components
  Array3D<double> w_;
};

#endif  // STAGGERED_GRID_H_
