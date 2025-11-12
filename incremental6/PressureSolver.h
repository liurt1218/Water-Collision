#ifndef PRESSURE_SOLVER_H_
#define PRESSURE_SOLVER_H_

#include "Array3D.h"
#include "MaterialType.h"

// A data type that computes a 3D array of fluid pressure values that minimize
// the divergence of the velocity field of a fluid in the next time step using
// the Conjugate Gradient Algorithm and encapsulates (stores) auxiliary (helper)
// 3D arrays to perform this calculation
//
// Exactly one instance of this class shall be owned by a StaggeredGrid.
class PressureSolver {
 public:
  // Creates auxiliary (helper) 3D arrays for the Conjugate Gradient Algorithm
  // representing the residual vector, direction vector, and matrix-mapped
  // direction vector for the pressure projection matrix equation, Ap = d, where
  // A maps pressures to scaled flipped velocity divergence values, p is the
  // "vector" (mathematically a vector but stored as a 3D array here) of the
  // pressure values of a StaggeredGrid's cells, and d is the "vector" of
  // scaled, flipped velocity divergence values for each cell. All arrays are
  // the same size as the StaggeredGrid that owns |this|: |nx| x |ny| x |nz|.
  PressureSolver(std::size_t nx, std::size_t ny, std::size_t nz);

  // Deallocates the data this grid stores.
  ~PressureSolver();

  // Computes pressure values for the grid cells to update grid velocities at
  // the next time step that are as divergence-free as possible.
  void ProjectPressure(const Array3D<MaterialType>& labels,
                       const Array3D<unsigned short>& neighbors,
                       const Array3D<double>& u, const Array3D<double>& v,
                       const Array3D<double>& w, Array3D<double>* p);

 private:
  // Don't allow copy constructor to be called.
  PressureSolver(const PressureSolver& other);

  // Don't allow copy-assignment operator to be called.
  PressureSolver& operator=(const PressureSolver& other);

  // Number of rows of data this array stores (x or i direction)
  const std::size_t nx_;

  // Number of columns of data this array stores (y or j direction)
  const std::size_t ny_;

  // Depth of data this array stores (z or k direction)
  const std::size_t nz_;

  // Residual values for pressure projection
  Array3D<double> r_;

  // Direction vectors used to "take steps" toward the minimum point of the
  // quadratic form for the pressure projection matrix equation
  Array3D<double> d_;

  // Matrix-mapped direction vector used to update the residual values in each
  // step of the Conjugate Gradient Algorithm
  Array3D<double> q_;
};

#endif  // PRESSURE_SOLVER_H_
