#include "PressureSolver.h"

#include <cassert>

#include "MaterialType.h"
#include "NeighborDirection.h"

// To disable assert*() calls, uncomment this line:
// #define NDEBUG

namespace {

// Sets a 3D array, |*r|, the same size as the StaggeredGrid from which the
// 3D velocity arrays |u|, |v|, and |w| originate, containing 0.0 if the
// corresponding grid cell is SOLID or EMPTY, or the negation of the divergence
// of the fluid velocity across the grid cell if the cell is FLUID.
void MakeResidualFromVelocityDivergence(const Array3D<MaterialType>& labels,
                                        const Array3D<double>& u,
                                        const Array3D<double>& v,
                                        const Array3D<double>& w,
                                        std::size_t nx, std::size_t ny,
                                        std::size_t nz, Array3D<double>* r) {
  for (std::size_t i = 1; i < nx - 1; i++) {
    for (std::size_t j = 1; j < ny - 1; j++) {
      for (std::size_t k = 1; k < nz - 1; k++) {
        if (labels(i, j, k) != FLUID) {
          (*r)(i, j, k) = 0.0;
          continue;
        }

        double du_dx = u(i + 1, j, k) - u(i, j, k);
        double dv_dy = v(i, j + 1, k) - v(i, j, k);
        double dw_dz = w(i, j, k + 1) - w(i, j, k);
        double velocity_divergence_of_cell_ijk = du_dx + dv_dy + dw_dz;
        (*r)(i, j, k) = -velocity_divergence_of_cell_ijk;
      }
    }
  }
}

// Computes q = A * d where A is the matrix from the pressure projection
// equation, Ap = d, relating fluid pressures to the resulting fluid velocity
// divergence. This is done without directly multiplying any matrices since A is
// large and sparse: we can simply select the few entries in each row that are
// nonzero and multiply just the appropriate values from |d| matching with those
// nonzero entries of A.
void ATimes(const Array3D<double>& d, const Array3D<unsigned short>& neighbors,
            std::size_t nx, std::size_t ny, std::size_t nz,
            Array3D<double>* q) {
  const unsigned short CENTER = 7;

  for (std::size_t i = 1; i < nx - 1; i++) {
    for (std::size_t j = 1; j < ny - 1; j++) {
      for (std::size_t k = 1; k < nz - 1; k++) {
        unsigned short nbrs = neighbors(i, j, k);

        if (!nbrs) {
          (*q)(i, j, k) = 0.0;
          continue;
        }

        // Multiply A * d for the row of A corresponding to cell (i, j, k).
        // Store the result in q(i, j, k).
        (*q)(i, j, k) =
            ((nbrs & CENTER) * d(i, j, k)) -
            ((nbrs & NeighborDirection::LEFT) ? d(i - 1, j, k) : 0) -
            ((nbrs & NeighborDirection::DOWN) ? d(i, j - 1, k) : 0) -
            ((nbrs & NeighborDirection::BACK) ? d(i, j, k - 1) : 0) -
            ((nbrs & NeighborDirection::RIGHT) ? d(i + 1, j, k) : 0) -
            ((nbrs & NeighborDirection::UP) ? d(i, j + 1, k) : 0) -
            ((nbrs & NeighborDirection::FORWARD) ? d(i, j, k + 1) : 0);
      }
    }
  }
}

}  // namespace

PressureSolver::PressureSolver(std::size_t nx, std::size_t ny, std::size_t nz)
    : nx_(nx),
      ny_(ny),
      nz_(nz),
      r_(nx, ny, nz),
      d_(nx, ny, nz),
      q_(nx, ny, nz) {}

PressureSolver::~PressureSolver() {}

void PressureSolver::ProjectPressure(const Array3D<MaterialType>& labels,
                                     const Array3D<unsigned short>& neighbors,
                                     const Array3D<double>& u,
                                     const Array3D<double>& v,
                                     const Array3D<double>& w,
                                     Array3D<double>* p) {
  // Conjugate Gradient Algorithm
  //
  // Update |r_|, |d_|, and |q_| as we iterate to compute pressures |*p| that
  // minimize velocity divergence.
  (*p) = 0.0;

  MakeResidualFromVelocityDivergence(labels, u, v, w, nx_, ny_, nz_, &r_);
  d_.SetEqualTo(r_);

  double sigma = Dot(r_, r_);
  const double kFloatZero = 1.0e-6;
  double tolerance = kFloatZero * sigma;
  const std::size_t kMaxIters = 1000u;

  for (std::size_t iter = 0; iter < kMaxIters && sigma > tolerance; iter++) {
    ATimes(d_, neighbors, nx_, ny_, nz_, &q_);
    double alpha = sigma / Dot(d_, q_);
    p->PlusEquals(alpha, d_);   // *p += alpha * d_
    r_.PlusEquals(-alpha, q_);  // r_ -= alpha * q_
    double sigma_old = sigma;
    sigma = Dot(r_, r_);
    double beta = sigma / sigma_old;
    d_.EqualsPlusTimes(r_, beta, d_);  // d_ = r_ + beta * d_
  }
}
