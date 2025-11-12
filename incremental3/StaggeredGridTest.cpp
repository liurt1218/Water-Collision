#include <cassert>
#include <iostream>
#include <vector>

#include "Particle.h"
#include "SimulationParameters.h"
#include "StaggeredGrid.h"

namespace {

Eigen::Vector3d Make3d(double x, double y, double z) {
  Eigen::Vector3d p;
  p << x, y, z;
  return p;
}

Particle MakeParticle(double x, double y, double z, double vx, double vy,
                      double vz) {
  return Particle{.pos = Make3d(x, y, z), .vel = Make3d(vx, vy, vz)};
}

const double kFloatZero = 1.0e-6;

bool FuzzyEquals(double a, double b) {
  double a_b = a - b;
  return -kFloatZero < a_b && a_b < kFloatZero;
}

bool FuzzyNotZero(double x) { return x > kFloatZero || x < -kFloatZero; }

bool IsZero(const Array3D<double>* arr) {
  for (std::size_t i = 0; i < arr->nx(); i++) {
    for (std::size_t j = 0; j < arr->ny(); j++) {
      for (std::size_t k = 0; k < arr->nz(); k++) {
        if (FuzzyNotZero((*arr)(i, j, k))) {
          return false;
        }
      }
    }
  }

  return true;
}

bool IsSolid(StaggeredGrid::MaterialType label) {
  return label == StaggeredGrid::MaterialType::SOLID;
}

bool IsEmpty(StaggeredGrid::MaterialType label) {
  return label == StaggeredGrid::MaterialType::EMPTY;
}

bool IsFluid(StaggeredGrid::MaterialType label) {
  return label == StaggeredGrid::MaterialType::FLUID;
}

void CheckOuterCellsAreSolid(const StaggeredGrid& grid, std::size_t nx,
                             std::size_t ny, std::size_t nz) {
  // Outer cell labels should be |SOLID|.
  for (std::size_t j = 0; j < ny; j++) {
    for (std::size_t k = 0; k < nz; k++) {
      assert(IsSolid(grid.cell_labels()(0, j, k)));
      assert(IsSolid(grid.cell_labels()(nx - 1, j, k)));
    }
  }
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t k = 0; k < nz; k++) {
      assert(IsSolid(grid.cell_labels()(i, 0, k)));
      assert(IsSolid(grid.cell_labels()(i, ny - 1, k)));
    }
  }
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      assert(IsSolid(grid.cell_labels()(i, j, 0)));
      assert(IsSolid(grid.cell_labels()(i, j, nz - 1)));
    }
  }
}

void CheckEmptyParticleSplat(const StaggeredGrid& grid, std::size_t nx,
                             std::size_t ny, std::size_t nz) {
  // Check grid's arrays' sizes against size we used to initialize grid.
  assert(grid.u().nx() == nx + 1);
  assert(grid.u().ny() == ny);
  assert(grid.u().nz() == nz);

  assert(grid.v().nx() == nx);
  assert(grid.v().ny() == ny + 1);
  assert(grid.v().nz() == nz);

  assert(grid.w().nx() == nx);
  assert(grid.w().ny() == ny);
  assert(grid.w().nz() == nz + 1);

  // All velocities should be zero.
  assert(IsZero(&grid.u()));
  assert(IsZero(&grid.v()));
  assert(IsZero(&grid.w()));

  CheckOuterCellsAreSolid(grid, nx, ny, nz);

  // All interior cell labels should be |EMPTY|.
  for (std::size_t i = 1; i < nx - 1; i++) {
    for (std::size_t j = 1; j < ny - 1; j++) {
      for (std::size_t k = 1; k < nz - 1; k++) {
        assert(IsEmpty(grid.cell_labels()(i, j, k)));
      }
    }
  }
}

void CheckOneParticleSplat(const StaggeredGrid& grid, std::size_t nx,
                           std::size_t ny, std::size_t nz) {
  // Particle is at (2.75, 3.25, 2.5).
  //
  // Our grid has lower corner (0, 0, 0) with grid cell width of 1.
  // This particle is in the cell with indices (2, 3, 2).
  // Thus, the particle has weights (0.75, 0.25, 0.5). So its splats will be:
  // - Shift to (2.75, 2.75, 2), which is in cell (2, 2, 2) with weights
  //   (0.75, 0.75, 0). Splat the particle's u velocity, 10.0, onto the grid:
  //   . (i, j, k) = (2, 2, 2)
  //   . (w0, w1, w2) = (0.75, 0.75, 0)
  //   . 1-w0 = 0.25, 1-w1 = 0.25, 1-w2 = 1
  //   . u(i, j, k) = (1-w0)(1-w1)(1-w2)u = 0.0625 * 10.0 = 0.625
  //     But dividing by weights just yields 10.0
  assert(FuzzyEquals(grid.u()(2, 2, 2), 10.0));
  //   . u(i, j, k + 1) = (1-w0)(1-w1)w2u = 0
  assert(FuzzyEquals(grid.u()(2, 2, 3), 0.0));
  //   . u(i, j + 1, k) = (1-w0)w1(1-w2)u = 3/16 * 10.0 = 2-0.125 = 1.875
  //     But dividing by weights just yields 10.0
  assert(FuzzyEquals(grid.u()(2, 3, 2), 10.0));
  //   . u(i, j + 1, k + 1) = (1-w0)w1w2u = 0
  assert(FuzzyEquals(grid.u()(2, 3, 3), 0.0));
  //   . u(i + 1, j, k) = w0(1-w1)(1-w2)u = 3/16 * 10.0 = 1.875
  assert(FuzzyEquals(grid.u()(3, 2, 2), 10.0));
  //     But dividing by weights just yields 10.0
  //   . u(i + 1, j, k + 1) = w0(1-w1)w2u = 0
  assert(FuzzyEquals(grid.u()(3, 2, 3), 0.0));
  //   . u(i + 1, j + 1, k) = w0w1(1-w2)u = 9/16 * 10.0 = 5.625
  //     But dividing by weights just yields 10.0
  assert(FuzzyEquals(grid.u()(3, 3, 2), 10.0));
  //   . u(i + 1, j + 1, k + 1) = w0w1w2u = 0
  assert(FuzzyEquals(grid.u()(3, 3, 3), 0.0));
  for (std::size_t i = 0; i < nx + 1; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        if (!((i == 2 || i == 3) && (j == 2 || j == 3) && (k == 2 || k == 3))) {
          assert(FuzzyEquals(grid.u()(i, j, k), 0.0));
        }
      }
    }
  }
  // - Shift to (2.25, 3.25, 2), in cell (2, 3, 2) with weights (0.25, 0.25, 0).
  //   Splat the particle's v velocity, 20.0, onto the grid:
  //   . (i, j, k) = (2, 3, 2)
  //   . (w0, w1, w2) = (0.25, 0.25, 0)
  //   . (1-w0, 1-w1, 1-w2) = (0.75, 0.75, 1)
  //   . v(i, j, k) = (1-w0)(1-w1)(1-w2)v = 9/16 * 20.0 = 9/4 * 5.0 = 11.25
  //     But dividing by weights just yields 20.0
  assert(FuzzyEquals(grid.v()(2, 3, 2), 20.0));
  //   . v(i, j, k + 1) = (1-w0)(1-w1)w2v = 0
  assert(FuzzyEquals(grid.v()(2, 3, 3), 0.0));
  //   . v(i, j + 1, k) = (1-w0)w1(1-w2)v = 3/16 * 20.0 = 3/4 * 5.0 = 3.75
  //     But dividing by weights just yields 20.0
  assert(FuzzyEquals(grid.v()(2, 4, 2), 20.0));
  //   . v(i, j + 1, k + 1) = (1-w0)w1w2v = 0
  assert(FuzzyEquals(grid.v()(2, 4, 3), 0.0));
  //   . v(i + 1, j, k) = w0(1-w1)(1-w2)v = 3/16 * 20.0 = 3.75
  //     But dividing by weights just yields 20.0
  assert(FuzzyEquals(grid.v()(3, 3, 2), 20.0));
  //   . v(i + 1, j, k + 1) = w0(1-w1)w2v = 0
  assert(FuzzyEquals(grid.v()(3, 3, 3), 0.0));
  //   . v(i + 1, j + 1, k) = w0w1(1-w2)v = 1/16 * 20.0 = 5.0/4.0 = 1.25
  //     But dividing by weights just yields 20.0
  assert(FuzzyEquals(grid.v()(3, 4, 2), 20.0));
  //   . v(i + 1, j + 1, k + 1) = w0w1w2v = 0
  assert(FuzzyEquals(grid.v()(3, 4, 3), 0.0));
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny + 1; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        if (!((i == 2 || i == 3) && (j == 3 || j == 4) && (k == 2 || k == 3))) {
          assert(FuzzyEquals(grid.v()(i, j, k), 0.0));
        }
      }
    }
  }
  // - Shift to (2.25, 2.75, 2.5), in cell (2, 2, 2) with weights
  //   (0.25, 0.75, 0.5). Splat the particle's w velocity, 30.0, onto the grid:
  //   . (i, j, k) = (2, 2, 2)
  //   . (w0, w1, w2) = (0.25, 0.75, 0.5)
  //   . (1-w0, 1-w1, 1-w2) = (0.75, 0.25, 0.5)
  //   . w(i, j, k) = (1-w0)(1-w1)(1-w2)w = 3/4*1/4*1/2 * 30.0 = 3/32*30.0
  //     But dividing by weights just yields 30.0
  assert(FuzzyEquals(grid.w()(2, 2, 2), 30.0));
  //   . w(i, j, k + 1) = (1-w0)(1-w1)w2w = 3/4*1/4*1/2 * 30.0 = 3/32*30.0
  //     But dividing by weights just yields 30.0
  assert(FuzzyEquals(grid.w()(2, 2, 3), 30.0));
  //   . w(i, j + 1, k) = (1-w0)w1(1-w2)w = 3/4*3/4*1/2 * 30.0 = 9/32*30.0
  //     But dividing by weights just yields 30.0
  assert(FuzzyEquals(grid.w()(2, 3, 2), 30.0));
  //   . w(i, j + 1, k + 1) = (1-w0)w1w2w = 3/4*3/4*1/2 * 30.0 = 9/32*30.0
  //     But dividing by weights just yields 30.0
  assert(FuzzyEquals(grid.w()(2, 3, 3), 30.0));
  //   . w(i + 1, j, k) = w0(1-w1)(1-w2)w = 1/4*1/4*1/2 * 30.0 = 1/32*30.0
  //     But dividing by weights just yields 30.0
  assert(FuzzyEquals(grid.w()(3, 2, 2), 30.0));
  //   . w(i + 1, j, k + 1) = w0(1-w1)w2w = 1/4*1/4*1/2 * 30.0 = 1/32*30.0
  //     But dividing by weights just yields 30.0
  assert(FuzzyEquals(grid.w()(3, 2, 3), 30.0));
  //   . w(i + 1, j + 1, k) = w0w1(1-w2)w = 1/4*3/4*1/2 * 30.0 = 3/32*30.0
  //     But dividing by weights just yields 30.0
  assert(FuzzyEquals(grid.w()(3, 3, 2), 30.0));
  //   . w(i + 1, j + 1, k + 1) = w0w1w2w = 1/4*3/4*1/2 * 30.0 = 3/32*30.0
  //     But dividing by weights just yields 30.0
  assert(FuzzyEquals(grid.w()(3, 3, 3), 30.0));
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz + 1; k++) {
        if (!((i == 2 || i == 3) && (j == 2 || j == 3) && (k == 2 || k == 3))) {
          assert(FuzzyEquals(grid.w()(i, j, k), 0.0));
        }
      }
    }
  }
}

void CheckOneParticlePostSplatMaterial(const StaggeredGrid& grid,
                                       std::size_t nx, std::size_t ny,
                                       std::size_t nz) {
  CheckOuterCellsAreSolid(grid, nx, ny, nz);

  // All interior cell labels should be |EMPTY| except (2, 3, 2) that contains
  // the particle, which should be |FLUID|.
  assert(IsFluid(grid.cell_labels()(2, 3, 2)));
  for (std::size_t i = 1; i < nx - 1; i++) {
    for (std::size_t j = 1; j < ny - 1; j++) {
      for (std::size_t k = 1; k < nz - 1; k++) {
        if (!(i == 2 && j == 3 && k == 2)) {
          assert(IsEmpty(grid.cell_labels()(i, j, k)));
        }
      }
    }
  }
}

void CheckTwoParticlesSplat(const StaggeredGrid& grid, std::size_t nx,
                            std::size_t ny, std::size_t nz) {
  // Particle 2 is at (3.125, 3.125, 2.5).
  //
  // Our grid has lower corner (0, 0, 0) with grid cell width of 1.
  // This particle is in the cell with indices (3, 3, 2).
  // Thus, the particle has weights (0.125, 0.125, 0.5). So its splats will be:
  // - Shift to (3.125, 2.625, 2), which is in cell (3, 2, 2) with weights
  //   (0.125, 0.625, 0). Splat the particle's u velocity, 30.0, onto the grid:
  //   . (i, j, k) = (3, 2, 2)
  //   . (w0, w1, w2) = (0.125, 0.625, 0)
  //   . (1-w0, 1-w1, 1-w2) = (0.875, 0.375, 1)
  //   . u(i, j, k) = u(3, 2, 2) += (1-w0)(1-w1)(1-w2)u = 21/64 * 30.0
  //                              = 9.84375
  //     Added by first particle was
  //     u(i + 1, j, k) = u(3, 2, 2) = w0(1-w1)(1-w2)u = 3/16 * 10.0 = 1.875
  //     Total weights: 21/64 + 3/16 = 0.515625
  //     So u(3, 2, 2) = (9.84375 + 1.875) / 0.515625
  //                   = 22.72727272727272727272727272727272727272727272727
  assert(FuzzyEquals(grid.u()(3, 2, 2),
                     22.72727272727272727272727272727272727272727272727));
  //   . u(i, j, k + 1) = u(3, 2, 3) += (1-w0)(1-w1)w2u = 0.0
  //     Added by first particle was also 0.0
  assert(FuzzyEquals(grid.u()(3, 2, 3), 0.0));
  //   . u(i, j + 1, k) = u(3, 3, 2) += (1-w0)w1(1-w2)u = 35/64 * 30.0
  //                                  = 16.40625
  //     Added by first particle was
  //     u(i + 1, j + 1, k) = u(3, 3, 2) = w0w1(1-w2)u = 9/16 * 10.0 = 5.625
  //     Total weights: 35/64 + 9/16 = 1.109375
  //     So u(3, 3, 2) = (16.40625 + 5.625) / 1.109375
  //                   = 19.85915492957746478873239436619718309859154929577
  assert(FuzzyEquals(grid.u()(3, 3, 2),
                     19.85915492957746478873239436619718309859154929577));
  //   . u(i, j + 1, k + 1) = u(3, 3, 3) += (1-w0)w1w2u = 0.0
  //     Added by first particle was also 0.0
  assert(FuzzyEquals(grid.u()(3, 3, 3), 0.0));
  //   . u(i + 1, j, k) = u(4, 2, 2) = w0(1-w1)(1-w2)u = 3/64 * 30.0 = 1.40625
  //     Dividing by weight -> 30.0
  assert(FuzzyEquals(grid.u()(4, 2, 2), 30.0));
  //   . u(i + 1, j, k + 1) = u(4, 2, 3) = w0(1-w1)w2u = 0.0
  assert(FuzzyEquals(grid.u()(4, 2, 3), 0.0));
  //   . u(i + 1, j + 1, k) = u(4, 3, 2) = w0w1(1-w2)u = 5/64 * 30.0 = 2.34375
  //     Dividing by weight -> 30.0
  assert(FuzzyEquals(grid.u()(4, 3, 2), 30.0));
  //   . u(i + 1, j + 1, k + 1) = u(4, 3, 3) = w0w1w2u = 0.0
  assert(FuzzyEquals(grid.u()(4, 3, 3), 0.0));
  // Check first particle-only horizontal grid velocities.
  assert(FuzzyEquals(grid.u()(2, 2, 2), 10.0));
  assert(FuzzyEquals(grid.u()(2, 2, 3), 0.0));
  assert(FuzzyEquals(grid.u()(2, 3, 2), 10.0));
  assert(FuzzyEquals(grid.u()(2, 3, 3), 0.0));
  for (std::size_t i = 0; i < nx + 1; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        if (!((i == 2 || i == 3) && (j == 2 || j == 3) && (k == 2 || k == 3)) &&
            !(i == 4 && (j == 2 || j == 3) && (k == 2 || k == 3))) {
          assert(FuzzyEquals(grid.u()(i, j, k), 0.0));
        }
      }
    }
  }
  // - Shift to (2.625, 3.125, 2), which is in cell (2, 3, 2) with weights
  //   (0.625, 0.125, 0). Splat the particle's v velocity, 10.0, onto the grid:
  //   . (i, j, k) = (2, 3, 2)
  //   . (w0, w1, w2) = (0.625, 0.125, 0)
  //   . (1-w0, 1-w1, 1-w2) = (0.375, 0.875, 1)
  //   . v(i, j, k) = v(2, 3, 2) += (1-w0)(1-w1)(1-w2)v = 21/64 * 10.0
  //                              = 3.28125
  //     First particle added
  //     v(i, j, k) = (1-w0)(1-w1)(1-w2)v = 9/16 * 20.0 = 9/4 * 5.0 = 11.25
  //     But dividing by weights just yields 20.0
  //     Total weights: 21/64 + 9/16
  //     So v(2, 3, 2) = (3.28125 + 11.25) / (21/64 + 9/16)
  //                   = 16.31578947368421052631578947368421052631578947368
  assert(FuzzyEquals(grid.v()(2, 3, 2),
                     16.31578947368421052631578947368421052631578947368));
  //   . v(i, j, k + 1) = v(2, 3, 3) += (1-w0)(1-w1)w2v = 0.0
  //     First particle also contributed 0.0
  assert(FuzzyEquals(grid.v()(2, 3, 3), 0.0));
  //   . v(i, j + 1, k) = v(2, 4, 2) += (1-w0)w1(1-w2)v = 3/64 * 10.0 = 0.46875
  //     First particle added
  //   . v(i, j + 1, k) = v(2, 4, 2) = (1-w0)w1(1-w2)v = 3/16 * 20.0 = 3.75
  //     Total weights: 3/64 + 3/16
  //     So v(2, 4, 2) = (0.46875 + 3.75) / (3/64 + 3/16) = 18.0
  assert(FuzzyEquals(grid.v()(2, 4, 2), 18.0));
  //   . v(i, j + 1, k + 1) = v(2, 4, 3) = (1-w0)w1w2v = 0.0
  //     First particle also contributed 0.0
  assert(FuzzyEquals(grid.v()(2, 4, 3), 0.0));
  //   . v(i + 1, j, k) = v(3, 3, 2) += w0(1-w1)(1-w2)v = 35/64 * 10.0 = 5.46875
  //     First particle added
  //     v(i + 1, j, k) = v(3, 3, 2) = w0(1-w1)(1-w2)v = 3/16 * 20.0 = 3.75
  //     Total weights: 35/64 + 3/16
  //     So v(3, 3, 2) = (5.46875 + 3.75) / (35/64 + 3/16)
  //                   = 12.55319148936170212765957446808510638297872340426
  assert(FuzzyEquals(grid.v()(3, 3, 2),
                     12.55319148936170212765957446808510638297872340426));
  //   . v(i + 1, j, k + 1) = v(3, 3, 3) += w0(1-w1)w2v = 0.0
  //     First particle also contributed 0.0
  assert(FuzzyEquals(grid.v()(3, 3, 3), 0.0));
  //   . v(i + 1, j + 1, k) = v(3, 4, 2) += w0w1(1-w2)v = 5/64 * 10.0 = 0.78125
  //     First particle added
  //     v(i + 1, j + 1, k) = v(3, 4, 2) = w0w1(1-w2)v = 1/16 * 20.0 = 1.25
  //     Total weights: 5/64 + 1/16
  //     So v(3, 4, 2) = (0.78125 + 1.25) / (5/64 + 1/16)
  //                   = 14.44444444444444444444444444444444444444444444444
  assert(FuzzyEquals(grid.v()(3, 4, 2),
                     14.44444444444444444444444444444444444444444444444));
  //   . v(i + 1, j + 1, k + 1) = v(3, 4, 3) = w0w1w2v = 0.0
  //     First particle also contributed 0.0
  assert(FuzzyEquals(grid.v()(3, 4, 3), 0.0));
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny + 1; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        if (!((i == 2 || i == 3) && (j == 3 || j == 4) && (k == 2 || k == 3))) {
          assert(FuzzyEquals(grid.v()(i, j, k), 0.0));
        }
      }
    }
  }
  // - Shift to (2.625, 2.625, 2.5), which is in cell (2, 2, 2) with weights
  //   (0.625, 0.625, 0.5). Splat the particle's w velocity, 20.0, onto the
  //   grid:
  //   . (i, j, k) = (2, 2, 2)
  //   . (w0, w1, w2) = (0.625, 0.625, 0.5)
  //   . (1-w0, 1-w1, 1-w2) = (0.375, 0.375, 0.5)
  //   . w(i, j, k) = w(2, 2, 2) += (1-w0)(1-w1)(1-w2)w = 9/128 * 20.0 = 1.40625
  //     First particle added
  //     w(i, j, k) = w(2, 2, 2) = (1-w0)(1-w1)(1-w2)w = 3/32*30.0 = 2.8125
  //     Total weights: 9/128 + 3/32
  //     So w(2, 2, 2) = (1.40625 + 2.8125) / (9/128 + 3/32)
  //                   = 25.71428571428571428571428571428571428571428571429
  assert(FuzzyEquals(grid.w()(2, 2, 2),
                     25.71428571428571428571428571428571428571428571429));
  //   . w(i, j, k + 1) = w(2, 2, 3) += (1-w0)(1-w1)w2w = 9/128 * 20.0 = 1.40625
  //     First particle added
  //     w(i, j, k + 1) = w(2, 2, 3) = (1-w0)(1-w1)w2w = 3/32*30.0
  //     Total weights: 9/128 + 3/32
  //     So w(2, 2, 3) = same as previous
  assert(FuzzyEquals(grid.w()(2, 2, 3),
                     25.71428571428571428571428571428571428571428571429));
  //   . w(i, j + 1, k) = w(2, 3, 2) += (1-w0)w1(1-w2)w = 15/128 * 20.0
  //     First particle added
  //     w(i, j + 1, k) = w(2, 3, 2) = (1-w0)w1(1-w2)w = 9/32*30.0
  //     Total weights: 15/128 + 9/32
  //     So w(2, 3, 2) = (15/128*20.0 + 9/32*30.0) / (15/128 + 9/32)
  //                   = 27.05882352941176470588235294117647058823529411765
  assert(FuzzyEquals(grid.w()(2, 3, 2),
                     27.05882352941176470588235294117647058823529411765));
  //   . w(i, j + 1, k + 1) = w(2, 3, 3) += (1-w0)w1w2w = 15/128 * 20.0
  //     First particle added
  //     w(i, j + 1, k + 1) = w(2, 3, 3) = (1-w0)w1w2w = 9/32*30.0
  //     Total weights: 15/128 + 9/32
  //     So w(2, 3, 3) = same as previous
  assert(FuzzyEquals(grid.w()(2, 3, 3),
                     27.05882352941176470588235294117647058823529411765));
  //   . w(i + 1, j, k) = w(3, 2, 2) += w0(1-w1)(1-w2)w = 15/128 * 20.0
  //     First particle added
  //     w(i + 1, j, k) = w(3, 2, 2) = w0(1-w1)(1-w2)w = 1/32*30.0
  //     Total weights: 15/128 + 1/32
  //     So w(3, 2, 2) = (15/128 * 20.0 + 1/32*30.0) / (15/128 + 1/32)
  //                   = 22.10526315789473684210526315789473684210526315789
  assert(FuzzyEquals(grid.w()(3, 2, 2),
                     22.10526315789473684210526315789473684210526315789));
  //   . w(i + 1, j, k + 1) = w(3, 2, 3) += w0(1-w1)w2w = 15/128 * 20.0
  //     First particle added
  //     w(i + 1, j, k + 1) = w(3, 2, 3) = w0(1-w1)w2w = 1/32*30.0
  //     Total weights: 15/128 + 1/32
  //     So w(3, 2, 3) = same as previous
  assert(FuzzyEquals(grid.w()(3, 2, 3),
                     22.10526315789473684210526315789473684210526315789));
  //   . w(i + 1, j + 1, k) = w(3, 3, 2) += w0w1(1-w2)w = 25/128 * 20.0
  //     First particle added
  //     w(i + 1, j + 1, k) = w(3, 3, 2) += w0w1(1-w2)w = 3/32*30.0
  //     Total weights: 25/128 + 3/32
  //     So w(3, 3, 2) = (25/128 * 20.0 + 3/32*30.0) / (25/128 + 3/32)
  //                   = 23.24324324324324324324324324324324324324324324324
  assert(FuzzyEquals(grid.w()(3, 3, 2),
                     23.24324324324324324324324324324324324324324324324));
  //   . w(i + 1, j + 1, k + 1) = w(3, 3, 3) += w0w1w2w = 25/128 * 20.0
  //     First particle added
  //     w(i + 1, j + 1, k + 1) = w(3, 3, 3) += w0w1w2w = 3/32*30.0
  //     Total weights: 25/128 + 3/32
  //     So w(3, 3, 3) = same as previous
  assert(FuzzyEquals(grid.w()(3, 3, 3),
                     23.24324324324324324324324324324324324324324324324));
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz + 1; k++) {
        if (!((i == 2 || i == 3) && (j == 2 || j == 3) && (k == 2 || k == 3))) {
          assert(FuzzyEquals(grid.w()(i, j, k), 0.0));
        }
      }
    }
  }
}

void CheckTwoParticlePostSplatMaterial(const StaggeredGrid& grid,
                                       std::size_t nx, std::size_t ny,
                                       std::size_t nz) {
  CheckOuterCellsAreSolid(grid, nx, ny, nz);

  // All interior cell labels should be |EMPTY| except (2, 3, 2) that contains
  // the first particle and (3, 3, 2) that contains the second particle; both of
  // those cells should be |FLUID|.
  assert(IsFluid(grid.cell_labels()(2, 3, 2)));
  assert(IsFluid(grid.cell_labels()(3, 3, 2)));
  for (std::size_t i = 1; i < nx - 1; i++) {
    for (std::size_t j = 1; j < ny - 1; j++) {
      for (std::size_t k = 1; k < nz - 1; k++) {
        if (!((i == 2 || i == 3) && j == 3 && k == 2)) {
          assert(IsEmpty(grid.cell_labels()(i, j, k)));
        }
      }
    }
  }
}

void CheckAdvection(const std::vector<Particle>& particles) {
  // Particle 1 is at (x, y, z) = (2.75, 3.25, 2.5) --> cell (2, 3, 2).
  // - Horizontal velocity interpolation:
  //   . Shift to (2.75, 2.75, 2) --> cell (2, 2, 2).
  //   . Barycentric weights: (w0 = 0.75, w1 = 0.75, w2 = 0)
  //     So 1-w0 = 0.25, 1-w1 = 0.25, 1-w2 = 1
  //   . New particle position = x + dt * u_interp, where
  //     u_interp = (1-w0)(1-w1)(1-w2) * u(2, 2, 2) +
  //                (1-w0)(1-w1)w2 * u(2, 2, 2 + 1) +
  //                (1-w0)w1(1-w2) * u(2, 2 + 1, 2) +
  //                (1-w0)w1w2 * u(2, 2 + 1, 2 + 1) +
  //                w0(1-w1)(1-w2) * u(2 + 1, 2, 2) +
  //                w0(1-w1)w2 * u(2 + 1, 2, 2 + 1) +
  //                w0w1(1-w2) * u(2 + 1, 2 + 1, 2) +
  //                w0w1w2 * u(2 + 1, 2 + 1, 2 + 1)
  //              = 1/16 * u(2, 2, 2) + 3/16 * u(2, 3, 2) +
  //                3/16 * u(3, 2, 2) + 9/16 * u(3, 3, 2)
  //     From above:
  //     - u(2, 2, 2) = 10, u(2, 3, 2) = 10,
  //       u(3, 2, 2) = 22.72727272727272727272727272727272727272727272727,
  //       u(3, 3, 2) = 19.85915492957746478873239436619718309859154929577
  //     - So, u_interp = 17.93213828425096030729833546734955185659411011523
  //     From the input file, dt = 0.001111112
  //     So, the particle's new x position is x + dt * u_interp, which is:
  //     2.76992461403329065300896286811779769526248399488
  assert(FuzzyEquals(particles[0].pos[0],
                     2.76992461403329065300896286811779769526248399488));
  // - Vertical velocity interpolation:
  //   . Shift to (2.25, 3.25, 2) --> cell (2, 3, 2).
  //   . Barycentric weights: (w0 = 0.25, w1 = 0.25, w2 = 0)
  //     So 1-w0 = 0.75, 1-w1 = 0.75, 1-w2 = 1
  //   . New particle position = y + dt * v_interp, where
  //     v_interp = (1-w0)(1-w1)(1-w2) * v(2, 3, 2) +
  //                (1-w0)(1-w1)w2 * v(2, 3, 2 + 1) +
  //                (1-w0)w1(1-w2) * v(2, 3 + 1, 2) +
  //                (1-w0)w1w2 * v(2, 3 + 1, 2 + 1) +
  //                w0(1-w1)(1-w2) * v(2 + 1, 3, 2) +
  //                w0(1-w1)w2 * v(2 + 1, 3, 2 + 1) +
  //                w0w1(1-w2) * v(2 + 1, 3 + 1, 2) +
  //                w0w1w2 * v(2 + 1, 3 + 1, 2 + 1)
  //              = 9/16 * v(2, 3, 2) + 3/16 * v(2, 4, 2) +
  //                3/16 * v(3, 3, 2) + 1/16 * v(3, 4, 2)
  //     From above:
  //     - v(2, 3, 2) = 16.31578947368421052631578947368421052631578947368,
  //       v(2, 4, 2) = 18,
  //       v(3, 3, 2) = 12.55319148936170212765957446808510638297872340426,
  //       v(3, 4, 2) = 14.44444444444444444444444444444444444444444444444
  //     - So, v_interp = 15.80913276098046534776657956949110364563891999502
  //     So, the particle's new y position is y + dt * v_interp, which is:
  //     3.26756571712031852681348761975861639915391315167
  assert(FuzzyEquals(particles[0].pos[1],
                     3.26756571712031852681348761975861639915391315167));
  // - Depth velocity interpolation:
  //   . Shift to (2.25, 2.75, 2.5) --> cell (2, 2, 2).
  //   . Barycentric weights: (w0 = 0.25, w1 = 0.75, w2 = 0.5)
  //     So 1-w0 = 0.75, 1-w1 = 0.25, 1-w2 = 0.5
  //   . New particle position = z + dt * w_interp, where
  //     w_interp = (1-w0)(1-w1)(1-w2) * w(2, 2, 2) +
  //                (1-w0)(1-w1)w2 * w(2, 2, 2 + 1) +
  //                (1-w0)w1(1-w2) * w(2, 2 + 1, 2) +
  //                (1-w0)w1w2 * w(2, 2 + 1, 2 + 1) +
  //                w0(1-w1)(1-w2) * w(2 + 1, 2, 2) +
  //                w0(1-w1)w2 * w(2 + 1, 2, 2 + 1) +
  //                w0w1(1-w2) * w(2 + 1, 2 + 1, 2) +
  //                w0w1w2 * w(2 + 1, 2 + 1, 2 + 1)
  //              = 3/32 * w(2, 2, 2) + 3/32 * w(2, 2, 3) +
  //                9/32 * w(2, 3, 2) + 9/32 * w(2, 3, 3) +
  //                1/32 * w(3, 2, 2) + 1/32 * w(3, 2, 3) +
  //                3/32 * w(3, 3, 2) + 3/32 * w(3, 3, 3)
  //     From above:
  //     - w(2, 2, 2) = 25.71428571428571428571428571428571428571428571429
  //       w(2, 2, 3) = 25.71428571428571428571428571428571428571428571429
  //       w(2, 3, 2) = 27.05882352941176470588235294117647058823529411765
  //       w(2, 3, 3) = 27.05882352941176470588235294117647058823529411765
  //       w(3, 2, 2) = 22.10526315789473684210526315789473684210526315789
  //       w(3, 2, 3) = 22.10526315789473684210526315789473684210526315789
  //       w(3, 3, 2) = 23.24324324324324324324324324324324324324324324324
  //       w(3, 3, 3) = 23.24324324324324324324324324324324324324324324324
  //     - So, w_interp = 25.78170386219921823636993915631686529519346856808
  //     So, the particle's new z position is z + dt * w_interp, which is:
  //     2.52864636054173589777304947583585354483187300525
  assert(FuzzyEquals(particles[0].pos[2],
                     2.52864636054173589777304947583585354483187300525));

  // Particle 2 is at (x, y, z) = (3.125, 3.125, 2.5) --> cell (3, 3, 2).
  // - Horizontal velocity interpolation:
  //   . Shift to (3.125, 2.625, 2) --> cell (3, 2, 2).
  //   . Barycentric weights: (w0 = 0.125, w1 = 0.625, w2 = 0)
  //     So 1-w0 = 0.875, 1-w1 = 0.375, 1-w2 = 1
  //     w0 = 1/8, w1 = 5/8, 1-w0 = 7/8, 1-w1 = 3/8
  //   . New particle position = x + dt * u_interp, where
  //     u_interp = (1-w0)(1-w1)(1-w2) * u(3, 2, 2) +
  //                (1-w0)(1-w1)w2 * u(3, 2, 2 + 1) +
  //                (1-w0)w1(1-w2) * u(3, 2 + 1, 2) +
  //                (1-w0)w1w2 * u(3, 2 + 1, 2 + 1) +
  //                w0(1-w1)(1-w2) * u(3 + 1, 2, 2) +
  //                w0(1-w1)w2 * u(3 + 1, 2, 2 + 1) +
  //                w0w1(1-w2) * u(3 + 1, 2 + 1, 2) +
  //                w0w1w2 * u(3 + 1, 2 + 1, 2 + 1)
  //              = 21/64 * u(3, 2, 2) + 35/64 * u(3, 3, 2) +
  //                3/64 * u(4, 2, 2) + 5/64 * u(4, 3, 2)
  //     From above:
  //     - u(3, 2, 2) = 22.72727272727272727272727272727272727272727272727,
  //       u(3, 3, 2) = 19.85915492957746478873239436619718309859154929577,
  //       u(4, 2, 2) = 30, u(4, 3, 2) = 30
  //     - So, u_interp = 21/64 * u(3, 2, 2) + 35/64 * u(3, 3, 2) + 3/64 * 30 +
  //                      5/64 * 30
  //                    = 22.06786171574903969270166453265044814340588988476
  //     So, the particle's new x position is x + dt * u_interp, which is:
  //     3.14951986596670934699103713188220230473751600512
  assert(FuzzyEquals(particles[1].pos[0],
                     3.14951986596670934699103713188220230473751600512));
  // - Vertical velocity interpolation:
  //   . Shift to (2.625, 3.125, 2) --> cell (2, 3, 2).
  //   . Barycentric weights: (w0 = 0.625, w1 = 0.125, w2 = 0)
  //     So 1-w0 = 0.375, 1-w1 = 0.875, 1-w2 = 1
  //   . New particle position = y + dt * v_interp, where
  //     v_interp = (1-w0)(1-w1)(1-w2) * v(2, 3, 2) +
  //                (1-w0)(1-w1)w2 * v(2, 3, 2 + 1) +
  //                (1-w0)w1(1-w2) * v(2, 3 + 1, 2) +
  //                (1-w0)w1w2 * v(2, 3 + 1, 2 + 1) +
  //                w0(1-w1)(1-w2) * v(2 + 1, 3, 2) +
  //                w0(1-w1)w2 * v(2 + 1, 3, 2 + 1) +
  //                w0w1(1-w2) * v(2 + 1, 3 + 1, 2) +
  //                w0w1w2 * v(2 + 1, 3 + 1, 2 + 1)
  //              = 21/64 * v(2, 3, 2) + 3/64 * v(2, 4, 2) +
  //                35/64 * v(3, 3, 2) + 5/64 * v(3, 4, 2)
  //     From above:
  //     - v(2, 3, 2) = 16.31578947368421052631578947368421052631578947368,
  //       v(2, 4, 2) = 18,
  //       v(3, 3, 2) = 12.55319148936170212765957446808510638297872340426,
  //       v(3, 4, 2) = 14.44444444444444444444444444444444444444444444444
  //     - So, v_interp = 14.19086723901953465223342043050889635436108000498
  //     So, the particle's new y position is y + dt * v_interp, which is:
  //     3.14076764287968147318651238024138360084608684833
  assert(FuzzyEquals(particles[1].pos[1],
                     3.14076764287968147318651238024138360084608684833));
  // - Depth velocity interpolation:
  //   . Shift to (2.625, 2.625, 2.5) --> cell (2, 2, 2).
  //   . Barycentric weights: (w0 = 0.625, w1 = 0.625, w2 = 0.5)
  //     So 1-w0 = 0.375, 1-w1 = 0.375, 1-w2 = 0.5
  //   . New particle position = z + dt * w_interp, where
  //     w_interp = (1-w0)(1-w1)(1-w2) * w(2, 2, 2) +
  //                (1-w0)(1-w1)w2 * w(2, 2, 2 + 1) +
  //                (1-w0)w1(1-w2) * w(2, 2 + 1, 2) +
  //                (1-w0)w1w2 * w(2, 2 + 1, 2 + 1) +
  //                w0(1-w1)(1-w2) * w(2 + 1, 2, 2) +
  //                w0(1-w1)w2 * w(2 + 1, 2, 2 + 1) +
  //                w0w1(1-w2) * w(2 + 1, 2 + 1, 2) +
  //                w0w1w2 * w(2 + 1, 2 + 1, 2 + 1)
  //              = 9/128 * w(2, 2, 2) + 9/128 * w(2, 2, 3) +
  //                15/128 * w(2, 3, 2) + 15/128 * w(2, 3, 3) +
  //                15/128 * w(3, 2, 2) + 15/128 * w(3, 2, 3) +
  //                25/128 * w(3, 3, 2) + 25/128 * w(3, 3, 3)
  //     From above:
  //     - w(2, 2, 2) = 25.71428571428571428571428571428571428571428571429
  //       w(2, 2, 3) = 25.71428571428571428571428571428571428571428571429
  //       w(2, 3, 2) = 27.05882352941176470588235294117647058823529411765
  //       w(2, 3, 3) = 27.05882352941176470588235294117647058823529411765
  //       w(3, 2, 2) = 22.10526315789473684210526315789473684210526315789
  //       w(3, 2, 3) = 22.10526315789473684210526315789473684210526315789
  //       w(3, 3, 2) = 23.24324324324324324324324324324324324324324324324
  //       w(3, 3, 3) = 23.24324324324324324324324324324324324324324324324
  //     - So, w_interp = 24.21829613780078176363006084368313470480653143192
  //     So, the particle's new z position is z + dt * w_interp, which is:
  //     2.52690923945826410222695052416414645516812699475
  assert(FuzzyEquals(particles[1].pos[2],
                     2.52690923945826410222695052416414645516812699475));
}

}  // namespace

// Test the public interface of StaggeredGrid.
int main(int argc, char** argv) {
  SimulationParameters params = ReadSimulationParameters(argc, argv);

  std::size_t nx = 9, ny = 7, nz = 5;
  Eigen::Vector3d lower_corner(0.0, 0.0, 0.0);
  double dx = 1.0;
  StaggeredGrid grid(nx, ny, nz, lower_corner, dx);

  // Ensure the grid's arrays have the correct dimensions.
  assert(grid.p().nx() == nx);
  assert(grid.p().ny() == ny);
  assert(grid.p().nz() == nz);

  assert(grid.u().nx() == nx + 1);
  assert(grid.u().ny() == ny);
  assert(grid.u().nz() == nz);

  assert(grid.v().nx() == nx);
  assert(grid.v().ny() == ny + 1);
  assert(grid.v().nz() == nz);

  assert(grid.w().nx() == nx);
  assert(grid.w().ny() == ny);
  assert(grid.w().nz() == nz + 1);

  assert(grid.cell_labels().nx() == nx);
  assert(grid.cell_labels().ny() == ny);
  assert(grid.cell_labels().nz() == nz);

  std::vector<Particle> particles;

  // No particles!
  grid.ParticlesToGrid(particles);
  CheckEmptyParticleSplat(grid, nx, ny, nz);

  // One particle!
  particles.push_back(MakeParticle(2.75, 3.25, 2.5, 10.0, 20.0, 30.0));
  grid.ParticlesToGrid(particles);
  CheckOneParticleSplat(grid, nx, ny, nz);
  CheckOneParticlePostSplatMaterial(grid, nx, ny, nz);

  // Two particles
  particles.push_back(MakeParticle(3.125, 3.125, 2.5, 30.0, 10.0, 20.0));
  grid.ParticlesToGrid(particles);
  CheckTwoParticlesSplat(grid, nx, ny, nz);
  CheckTwoParticlePostSplatMaterial(grid, nx, ny, nz);

  // Advect particles
  for (std::vector<Particle>::iterator p = particles.begin();
       p != particles.end(); p++) {
    p->pos = grid.Advect(p->pos, params.dt_seconds());
  }
  CheckAdvection(particles);

  // If nothing crashed up until this point, everything worked correctly!
  std::cout << "All StaggeredGrid assertion tests passed!" << std::endl;

  return EXIT_SUCCESS;
}
