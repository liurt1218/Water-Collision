#include <Eigen/Dense>
#include <cassert>
#include <iostream>

// To disable assert*() calls, uncomment this line:
// #define NDEBUG

// This represents a triplet of indices as a column vector of nonnegative
// integers:
// [ i ]
// [ j ]
// [ k ]
typedef Eigen::Matrix<std::size_t, 3, 1> GridIndices;
//
// Returns the indices of the grid cell containing the point |p| for a grid with
// lower corner |lc| and grid cell width (spacing) |dx|.
//
// If |dx| <= 0 or |p|'s location relative to |lc| would result in negative
// indices being returned and assertions are on, then assertion failures will
// crash this program.
inline GridIndices floor(const Eigen::Vector3d& p, const Eigen::Vector3d& lc,
                         double dx) {
  // Ensure grid spacings are positive.
  assert(dx > 0.0);

  // Compute |p|'s location relative to |lc|.
  // Dividing by |dx| yields a 3D vector indicating the number of grid
  // cells (including fractions of grid cells, as the vector elements are
  // floating-point values) away from |lc| that |p| is located.
  Eigen::Vector3d p_lc_over_dx = (p - lc) / dx;

  // Ensure we won't end up with negative indices.
  assert(p_lc_over_dx[0] >= 0.0);
  assert(p_lc_over_dx[1] >= 0.0);
  assert(p_lc_over_dx[2] >= 0.0);

  // Indices are valid. Construct and return them.
  // This casts the elements of the vector above as nonnegative integers.
  return p_lc_over_dx.cast<std::size_t>();
}

void print(const GridIndices& indices) {
  std::cout << "Indices: " << std::endl;
  std::cout << indices << std::endl;
}

inline Eigen::Vector3d weights(const Eigen::Vector3d& p,
                               const Eigen::Vector3d& lc, double dx,
                               const GridIndices& indices) {
  return (p - lc) / dx - indices.cast<double>();
}

void PrintPointIndicesAndWeights(const Eigen::Vector3d& p,
                                 const Eigen::Vector3d& lc, double dx) {
  GridIndices p_indices = floor(p, lc, dx);
  print(p_indices);
  Eigen::Vector3d w = weights(p, lc, dx, p_indices);
  std::cout << "Weights = " << std::endl;
  std::cout << w << std::endl << std::endl;
}

int main(int argc, char** argv) {
  // Let's have our grid's lower corner be at (3, 4, 5) with a grid spacing of
  // 2. So, each grid cell will be a 2 x 2 x 2 cube.
  Eigen::Vector3d lc(3, 4, 5);
  double dx = 2.0;

  // Expect (i, j, k) == (0, 3, 4), (w0, w1, w2) == (0.5, 0, 0.5)
  Eigen::Vector3d p1(4, 10, 14);
  PrintPointIndicesAndWeights(p1, lc, dx);

  // Expect (i, j, k) == (3, 4, 16), (w0, w1, w2) == (0.5, 0.5, 0)
  Eigen::Vector3d p2(10, 13, 37);
  PrintPointIndicesAndWeights(p2, lc, dx);

  // Expect (i, j, k) == (0, 0, 0), (w0, w1, w2) == (0, 0, 0)
  Eigen::Vector3d p3(3, 4, 5);
  PrintPointIndicesAndWeights(p3, lc, dx);
  return 0;
}
