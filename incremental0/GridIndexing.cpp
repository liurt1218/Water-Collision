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

// Prints the provided |indices|.
void Print(const GridIndices& indices) {
  std::cout << "Indices: " << std::endl;
  std::cout << indices << std::endl;
}

int main(int argc, char** argv) {
  // Make a grid with spacing of 2 with lower corner (1, 2, 3).
  double dx = 2.0;
  Eigen::Vector3d lc(1, 2, 3);

  // (i, j, k) should be (floor((6-1)/2), floor((4-2)/2), floor((3-3)/2)),
  // which is (floor(2.5), floor(1), floor(0)) = (2, 1, 0).
  Eigen::Vector3d p1(6, 4, 3);
  GridIndices p1_indices = floor(p1, lc, dx);
  Print(p1_indices);

  // (i, j, k) should be (3, 8, 9).
  Eigen::Vector3d p2(7, 18, 22);
  GridIndices p2_indices = floor(p2, lc, dx);
  Print(p2_indices);

  // Should lead to an assertion failure for index 0 (x-coordinate) being
  // less than the x-coordinate of |lc|. The program should crash before
  // it even gets to the Print command below.
  Eigen::Vector3d p3(-1, 2, 3);
  GridIndices p3_indices = floor(p3, lc, dx);
  Print(p3_indices);

  return 0;
}
