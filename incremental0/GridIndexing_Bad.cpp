#include <Eigen/Dense>
#include <iostream>

// Set the parameters |i|, |j|, and |k| to be the indices of the grid cell
// containing the point |p| for a grid with lower corner |lc| and grid cell
// width (spacing) |dx|.
inline void floor(const Eigen::Vector3d& p, const Eigen::Vector3d& lc,
                  double dx, int& i, int& j, int& k) {
  // Compute |p|'s location relative to |lc|.
  // Dividing by |dx| yields a 3D vector indicating the number of grid
  // cells (including fractions of grid cells, as the vector elements are
  // floating-point values) away from |lc| that |p| is located.
  Eigen::Vector3d p_lc_over_dx = (p - lc) / dx;

  // Set the indices to the floor of |p_lc_over_dx|'s elements.
  i = static_cast<int>(p_lc_over_dx[0]);
  j = static_cast<int>(p_lc_over_dx[1]);
  k = static_cast<int>(p_lc_over_dx[2]);
}

int main(int argc, char** argv) {
  // Make a grid with spacing of 2 with lower corner (1, 2, 3).
  double dx = 2.0;
  Eigen::Vector3d lc(1, 2, 3);

  // (i, j, k) should be (floor((6-1)/2), floor((4-2)/2), floor((3-3)/2)),
  // which is (floor(2.5), floor(1), floor(0)) = (2, 1, 0).
  Eigen::Vector3d p1(6, 4, 3);
  int i, j, k;
  floor(p1, lc, dx, i, j, k);
  std::cout << "i = " << i << ", j = " << j << ", k = " << k << std::endl;

  // (i, j, k) should be (3, 8, 9).
  Eigen::Vector3d p2(7, 18, 22);
  floor(p2, lc, dx, i, j, k);
  std::cout << "i = " << i << ", j = " << j << ", k = " << k << std::endl;

  // (i, j, k) should be (-1, 0, 0)--wait, negative indices aren't
  // allowed, right?!
  Eigen::Vector3d p3(-1, 2, 3);
  floor(p3, lc, dx, i, j, k);
  std::cout << "i = " << i << ", j = " << j << ", k = " << k << std::endl;

  return 0;
}
