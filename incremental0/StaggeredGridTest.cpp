#include <iostream>

#include "StaggeredGrid.h"

void PrintArrayDimensions(char symbol, const Array3D<double>& arr3d) {
  std::cout << "- " << symbol << " array is ";
  std::cout << arr3d.nx() << " x ";
  std::cout << arr3d.ny() << " x ";
  std::cout << arr3d.nz() << std::endl;
}

int main(int argc, char** argv) {
  StaggeredGrid grid(3, 4, 5);
  std::cout << "Created StaggeredGrid:" << std::endl;

  PrintArrayDimensions('p', grid.p());
  PrintArrayDimensions('u', grid.u());
  PrintArrayDimensions('v', grid.v());
  PrintArrayDimensions('w', grid.w());

  return 0;
}
