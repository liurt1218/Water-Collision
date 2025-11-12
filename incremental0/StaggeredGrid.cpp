#include "StaggeredGrid.h"

StaggeredGrid::StaggeredGrid(std::size_t nx, std::size_t ny, std::size_t nz)
    : nx_(nx),
      ny_(ny),
      nz_(nz),
      ny_nz_(ny * nz),
      p_(nx, ny, nz),
      u_(nx + 1, ny, nz),
      v_(nx, ny + 1, nz),
      w_(nx, ny, nz + 1) {}

StaggeredGrid::~StaggeredGrid() {}
