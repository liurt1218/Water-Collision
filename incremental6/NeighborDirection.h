#ifndef NEIGHBOR_DIRECTION_H_
#define NEIGHBOR_DIRECTION_H_

enum NeighborDirection {
  LEFT = 8,
  DOWN = 16,
  BACK = 32,
  RIGHT = 64,
  UP = 128,
  FORWARD = 256
};

const NeighborDirection kNeighborDirections[] = {LEFT,  DOWN, BACK,
                                                 RIGHT, UP,   FORWARD};

#endif  // NEIGHBOR_DIRECTION_H_
