#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <Eigen/Dense>
#include <vector>

struct Particle {
  Eigen::Vector3d pos;
  Eigen::Vector3d vel;
};

// Returns an array of Particles read from the file with relative path
// |input_file|.
std::vector<Particle> ReadParticles(const std::string& input_file);

#endif  // PARTICLE_H_
