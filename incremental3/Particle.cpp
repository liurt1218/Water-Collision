#include "Particle.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace {

std::string ReadLine(std::ifstream* in) {
  std::string line;
  std::getline(*in, line);
  return line;
}

int ReadNumParticles(std::ifstream* in) {
  int num_particles = -1;

  std::string line = ReadLine(in);
  std::istringstream ss(line);
  if (!(ss >> num_particles)) {
    return -1;
  }

  return num_particles;
}

Eigen::Vector3d ReadVector3d(std::istringstream* line_ss) {
  Eigen::Vector3d vec;
  (*line_ss) >> vec[0] >> vec[1] >> vec[2];
  return vec;
}

Particle ReadParticle(const std::string& line) {
  std::istringstream ss(line);
  return Particle{.pos = ReadVector3d(&ss), .vel = ReadVector3d(&ss)};
}

}  // namespace

std::vector<Particle> ReadParticles(const std::string& input_file) {
  std::ifstream in(input_file.c_str(), std::ios::in);

  int alleged_num_particles = ReadNumParticles(&in);

  std::vector<Particle> particles;

  std::string line;
  while (std::getline(in, line)) {
    particles.push_back(ReadParticle(line));
  }
  assert(particles.size() == alleged_num_particles);
  std::cout << "Read " << particles.size() << " particles." << std::endl;

  in.close();

  return particles;
}
