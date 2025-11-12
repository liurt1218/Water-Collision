#include <iostream>
#include <vector>

#include "Particle.h"
#include "SimulationParameters.h"

// This is a test program for now, but we'll build up to having this be the main
// driver program for our simulator. As such, some test functions like these are
// only intended to be in incomplete versions of this driver program, like this
// version.

namespace {

void PrintAsRow(const Eigen::Vector3d& column_vector) {
  std::cout << column_vector.transpose();
}

void PrintSimulationParameters(const SimulationParameters& params) {
  std::cout << std::endl;
  std::cout << "dt = " << params.dt_seconds() << " seconds" << std::endl;
  std::cout << "duration = " << params.duration_seconds() << " seconds"
            << std::endl;
  std::cout << "density = " << params.density() << std::endl;
  std::cout << "Grid has " << params.nx() << " x " << params.ny() << " x "
            << params.nz() << " grid cells." << std::endl;
  std::cout << "lower corner is ";
  PrintAsRow(params.lc());
  std::cout << std::endl;
  std::cout << "dx = " << params.dx() << std::endl;
  std::cout << "flip ratio = " << params.flip_ratio() << std::endl;
  std::cout << "output file name pattern = "
            << params.output_file_name_pattern() << std::endl;
}

void PrintParticles(const std::vector<Particle>& particles) {
  std::cout << std::endl;
  std::cout << particles.size() << " particles:" << std::endl;
  for (Particle p : particles) {
    std::cout << "pos = ";
    PrintAsRow(p.pos);
    std::cout << ", vel = ";
    PrintAsRow(p.vel);
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  SimulationParameters params = ReadSimulationParameters(argc, argv);
  PrintSimulationParameters(params);

  std::vector<Particle> particles = ReadParticles(params.input_file());
  PrintParticles(particles);

  return EXIT_SUCCESS;
}
