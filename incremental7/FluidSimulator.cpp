#include <fstream>
#include <iostream>
#include <vector>

#include "Particle.h"
#include "SimulationParameters.h"
#include "StaggeredGrid.h"

namespace {

void Write3d(const Eigen::Vector3d& vec, std::ofstream* out) {
  (*out) << vec[0] << " " << vec[1] << " " << vec[2];
}

// Writes the position and velocity of each particle to the file with the
// specified |output_file_name|.
void WriteParticles(const char* output_file_name,
                    const std::vector<Particle>& particles) {
  std::ofstream out(output_file_name, std::ios::out);
  out << particles.size() << std::endl;
  for (std::vector<Particle>::const_iterator p = particles.begin();
       p != particles.end(); p++) {
    Write3d(p->pos, &out);
    out << " ";
    Write3d(p->vel, &out);
    out << std::endl;
  }
  out.close();
  std::cout << "Output file " << output_file_name << " saved." << std::endl;
}

}  // namespace

// Run a physics-based fluid simulation and print the resulting fluid particle
// positions at each time step to files.
int main(int argc, char** argv) {
  SimulationParameters params = ReadSimulationParameters(argc, argv);

  StaggeredGrid grid(params.nx(), params.ny(), params.nz(), params.lc(),
                     params.dx());

  std::vector<Particle> particles = ReadParticles(params.input_file());

  grid.ParticlesToGrid(particles);

  char output_file_name[100];
  int frame = 0;
  const double kFirstPositiveFrameTime = 1.0 / 30.0 - 0.0001;
  for (double time = 0.0, frame_time = -1.0; time < params.duration_seconds();
       time += params.dt_seconds(), frame_time -= params.dt_seconds()) {
    if (frame_time < 0.0) {
      sprintf(output_file_name, params.output_file_name_pattern().c_str(),
              frame);
      WriteParticles(output_file_name, particles);
      frame_time = kFirstPositiveFrameTime;
      frame++;
    }

    // Advect particles
    for (std::vector<Particle>::iterator p = particles.begin();
         p != particles.end(); p++) {
      p->pos = grid.Advect(p->pos, params.dt_seconds());
    }

    grid.ParticlesToGrid(particles);

    grid.ApplyGravity(params.dt_seconds());

    grid.ProjectPressure();

    for (std::vector<Particle>::iterator p = particles.begin();
         p != particles.end(); p++) {
      p->vel = grid.GridToParticle(params.flip_ratio(), *p);
    }
  }

  return EXIT_SUCCESS;
}
