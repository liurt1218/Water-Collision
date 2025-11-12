#ifndef SIMULATION_PARAMETERS_H_
#define SIMULATION_PARAMETERS_H_

#include <Eigen/Dense>
#include <string>

// A data type holding configuration settings for a FLIP/PIC simulation
class SimulationParameters {
 public:
  // Creates a new set of configuration settings for a simulation.
  SimulationParameters(double dt_seconds, double duration_seconds,
                       double density,
                       const Eigen::Matrix<std::size_t, 3, 1>& dimensions,
                       double dx, const Eigen::Vector3d& lc, double flip_ratio,
                       const std::string& input_file,
                       const std::string& output_file_name_pattern);

  // Copy constructor
  // The C++ compiler should NOT invoke this copy constructor when doing this:
  //
  // SimulationParameters MakeUnnamedSimParams() {
  //   return SimulationParameters(...);
  // }
  //
  // int main(...) {
  //   SimulationParameters s = MakeUnnamedSimParams();
  // }
  //
  // The C++ compiler should apply unnamed return value optimization to
  // eliminate the unnecessary copying of the unnamed object returned from
  // MakeSimParams() into s, and instead simply create s with the unnamed
  // returned object as its value. Similarly, when we return a named object:
  //
  // SimulationParameters MakeNamedSimParams() {
  //   SimulationParameters s(...);
  //   return s;
  // }
  //
  // int main(...) {
  //   SimulationParameters s = MakeNamedSimParams();
  // }
  //
  // The C++ compiler should apply named return value optimization to eliminate
  // the unnecessary copying of the returned object into s. Starting in C++17,
  // it is unnecessary to declare this copy constructor when return value
  // optimization is applied. However, prior to C++17, the C++ compiler still
  // requires this copy constructor to be declared when it *appears* to be
  // invoked as the compiler pretends return value optimization will not be
  // applied, and then the compiler applies it anyway. Due to subtleties in how
  // each compiler applies or refuses to apply return value optimization, we
  // explicitly implement this copy constructor to instantiate a complete
  // SimulationParameters instance and then promptly cause an assertion failure.
  // So, if this copy constructor is ever invoked, it will crash the program.
  SimulationParameters(const SimulationParameters& other);

  // Returns a new set of configuration settings for a simulation read from a
  // .json file.
  static SimulationParameters CreateFromJsonFile(
      const std::string& input_file_path);

  // Destroys this set of configuration settings.
  ~SimulationParameters();

  double dt_seconds() const { return dt_seconds_; }
  double duration_seconds() const { return duration_seconds_; }
  double density() const { return density_; }
  std::size_t nx() const { return dimensions_[0]; }
  std::size_t ny() const { return dimensions_[1]; }
  std::size_t nz() const { return dimensions_[2]; }
  double dx() const { return dx_; }
  const Eigen::Vector3d& lc() const { return lc_; }
  double flip_ratio() const { return flip_ratio_; }
  const std::string& input_file() const { return input_file_; }
  const std::string& output_file_name_pattern() const {
    return output_file_name_pattern_;
  }

 private:
  // Don't allow |this| to be assigned to another instance.
  SimulationParameters& operator=(const SimulationParameters& other);

  // Time step size for the simulation, in seconds
  const double dt_seconds_;

  // Duration of the simulation, in seconds
  const double duration_seconds_;

  // Density of the fluid
  const double density_;

  // Number of grid cells in each coordinate direction: (nx, ny, nz)
  const Eigen::Matrix<std::size_t, 3, 1> dimensions_;

  // Width (side length) of each grid cell--all are assumed to be cubes
  const double dx_;

  // Lower corner (minimum x, y, and z coordinates) of the grid
  const Eigen::Vector3d lc_;

  // Amount of FLIP vs. PIC
  // Higher values lead to less dissipation/damping/filtering/smoothing
  // via viscosity, but more proneness to noise
  const double flip_ratio_;

  // Input .json file containing initial particle positions and velocities
  const std::string input_file_;

  // Naming pattern for particle position data files output from the
  // simulation; e.g., "fluid%03d.txt" will lead to output files named
  // "fluid_001.txt", "fluid_002.txt", etc.
  const std::string output_file_name_pattern_;
};

// Reads a set of configuration settings from a file specified in a command-line
// argument.
SimulationParameters ReadSimulationParameters(int argc, char** argv);

#endif  // SIMULATION_PARAMETERS_H_
