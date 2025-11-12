#include "SimulationParameters.h"

#include <cassert>
#include <fstream>

#include "json/json.h"

// To disable assert*() calls, uncomment this line:
// #define NDEBUG

SimulationParameters::SimulationParameters(
    double dt_seconds, double duration_seconds, double density,
    const Eigen::Matrix<std::size_t, 3, 1>& dimensions, double dx,
    const Eigen::Vector3d& lc, double flip_ratio, const std::string& input_file,
    const std::string& output_file_name_pattern)
    : dt_seconds_(dt_seconds),
      duration_seconds_(duration_seconds),
      density_(density),
      dimensions_(dimensions),
      dx_(dx),
      lc_(lc),
      flip_ratio_(flip_ratio),
      input_file_(input_file),
      output_file_name_pattern_(output_file_name_pattern) {}

SimulationParameters::SimulationParameters(const SimulationParameters& other)
    : dt_seconds_(other.dt_seconds_),
      duration_seconds_(other.duration_seconds_),
      density_(other.density_),
      dimensions_(other.dimensions_),
      dx_(other.dx_),
      lc_(other.lc_),
      flip_ratio_(other.flip_ratio_),
      input_file_(other.input_file_),
      output_file_name_pattern_(other.output_file_name_pattern_) {
  assert(false);
}

SimulationParameters SimulationParameters::CreateFromJsonFile(
    const std::string& input_file_path) {
  std::ifstream in(input_file_path, std::ios::in);

  Json::Reader json_reader;
  Json::Value json_root;

  bool read_succeeded = json_reader.parse(in, json_root);
  assert(read_succeeded);

  double dt_seconds = json_root.get("dt", 1.0 / 300.0).asDouble();
  double duration_seconds = json_root.get("total_time", 1.0).asDouble();
  double density = json_root.get("density", 1.0 / 300.0).asDouble();
  double flip_ratio = json_root.get("flipRatio", 0.95).asDouble();

  int nx = json_root["res"][0].asInt();
  int ny = json_root["res"][1].asInt();
  int nz = json_root["res"][2].asInt();
  Eigen::Matrix<std::size_t, 3, 1> dimensions;
  dimensions << nx, ny, nz;

  double dx = json_root["h"].asDouble();

  double lc_x = json_root["lc"][0].asDouble();
  double lc_y = json_root["lc"][1].asDouble();
  double lc_z = json_root["lc"][2].asDouble();
  Eigen::Vector3d lc;
  lc << lc_x, lc_y, lc_z;

  std::string input_file = json_root["particles"].asString();
  std::string output_file_name_pattern =
      json_root.get("output_fname", std::string("output.%04d.txt")).asString();

  return SimulationParameters(dt_seconds, duration_seconds, density, dimensions,
                              dx, lc, flip_ratio, input_file,
                              output_file_name_pattern);
}

SimulationParameters::~SimulationParameters() {}

SimulationParameters ReadSimulationParameters(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "ERROR: .json file argument not found!" << std::endl;
    std::cout << "Usage: ./FluidSimulator [.json file path]" << std::endl;
    assert(false);  // crash the program
  }

  return SimulationParameters::CreateFromJsonFile(argv[1]);
}
