// Offline single- vs multi-sweep CenterPoint over extracted recording scans.
//
// Usage: centerpoint_sweep_eval engine.plan scans_dir out.jsonl SWEEPS [MAX_SPAN_S]
//
// scans_dir/scans.csv (from tools/extract_lidar_sweeps.py) lists, per scan: index, scan start
// and end (ns), the os_sensor pose in the map at scan end (x y z qx qy qz qw), pose_ok and the
// float32 [N,4] point file (os_sensor x y z, reflectivity/255). Writes one JSON line per scan
// with every box at or above the 0.1 score floor and the timing breakdown.
#include "bev_lidar_detection/centerpoint_detector.hpp"
#include "bev_lidar_detection/sweep_accumulator.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace bev_lidar_detection;

std::vector<PointXYZI> load(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("Cannot open " + path);
  const auto n = static_cast<std::size_t>(f.tellg());
  if (n % sizeof(PointXYZI)) throw std::runtime_error("Bad point file " + path);
  std::vector<PointXYZI> points(n / sizeof(PointXYZI));
  f.seekg(0);
  f.read(reinterpret_cast<char *>(points.data()), static_cast<std::streamsize>(n));
  return points;
}

int main(int argc, char **argv) {
  try {
    if (argc < 5 || argc > 6)
      throw std::runtime_error("Usage: centerpoint_sweep_eval engine.plan scans_dir out.jsonl SWEEPS [MAX_SPAN_S]");
    const std::string dir = argv[2];
    const auto sweeps = static_cast<std::size_t>(std::stoul(argv[4]));
    const double span = argc > 5 ? std::stod(argv[5]) : 0.5;
    CenterPointConfig config;
    config.engine_path = argv[1];
    config.score_threshold = 0.1f;  // keep everything; thresholds are applied in the analysis
    CenterPointDetector detector(config);
    SweepAccumulator accumulator(sweeps, span);
    std::ifstream index(dir + "/scans.csv");
    std::ofstream out(argv[3]);
    if (!index || !out) throw std::runtime_error("Cannot open scans.csv or the output file");
    const auto &names = detector.classNames();
    std::string line;
    std::getline(index, line);  // header
    std::vector<PointXYZI> merged;
    std::vector<float> ages;
    std::size_t frames = 0;
    while (std::getline(index, line)) {
      std::stringstream row(line);
      std::string cell;
      std::vector<std::string> c;
      while (std::getline(row, cell, ',')) c.push_back(cell);
      if (c.size() != 12) throw std::runtime_error("Bad scans.csv row: " + line);
      SensorPose pose;
      pose.t = {std::stod(c[3]), std::stod(c[4]), std::stod(c[5])};
      pose.q = {std::stod(c[6]), std::stod(c[7]), std::stod(c[8]), std::stod(c[9])};
      const bool pose_ok = c[10] == "1";
      const std::int64_t end_ns = std::stoll(c[2]);
      auto points = load(dir + "/" + c[11]);
      const auto start = std::chrono::steady_clock::now();
      std::vector<Box3D> boxes;
      std::size_t used = 1;
      if (sweeps == 1 || !pose_ok) {
        if (!pose_ok) accumulator.clear();  // no pose: cannot align earlier sweeps
        boxes = detector.infer(points);
      } else {
        accumulator.add(std::move(points), end_ns, pose);
        accumulator.build(merged, ages);
        used = accumulator.sweeps();
        boxes = detector.infer(merged, ages);
      }
      const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      const auto &s = detector.lastStats();
      out << std::setprecision(6) << "{\"i\":" << c[0] << ",\"end_ns\":" << end_ns << ",\"sweeps_used\":" << used
          << ",\"pose_ok\":" << (pose_ok ? "true" : "false") << ",\"points\":" << s.accepted_points
          << ",\"pillars\":" << s.accepted_pillars << ",\"total_ms\":" << total_ms
          << ",\"detector_ms\":" << s.processing_ms << ",\"pre_ms\":" << s.preprocess_ms
          << ",\"net_ms\":" << s.inference_ms << ",\"post_ms\":" << s.postprocess_ms << ",\"boxes\":[";
      for (std::size_t i = 0; i < boxes.size(); ++i) {
        const auto &b = boxes[i];
        out << (i ? "," : "") << "[\"" << names.at(static_cast<std::size_t>(b.class_id)) << "\"," << b.score << ','
            << b.x << ',' << b.y << ',' << b.z << ',' << b.size_x << ',' << b.size_y << ',' << b.size_z << ',' << b.yaw << ']';
      }
      out << "]}\n";
      if (++frames % 200 == 0) std::cerr << frames << " scans\n";
    }
    std::cerr << "done: " << frames << " scans, " << sweeps << " sweep(s)\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
