#pragma once

#include "bev_lidar_detection/detector.hpp"

#include <memory>
#include <string>
#include <vector>

namespace bev_lidar_detection {

struct CenterPointConfig {
  std::string engine_path;
  // BatchNorm-folded pillar feature network; empty means pfn_weights.bin beside the engine.
  std::string pfn_weights_path;
  float score_threshold = 0.3f;
  std::size_t max_output = 200;
  // Added to os_sensor z to match the nuScenes LIDAR_TOP height (about 1.84 m).
  float z_offset = 0.0f;
  // Model intensity = PointXYZI::intensity * intensity_scale (nuScenes uses 0..255).
  float intensity_scale = 255.0f;
};

// OpenPCDet CenterPoint-PointPillar, nuScenes weights: 360 degrees, +-51.2 m, 10 classes.
// Points are grouped by pillar on the CPU, a CUDA kernel computes the exact dynamic-pillar
// features into the BEV canvas, TensorRT runs backbone + heads, and the CPU decodes boxes.
class CenterPointDetector final : public ObjectDetector {
 public:
  static constexpr std::size_t max_points = 2000000;  // several accumulated sweeps
  explicit CenterPointDetector(const CenterPointConfig &config);
  ~CenterPointDetector() override;
  CenterPointDetector(const CenterPointDetector &) = delete;
  CenterPointDetector &operator=(const CenterPointDetector &) = delete;
  std::vector<Box3D> infer(const std::vector<PointXYZI> &points) override;
  // Multi-sweep input: ages[i] is how many seconds older point i's sweep is than the newest
  // sweep (the model's time-lag feature, 0 for the newest). Older sweeps must already be
  // transformed into the newest sweep's os_sensor frame.
  std::vector<Box3D> infer(const std::vector<PointXYZI> &points, const std::vector<float> &ages);
  const DetectorStats &lastStats() const noexcept override;
  const std::vector<std::string> &classNames() const noexcept override;
  std::size_t maxPoints() const noexcept override { return max_points; }
  const char *modelName() const noexcept override { return "CenterPoint_Pillar_nuScenes_FP16"; }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Rotated bird's-eye-view IoU of two boxes (x, y, size_x, size_y, yaw). Exposed for tests.
double bev_iou(const Box3D &a, const Box3D &b);

}  // namespace bev_lidar_detection
