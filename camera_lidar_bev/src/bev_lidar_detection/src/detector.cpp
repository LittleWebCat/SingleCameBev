#include "bev_lidar_detection/detector.hpp"
#include "pointpillar.hpp"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace bev_lidar_detection {
class Detector::Impl {
 public:
  Impl(const std::string &engine_path, float score_threshold, std::size_t max_output)
      : max_output_(max_output) {
    if (!std::isfinite(score_threshold) || score_threshold <= 0 || score_threshold >= 1 || max_output == 0)
      throw std::invalid_argument("Invalid PointPillars score threshold/output limit");
    pointpillar::lidar::CoreParameter p;
    auto &v = p.voxelization;
    v.min_range = nvtype::Float3(0, -39.68f, -3);
    v.max_range = nvtype::Float3(69.12f, 39.68f, 1);
    v.voxel_size = nvtype::Float3(0.16f, 0.16f, 4);
    v.grid_size = v.compute_grid_size(v.max_range, v.min_range, v.voxel_size);
    v.max_voxels = Detector::max_pillars;
    v.max_points_per_voxel = 32;
    v.max_points = Detector::max_points;
    v.num_feature = 4;
    p.lidar_model = engine_path;
    p.lidar_post.min_range = v.min_range;
    p.lidar_post.max_range = v.max_range;
    p.lidar_post.feature_size = nvtype::Int2(v.grid_size.x/2, v.grid_size.y/2);
    p.lidar_post.score_thresh = score_threshold;
    core_ = pointpillar::lidar::create_core(p);
    if (!core_) throw std::runtime_error("Failed to load PointPillars engine " + engine_path);
    if (cudaStreamCreate(&stream_) != cudaSuccess) throw std::runtime_error("CUDA stream creation failed");
    accepted_.reserve(Detector::max_points);
  }
  ~Impl() { if (stream_) cudaStreamDestroy(stream_); }
  std::vector<Box3D> infer(const std::vector<PointXYZI> &points) {
    const auto start = std::chrono::steady_clock::now();
    stats_ = {};
    stats_.input_points = points.size();
    if (points.size() > Detector::max_points) throw std::invalid_argument("PointPillars input exceeds 300000 points");
    accepted_.clear();
    for (const auto &p : points) {
      if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) && std::isfinite(p.intensity)
          && p.x >= 0 && p.x < 69.12f && p.y >= -39.68f && p.y < 39.68f && p.z >= -3 && p.z < 1
          && p.intensity >= 0 && p.intensity <= 1)
        accepted_.push_back(p);
    }
    stats_.accepted_points = accepted_.size();
    std::vector<Box3D> out;
    if (!accepted_.empty()) {
      static_assert(sizeof(PointXYZI) == 4*sizeof(float));
      const auto boxes = core_->forward(reinterpret_cast<const float *>(accepted_.data()), accepted_.size(), stream_);
      stats_.input_pillars = core_->input_pillars();
      stats_.accepted_pillars = std::min(stats_.input_pillars, Detector::max_pillars);
      stats_.truncated_pillars = stats_.input_pillars - stats_.accepted_pillars;
      for (const auto &b : boxes) {
        // Upstream .w means dx (Car length), .l means dy (Car width).
        if (std::isfinite(b.x) && std::isfinite(b.y) && std::isfinite(b.z) && std::isfinite(b.w)
            && std::isfinite(b.l) && std::isfinite(b.h) && std::isfinite(b.rt) && std::isfinite(b.score)
            && b.w > 0 && b.l > 0 && b.h > 0 && b.id >= 0 && b.id <= 2) {
          out.push_back({b.x,b.y,b.z,b.w,b.l,b.h,std::remainder(b.rt,2.0f*3.14159265358979323846f),b.score,b.id});
          if (out.size() >= max_output_) break;
        }
      }
    }
    stats_.output_boxes = out.size();
    stats_.processing_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return out;
  }
  DetectorStats stats_;
 private:
  std::shared_ptr<pointpillar::lidar::Core> core_;
  cudaStream_t stream_ = nullptr;
  std::size_t max_output_;
  std::vector<PointXYZI> accepted_;
};
Detector::Detector(const std::string &p,float s,std::size_t n):impl_(std::make_unique<Impl>(p,s,n)){}
Detector::~Detector() = default;
std::vector<Box3D> Detector::infer(const std::vector<PointXYZI> &p) { return impl_->infer(p); }
const DetectorStats &Detector::lastStats() const noexcept { return impl_->stats_; }
}
