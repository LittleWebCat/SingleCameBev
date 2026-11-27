#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace bev_visualization {

// SMOKE/KITTI class order: Car=0, Cyclist=1, Pedestrian=2. Coordinates use the
// optical frame: right/down/forward. center_xyz is the learned geometric center,
// before the upstream KITTI serialization moves Y to the bottom of the box.
struct Camera3DBox {
  int class_id{};
  double score{};
  cv::Vec3d center_xyz{};
  cv::Vec3d size_lhw{};
  double rotation_y{};
};

// Single-batch processed CHW heads: heatmap[3,H,W], regression[8,H,W]. Regression
// channels are depth, projected XY offsets, dimension LHW offsets, sine/cosine.
// Bad tensor schemas/parameters throw; invalid individual predictions are dropped.
// The native image is uniformly scaled by feature_width/image_width and centered
// vertically, matching the official SMOKE affine crop/padding transform.
std::vector<Camera3DBox> decode_smoke(
    const std::vector<float> &heatmap, const std::vector<float> &regression,
    int feature_width, int feature_height, const cv::Matx33d &K,
    int image_width, int image_height, double threshold = 0.25,
    std::size_t topk = 50);

// Corner index bits select positive local X(length), Y(height), Z(width).
// Local corners are centered on center_xyz and rotated about optical camera Y.
std::array<cv::Vec3d, 8> camera_box_corners(const Camera3DBox &box);

// Native pinhole projection; distortion is handled by the caller's image path.
// Returns nullopt if any corner is at/behind min_depth or projection is nonfinite.
std::optional<std::array<cv::Point2d, 8>> project_camera_box(
    const Camera3DBox &box, const cv::Matx33d &K, double min_depth = 0.3);

}  // namespace bev_visualization
