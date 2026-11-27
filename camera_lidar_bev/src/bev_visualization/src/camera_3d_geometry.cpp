#include "bev_visualization/camera_3d_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace bev_visualization {
namespace {
constexpr double kSmokePi = 3.14159;  // Official smoke_coder.py inference constant.
constexpr double kOrientationEpsilon = 1e-7;
constexpr std::array<std::array<double, 3>, 3> kDimensions{{
    {{3.88, 1.63, 1.53}}, {{1.78, 1.70, 0.58}}, {{0.88, 1.73, 0.67}}}};

bool finite(const cv::Vec3d &value) {
  return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}
void validate_intrinsics(const cv::Matx33d &K) {
  for (const auto value : K.val)
    if (!std::isfinite(value)) throw std::invalid_argument("Camera intrinsics must be finite");
  if (K(0, 0) <= 0 || K(1, 1) <= 0 || std::abs(K(2, 0)) > 1e-12 ||
      std::abs(K(2, 1)) > 1e-12 || std::abs(K(2, 2)-1.0) > 1e-12 ||
      std::abs(cv::determinant(K)) < 1e-12)
    throw std::invalid_argument("Expected nonsingular native pinhole camera intrinsics");
}
void validate_box(const Camera3DBox &box) {
  if (box.class_id < 0 || box.class_id > 2 || !std::isfinite(box.score) ||
      box.score < 0 || box.score > 1 || !finite(box.center_xyz) ||
      !finite(box.size_lhw) || box.size_lhw[0] <= 0 || box.size_lhw[1] <= 0 ||
      box.size_lhw[2] <= 0 || !std::isfinite(box.rotation_y))
    throw std::invalid_argument("Invalid camera 3D class, confidence or geometry");
}
bool probability(float value) { return std::isfinite(value) && value >= 0 && value <= 1; }
struct Peak {float score; std::size_t flat;};
}  // namespace

std::vector<Camera3DBox> decode_smoke(
    const std::vector<float> &heatmap, const std::vector<float> &regression,
    int feature_width, int feature_height, const cv::Matx33d &K,
    int image_width, int image_height, double threshold, std::size_t topk) {
  if (feature_width <= 0 || feature_height <= 0 || image_width <= 0 || image_height <= 0 ||
      !std::isfinite(threshold) || threshold < 0 || threshold > 1 || !topk)
    throw std::invalid_argument("Invalid SMOKE feature/image dimensions, threshold or top K");
  const auto width = static_cast<std::size_t>(feature_width);
  const auto height = static_cast<std::size_t>(feature_height);
  if (width > std::numeric_limits<std::size_t>::max()/height/8)
    throw std::invalid_argument("SMOKE tensor shape overflows storage");
  const std::size_t plane = width*height;
  if (heatmap.size() != 3*plane || regression.size() != 8*plane)
    throw std::invalid_argument("SMOKE expects processed CHW heatmap[3,H,W] and regression[8,H,W]");
  validate_intrinsics(K);
  const cv::Matx33d inverse = K.inv();
  const double scale = static_cast<double>(feature_width)/image_width;
  const double shift_y = (feature_height-image_height*scale)/2.0;
  std::vector<Peak> peaks;
  peaks.reserve(std::min(heatmap.size(), topk));
  for (std::size_t class_id = 0; class_id < 3; ++class_id) {
    const auto offset = class_id*plane;
    for (int y = 0; y < feature_height; ++y) for (int x = 0; x < feature_width; ++x) {
      const auto index = static_cast<std::size_t>(y)*width+x;
      const auto score = heatmap[offset+index];
      if (!probability(score) || score <= threshold) continue;
      bool maximum = true;
      for (int dy = -1; dy <= 1 && maximum; ++dy) for (int dx = -1; dx <= 1; ++dx) {
        const int nx = x+dx, ny = y+dy;
        if (nx < 0 || ny < 0 || nx >= feature_width || ny >= feature_height) continue;
        const auto neighbor = heatmap[offset+static_cast<std::size_t>(ny)*width+nx];
        if (probability(neighbor) && neighbor > score) {maximum = false; break;}
      }
      if (maximum) peaks.push_back({score, offset+index});
    }
  }
  // Global top K equals upstream per-class top K followed by global top K.
  // Flat-index ordering makes ties deterministic; torch.topk leaves tie order unspecified.
  const auto compare = [](const Peak &a, const Peak &b) {
    return a.score != b.score ? a.score > b.score : a.flat < b.flat;
  };
  const auto selected = std::min(topk, peaks.size());
  std::partial_sort(peaks.begin(), peaks.begin()+selected, peaks.end(), compare);
  peaks.resize(selected);
  std::vector<Camera3DBox> result;
  result.reserve(selected);
  for (const auto &peak : peaks) {
    const auto index = peak.flat%plane;
    std::array<double, 8> offsets{};
    bool valid = true;
    for (std::size_t channel = 0; channel < offsets.size(); ++channel) {
      offsets[channel] = regression[channel*plane+index];
      if (!std::isfinite(offsets[channel])) valid = false;
    }
    if (!valid) continue;
    // These channels are already sigmoid()-0.5 and L2-normalized in the model
    // head. Reject a raw/unprocessed or corrupt regression head rather than
    // supplying dimensions or heading when the model did not emit usable values.
    for (std::size_t channel = 3; channel <= 5; ++channel)
      if (std::abs(offsets[channel]) > 0.500001) valid = false;
    const double orientation_norm = std::hypot(offsets[6], offsets[7]);
    if (std::abs(orientation_norm-1.0) > 0.01) valid = false;
    if (!valid) continue;
    const double depth = 28.01+16.32*offsets[0];
    const double u = (static_cast<double>(index%width)+offsets[1])/scale;
    const double v = (static_cast<double>(index/width)+offsets[2]-shift_y)/scale;
    if (!std::isfinite(depth) || depth <= 0 || !std::isfinite(u) || !std::isfinite(v) ||
        u < 0 || v < 0 || u >= image_width || v >= image_height) continue;
    Camera3DBox box;
    box.class_id = static_cast<int>(peak.flat/plane);
    box.score = peak.score;
    box.center_xyz = inverse*cv::Vec3d(u*depth, v*depth, depth);
    for (int axis = 0; axis < 3; ++axis)
      box.size_lhw[axis] = kDimensions[box.class_id][axis]*std::exp(offsets[3+axis]);
    double alpha = std::atan(offsets[6]/(offsets[7]+kOrientationEpsilon));
    alpha += offsets[7] >= 0 ? -kSmokePi/2 : kSmokePi/2;
    box.rotation_y = alpha+std::atan(box.center_xyz[0]/(box.center_xyz[2]+kOrientationEpsilon));
    if (box.rotation_y > kSmokePi) box.rotation_y -= 2*kSmokePi;
    if (box.rotation_y < -kSmokePi) box.rotation_y += 2*kSmokePi;
    if (!finite(box.center_xyz) || !finite(box.size_lhw) || !std::isfinite(box.rotation_y)) continue;
    result.push_back(box);
  }
  return result;
}

std::array<cv::Vec3d, 8> camera_box_corners(const Camera3DBox &box) {
  validate_box(box);
  const double cosine = std::cos(box.rotation_y), sine = std::sin(box.rotation_y);
  std::array<cv::Vec3d, 8> corners{};
  for (std::size_t i = 0; i < corners.size(); ++i) {
    const double x = (i&1 ? 0.5 : -0.5)*box.size_lhw[0];
    const double y = (i&2 ? 0.5 : -0.5)*box.size_lhw[1];
    const double z = (i&4 ? 0.5 : -0.5)*box.size_lhw[2];
    corners[i] = box.center_xyz+cv::Vec3d(cosine*x+sine*z, y, -sine*x+cosine*z);
    if (!finite(corners[i])) throw std::invalid_argument("Camera box corners exceed finite coordinates");
  }
  return corners;
}

std::optional<std::array<cv::Point2d, 8>> project_camera_box(
    const Camera3DBox &box, const cv::Matx33d &K, double min_depth) {
  validate_intrinsics(K);
  if (!std::isfinite(min_depth) || min_depth <= 0)
    throw std::invalid_argument("Camera near-plane depth must be finite and positive");
  const auto corners = camera_box_corners(box);
  std::array<cv::Point2d, 8> projected{};
  for (std::size_t i = 0; i < corners.size(); ++i) {
    if (corners[i][2] <= min_depth) return std::nullopt;
    const auto homogeneous = K*corners[i];
    const double u = homogeneous[0]/homogeneous[2], v = homogeneous[1]/homogeneous[2];
    if (!std::isfinite(u) || !std::isfinite(v)) return std::nullopt;
    projected[i] = cv::Point2d(u, v);
  }
  return projected;
}
}  // namespace bev_visualization
