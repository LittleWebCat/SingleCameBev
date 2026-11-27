#include "bev_visualization/object_fusion.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace bev_visualization {
namespace {

bool finite(const XYZ &point) {
  return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}
bool supported_class(const std::string &name) {
  return name == "person" || name == "bicycle" || name == "car" ||
         name == "motorcycle" || name == "bus" || name == "truck";
}
double squared_distance(const XYZ &left, const XYZ &right) {
  const double dx = left.x-right.x, dy = left.y-right.y, dz = left.z-right.z;
  return dx*dx + dy*dy + dz*dz;
}

constexpr std::size_t missing_row = std::numeric_limits<std::size_t>::max();

// Ouster return IDs occupy a compact range even after invalid returns are
// removed. Use bounded direct indexing for that common case; wide arbitrary
// UINT32 IDs retain the same hash lookup and validation semantics.
class SourceRows {
 public:
  explicit SourceRows(const std::vector<std::uint32_t> &indices) {
    if (indices.empty()) return;
    const auto bounds = std::minmax_element(indices.begin(), indices.end());
    minimum_ = *bounds.first;
    const std::uint64_t span = static_cast<std::uint64_t>(*bounds.second)-minimum_+1;
    const auto count = static_cast<std::uint64_t>(indices.size());
    const std::uint64_t doubled_count = count > std::numeric_limits<std::uint64_t>::max()/2 ?
        std::numeric_limits<std::uint64_t>::max() : count*2;
    const auto dense_limit = std::max<std::uint64_t>(1024, doubled_count);
    if (span <= dense_limit && span <= std::numeric_limits<std::size_t>::max()) {
      dense_.assign(static_cast<std::size_t>(span), missing_row);
      for (std::size_t row = 0; row < indices.size(); ++row) {
        auto &slot = dense_[static_cast<std::size_t>(static_cast<std::uint64_t>(indices[row])-minimum_)];
        if (slot != missing_row) throw std::invalid_argument("Full geometry has duplicate source indices");
        slot = row;
      }
    } else {
      sparse_.reserve(indices.size());
      for (std::size_t row = 0; row < indices.size(); ++row)
        if (!sparse_.emplace(indices[row], row).second)
          throw std::invalid_argument("Full geometry has duplicate source indices");
    }
  }
  std::size_t find(std::uint32_t index) const {
    if (!dense_.empty()) {
      if (index < minimum_) return missing_row;
      const auto offset = static_cast<std::uint64_t>(index)-minimum_;
      return offset < dense_.size() ? dense_[static_cast<std::size_t>(offset)] : missing_row;
    }
    const auto found = sparse_.find(index);
    return found == sparse_.end() ? missing_row : found->second;
  }
 private:
  std::uint32_t minimum_{};
  std::vector<std::size_t> dense_;
  std::unordered_map<std::uint32_t, std::size_t> sparse_;
};

struct VisibleObstacle {
  std::size_t classified_index{};
  XYZ point{};
  double u{}, v{}, depth{};
  std::size_t pixel{};
};

struct Voxel {
  std::int64_t x{}, y{}, z{};
  bool operator==(const Voxel &other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};
struct VoxelHash {
  std::size_t operator()(const Voxel &voxel) const {
    const auto x = std::hash<std::int64_t>{}(voxel.x);
    const auto y = std::hash<std::int64_t>{}(voxel.y);
    const auto z = std::hash<std::int64_t>{}(voxel.z);
    return x ^ (y + 0x9e3779b9U + (x << 6U) + (x >> 2U)) ^
           (z + 0x9e3779b9U + (y << 6U) + (y >> 2U));
  }
};
Voxel voxel_for(const XYZ &point, double radius) {
  return {static_cast<std::int64_t>(std::floor(point.x/radius)),
          static_cast<std::int64_t>(std::floor(point.y/radius)),
          static_cast<std::int64_t>(std::floor(point.z/radius))};
}

// Bucket entries are removed once reached. Dense neighboring surfaces therefore
// do not repeatedly scan every already-visited return. Actual adjacency is the
// Euclidean radius, rather than unconditional 26-neighbor voxel connectivity.
std::vector<std::vector<std::size_t>> components(
    const std::vector<VisibleObstacle> &points, const std::vector<std::size_t> &selected,
    double radius) {
  std::unordered_map<Voxel, std::vector<std::size_t>, VoxelHash> buckets;
  buckets.reserve(selected.size());
  for (const auto index : selected) buckets[voxel_for(points[index].point, radius)].push_back(index);
  std::vector<std::vector<std::size_t>> result;
  const double radius_squared = radius*radius;
  while (!buckets.empty()) {
    auto first = buckets.begin();
    const auto seed = first->second.back();
    first->second.pop_back();
    if (first->second.empty()) buckets.erase(first);
    std::vector<std::size_t> component{seed};
    for (std::size_t cursor = 0; cursor < component.size(); ++cursor) {
      const auto index = component[cursor];
      const auto voxel = voxel_for(points[index].point, radius);
      for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
        const Voxel neighbor{voxel.x+dx, voxel.y+dy, voxel.z+dz};
        auto found = buckets.find(neighbor);
        if (found == buckets.end()) continue;
        auto &unvisited = found->second;
        std::size_t position = 0;
        while (position < unvisited.size()) {
          if (squared_distance(points[index].point, points[unvisited[position]].point) <= radius_squared) {
            component.push_back(unvisited[position]);
            unvisited[position] = unvisited.back();
            unvisited.pop_back();
          } else {
            ++position;
          }
        }
        if (unvisited.empty()) buckets.erase(found);
      }
    }
    result.push_back(std::move(component));
  }
  return result;
}

struct SupportedComponent {
  std::vector<std::size_t> indices;
  XYZ minimum{}, maximum{};
  double median_depth{};
};

bool acceptable_extent(const XYZ &size, const std::string &name, const FusionConfig &config) {
  if (!finite(size) || size.x < 0 || size.y < 0 || size.z < 0 ||
      std::max(size.x, size.y) > config.max_observed_extent_m ||
      size.z > config.max_observed_height_m) return false;
  // Loose upper bounds reject long background surfaces. They never supply an
  // expected full size or expand the measured points into a class-shaped box.
  if (name == "person") return std::max(size.x, size.y) <= 2.5 && size.z <= 3.0;
  if (name == "bicycle" || name == "motorcycle")
    return std::max(size.x, size.y) <= 4.5 && size.z <= 3.0;
  return true;
}

}  // namespace

void validate_fusion_config(const FusionConfig &config) {
  const std::array<double, 8> finite_values{config.min_confidence, config.cluster_resolution_m,
      config.min_optical_depth_m, config.max_optical_depth_m, config.max_observed_extent_m,
      config.max_observed_height_m, config.ambiguity_depth_m, config.ambiguity_support_ratio};
  for (const auto value : finite_values)
    if (!std::isfinite(value)) throw std::invalid_argument("Fusion configuration must be finite");
  if (config.min_confidence < 0 || config.min_confidence > 1 ||
      config.min_support_points < 3 || config.min_support_points > 10000 ||
      config.min_visible_pixels < 3 || config.min_visible_pixels > config.min_support_points ||
      config.min_central_points < 1 || config.min_central_points > config.min_support_points ||
      config.cluster_resolution_m < 0.05 || config.cluster_resolution_m > 2.0 ||
      config.min_optical_depth_m <= 0 || config.max_optical_depth_m <= config.min_optical_depth_m ||
      config.max_optical_depth_m > 200 || config.max_observed_extent_m <= 0 ||
      config.max_observed_extent_m > 30 || config.max_observed_height_m <= 0 ||
      config.max_observed_height_m > 10 || config.ambiguity_depth_m < 0 ||
      config.ambiguity_depth_m > 10 || config.ambiguity_support_ratio <= 0 ||
      config.ambiguity_support_ratio > 1 || !config.max_detections ||
      config.max_detections > 1000 || config.max_points_per_detection < config.min_support_points ||
      config.max_points_per_detection > 100000)
    throw std::invalid_argument("Fusion configuration bounds are inconsistent");
}

FusionResult fuse_camera_objects(const std::vector<XYZ> &full_geometry,
    const std::vector<std::uint32_t> &full_source_indices,
    const std::vector<ClassifiedPoint> &classified,
    const std::vector<CameraDetection> &detections,
    const Camera &camera, const Transform &camera_from_sensor, const FusionConfig &config) {
  validate_fusion_config(config);
  validate_camera(camera);
  validate_transform(camera_from_sensor);
  if (full_geometry.size() != full_source_indices.size())
    throw std::invalid_argument("Full geometry and source-index arrays disagree");
  if (detections.size() > config.max_detections)
    throw std::invalid_argument("Detection count exceeds the configured bounded workload");
  const SourceRows full_rows(full_source_indices);
  // Projection already returns full input rows. Store the joined classified
  // row once here, avoiding a second source-ID map and a per-visible-point hash.
  std::vector<std::size_t> classified_by_full_row(full_geometry.size(), missing_row);
  for (std::size_t row = 0; row < classified.size(); ++row) {
    const auto &point = classified[row];
    const auto full_row = full_rows.find(point.source_index);
    if (full_row == missing_row)
      throw std::invalid_argument("Classified source index is missing from the full scan");
    auto &matched_row = classified_by_full_row[full_row];
    if (matched_row != missing_row)
      throw std::invalid_argument("Classified geometry has duplicate source indices");
    matched_row = row;
    if (finite(point.point) && finite(full_geometry[full_row]) &&
        squared_distance(point.point, full_geometry[full_row]) > 1e-8)
      throw std::invalid_argument("Matched source indices have inconsistent geometry");
  }

  FusionResult result;
  const auto projection = project_visible_sensor_points(full_geometry, camera, camera_from_sensor);
  result.visible_geometry_points = projection.points.size();
  std::vector<VisibleObstacle> visible;
  visible.reserve(projection.points.size());
  for (const auto &projected : projection.points) {
    if (projected.optical_depth < config.min_optical_depth_m ||
        projected.optical_depth > config.max_optical_depth_m) continue;
    const auto classified_row = classified_by_full_row[projected.point_index];
    if (classified_row == missing_row) continue;  // Includes self-masked occluders.
    const auto &point = classified[classified_row];
    if (point.classification != 3 || !finite(point.point) ||
        !std::isfinite(point.height_above_ground) || point.height_above_ground <= 0 ||
        std::max({std::abs(point.point.x), std::abs(point.point.y), std::abs(point.point.z)}) > 1e6) continue;
    visible.push_back({classified_row, point.point, projected.u, projected.v,
                      projected.optical_depth,
                      static_cast<std::size_t>(projected.pixel_y)*camera.width + projected.pixel_x});
  }
  result.supported_obstacle_points = visible.size();

  std::vector<std::size_t> detection_order(detections.size());
  std::iota(detection_order.begin(), detection_order.end(), 0);
  std::stable_sort(detection_order.begin(), detection_order.end(), [&](std::size_t left, std::size_t right) {
    const auto a = detections[left].confidence, b = detections[right].confidence;
    // Nonfinite inputs are rejected below; preserve a strict ordering here.
    return (std::isfinite(a) ? a : -1.0) > (std::isfinite(b) ? b : -1.0);
  });
  std::vector<bool> claimed(classified.size(), false);
  for (const auto detection_index : detection_order) {
    ++result.detections_considered;
    const auto &detection = detections[detection_index];
    if (!supported_class(detection.class_name) || !std::isfinite(detection.confidence) ||
        detection.confidence < config.min_confidence || detection.confidence > 1 ||
        !std::isfinite(detection.x_min) || !std::isfinite(detection.y_min) ||
        !std::isfinite(detection.x_max) || !std::isfinite(detection.y_max) ||
        detection.x_min >= detection.x_max || detection.y_min >= detection.y_max) {
      ++result.rejected_invalid_detections; continue;
    }
    const double x_min = std::max(0.0, detection.x_min), y_min = std::max(0.0, detection.y_min);
    const double x_max = std::min(static_cast<double>(camera.width), detection.x_max);
    const double y_max = std::min(static_cast<double>(camera.height), detection.y_max);
    if (x_min >= x_max || y_min >= y_max) { ++result.rejected_invalid_detections; continue; }
    std::vector<std::size_t> selected;
    for (std::size_t row = 0; row < visible.size(); ++row) {
      const auto &point = visible[row];
      if (!claimed[point.classified_index] && point.u >= x_min && point.u < x_max &&
          point.v >= y_min && point.v < y_max) selected.push_back(row);
    }
    if (selected.size() < config.min_support_points || selected.size() > config.max_points_per_detection) {
      ++result.rejected_unsupported; continue;
    }
    const double central_x_min = x_min+(x_max-x_min)*0.2;
    const double central_x_max = x_max-(x_max-x_min)*0.2;
    const double central_y_min = y_min+(y_max-y_min)*0.2;
    const double central_y_max = y_max-(y_max-y_min)*0.2;
    std::vector<SupportedComponent> supported;
    for (auto &component : components(visible, selected, config.cluster_resolution_m)) {
      if (component.size() < config.min_support_points) continue;
      XYZ minimum = visible[component.front()].point, maximum = minimum;
      std::size_t central_count = 0;
      std::unordered_set<std::size_t> pixels;
      std::vector<double> depths;
      depths.reserve(component.size());
      for (const auto row : component) {
        const auto &point = visible[row];
        minimum.x = std::min(minimum.x, point.point.x); maximum.x = std::max(maximum.x, point.point.x);
        minimum.y = std::min(minimum.y, point.point.y); maximum.y = std::max(maximum.y, point.point.y);
        minimum.z = std::min(minimum.z, point.point.z); maximum.z = std::max(maximum.z, point.point.z);
        central_count += point.u >= central_x_min && point.u < central_x_max &&
                         point.v >= central_y_min && point.v < central_y_max;
        pixels.insert(point.pixel);
        depths.push_back(point.depth);
      }
      const XYZ size{maximum.x-minimum.x, maximum.y-minimum.y, maximum.z-minimum.z};
      if (central_count < config.min_central_points || pixels.size() < config.min_visible_pixels ||
          !acceptable_extent(size, detection.class_name, config)) continue;
      const auto middle = depths.begin()+static_cast<std::ptrdiff_t>(depths.size()/2);
      std::nth_element(depths.begin(), middle, depths.end());
      supported.push_back({std::move(component), minimum, maximum, *middle});
    }
    if (supported.empty()) { ++result.rejected_unsupported; continue; }
    std::sort(supported.begin(), supported.end(), [](const auto &left, const auto &right) {
      if (left.median_depth != right.median_depth) return left.median_depth < right.median_depth;
      return left.indices.size() > right.indices.size();
    });
    const auto &foreground = supported.front();
    bool ambiguous = false;
    for (std::size_t row = 1; row < supported.size(); ++row) {
      if (supported[row].median_depth-foreground.median_depth > config.ambiguity_depth_m) break;
      if (static_cast<double>(supported[row].indices.size()) >=
          config.ambiguity_support_ratio*foreground.indices.size()) { ambiguous = true; break; }
    }
    if (ambiguous) { ++result.rejected_ambiguous; continue; }
    ObjectCandidate candidate;
    candidate.class_name = detection.class_name;
    candidate.confidence = detection.confidence;
    candidate.center = {(foreground.minimum.x+foreground.maximum.x)*0.5,
                        (foreground.minimum.y+foreground.maximum.y)*0.5,
                        (foreground.minimum.z+foreground.maximum.z)*0.5};
    candidate.size = {foreground.maximum.x-foreground.minimum.x,
                      foreground.maximum.y-foreground.minimum.y,
                      foreground.maximum.z-foreground.minimum.z};
    candidate.visible_support = foreground.indices.size();
    candidate.classified_indices.reserve(foreground.indices.size());
    for (const auto row : foreground.indices) {
      const auto index = visible[row].classified_index;
      candidate.classified_indices.push_back(index);
      claimed[index] = true;
    }
    std::sort(candidate.classified_indices.begin(), candidate.classified_indices.end());
    result.candidates.push_back(std::move(candidate));
  }
  return result;
}

}  // namespace bev_visualization
