#pragma once

#include "bev_core/grid.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace bev_geometry {

// Sensor coordinates: x forward, y left, z up. No assumed vehicle transform.
struct Point { double x; double y; double z; };

enum class Label : std::uint8_t {
  Invalid = 0, Unknown = 1, Ground = 2, Obstacle = 3, Masked = 4
};

struct Config {
  // Positive distance from the sensor origin to the floor at x=y=0.
  // NaN permits a provisional low-surface estimate; it is not a measured height.
  double ground_height_m{std::numeric_limits<double>::quiet_NaN()};
  double ground_height_tolerance_m{0.35};
  double auto_ground_height_min_m{0.35};
  double auto_ground_height_max_m{4.0};
  double max_slope_deg{15.0};
  double fit_min_range_m{1.0};
  double fit_max_range_m{18.0};
  double min_range_m{0.5};
  double max_range_m{60.0};
  double ground_distance_m{0.10};
  double obstacle_min_height_m{0.15};
  double obstacle_max_height_m{3.0};
  double patch_size_m{1.0};
  double local_support_radius_m{2.0};
  std::size_t min_ground_points{80};
  std::size_t min_ground_patches{12};
  double min_ground_span_m{3.0};
  // Measurements are needed before enabling this vehicle/body exclusion box.
  bool self_mask_enabled{false};
  double self_x_min_m{-0.5}, self_x_max_m{0.5};
  double self_y_min_m{-0.5}, self_y_max_m{0.5};
  double self_z_min_m{-0.5}, self_z_max_m{0.5};
  bev_core::GridSpec grid{};
};

struct GroundModel {
  bool valid{false};
  // z = a*x + b*y + c, all in sensor coordinates.
  double a{0.0}, b{0.0}, c{0.0};
  std::size_t support_points{0}, support_patches{0};
  // Fit RMS over equally weighted low-patch seeds; dense walls near the floor
  // do not inflate this metric merely by contributing many tolerated returns.
  double rms_m{std::numeric_limits<double>::quiet_NaN()};
  double slope_deg{std::numeric_limits<double>::quiet_NaN()};
  bool used_height_prior{false};
  std::string reason{"no supported floor"};
};

struct Result {
  // One label for every original input index, including invalid/masked points.
  std::vector<Label> labels;
  // Height against the supported local floor used for the label; NaN otherwise.
  std::vector<double> heights_above_ground;
  GroundModel ground;
  bev_core::Grid grid;
  explicit Result(const bev_core::GridSpec &spec) : grid(spec) {}
};

// Throws invalid_argument on inconsistent/nonfinite configuration.
void validate_config(const Config &config);

// Single scan only. Free cells need observed ground and a clear 2D line of sight.
// No supported floor means no Ground/Obstacle labels and an unknown grid.
Result classify_and_grid(const std::vector<Point> &points, const Config &config = {});

}  // namespace bev_geometry
