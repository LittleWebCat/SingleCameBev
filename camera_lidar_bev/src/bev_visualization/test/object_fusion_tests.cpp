#include "bev_visualization/object_fusion.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace bev_visualization;
namespace {
void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, const std::string &message) {
  require(std::abs(actual-expected) < 1e-8, message);
}
template<typename F> void rejects(F function, const std::string &message) {
  bool rejected = false;
  try { function(); } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, message);
}
Camera camera() { return {100, 80, {80,0,50,0,80,40,0,0,1}, {0,0,0,0,0}}; }
CameraDetection detection(std::string name = "car", double confidence = 0.9) {
  return {std::move(name), confidence, 34, 30, 65, 50};
}
struct Scene {
  std::vector<XYZ> full;
  std::vector<std::uint32_t> ids;
  std::vector<ClassifiedPoint> classified;
  void point(XYZ xyz, bool classified_return = true, std::uint8_t label = 3,
             double height = 1.0) {
    const auto id = static_cast<std::uint32_t>(full.size()*7+100);  // Not a row index.
    full.push_back(xyz); ids.push_back(id);
    if (classified_return) classified.push_back({xyz, id, label, height});
  }
  void surface(double center_x, double center_y, double z, bool classified_return = true) {
    for (int row = 0; row < 4; ++row) for (int column = 0; column < 4; ++column)
      point({center_x+(column-1.5)*0.2, center_y+(row-1.5)*0.2, z}, classified_return);
  }
  FusionResult fuse(const std::vector<CameraDetection> &detections,
                    const FusionConfig &config = {}) const {
    return fuse_camera_objects(full, ids, classified, detections, camera(), identity_transform(), config);
  }
};

void test_observed_extent_and_original_indices() {
  Scene scene;
  scene.surface(-0.5, 0, 6);
  std::reverse(scene.classified.begin(), scene.classified.end());
  const auto original = scene.classified;
  const auto result = scene.fuse({detection()});
  require(result.candidates.size() == 1, "Supported native-image obstacle should yield an observed extent");
  const auto &candidate = result.candidates.front();
  require(candidate.class_name == "car" && candidate.confidence == 0.9,
          "Camera class/confidence changed during geometry association");
  near(candidate.center.x, -0.5, "Observed center x is wrong");
  near(candidate.center.y, 0, "Observed center y is wrong");
  near(candidate.center.z, 6, "Observed center z is wrong");
  near(candidate.size.x, 0.6, "Observed x extent is wrong");
  near(candidate.size.y, 0.6, "Observed y extent is wrong");
  near(candidate.size.z, 0, "A single observed face must not receive fabricated box depth");
  require(candidate.visible_support == 16 && candidate.classified_indices.size() == 16,
          "Visible support must preserve every accepted return");
  for (std::size_t row = 0; row < candidate.classified_indices.size(); ++row) {
    require(candidate.classified_indices[row] == row, "Classified input rows were confused with source IDs");
    require(scene.classified[row].source_index == original[row].source_index &&
            scene.classified[row].classification == original[row].classification,
            "Fusion changed source geometry/classification");
  }
}

void test_full_geometry_occludes_masked_body() {
  Scene scene;
  scene.surface(-0.5, 0, 6);
  // Identical camera rays at half the optical depth, absent from the classified
  // cloud because the vehicle filter removed them. They must still occlude.
  const auto far = scene.full;
  for (const auto &point : far) scene.point({point.x*0.5, point.y*0.5, point.z*0.5}, false);
  const auto result = scene.fuse({detection()});
  require(result.candidates.empty() && result.supported_obstacle_points == 0,
          "Self-masked near vehicle geometry failed to occlude a distant candidate");
  require(result.visible_geometry_points == 16, "Native optical-depth visibility count is wrong");
}

void test_foreground_and_background_stay_separate() {
  Scene scene;
  scene.surface(-0.5, 0, 6);
  scene.surface(0.9, 0, 12);
  const auto result = scene.fuse({detection()});
  require(result.candidates.size() == 1, "Distinct far background should not invalidate a supported foreground");
  const auto &candidate = result.candidates.front();
  near(candidate.center.z, 6, "Distant background became the observed object");
  near(candidate.size.z, 0, "Foreground and background were merged into a deep box");
  require(std::all_of(candidate.classified_indices.begin(), candidate.classified_indices.end(),
                     [](std::size_t row) { return row < 16; }),
          "Background source returns entered the foreground candidate");
}

void test_ambiguous_components_and_metric_radius() {
  Scene scene;
  scene.surface(-0.65, 0, 6);
  scene.surface(0.65, 0, 6.6);
  auto box = detection(); box.x_min = 30; box.x_max = 70;
  const auto result = scene.fuse({box});
  require(result.candidates.empty() && result.rejected_ambiguous == 1,
          "Two supported objects at similar depths must remain an ambiguous camera association");
  // Neighboring hash cells alone must not connect returns beyond the metric
  // radius. Two layers 0.6m apart occupy adjacent 0.45m cells but remain separate.
  Scene layers;
  layers.surface(-0.5, 0, 6.05);
  layers.surface(-0.15, 0.45, 6.65);
  FusionConfig config;
  config.min_support_points = 6;
  config.min_visible_pixels = 6;
  const auto metric = layers.fuse({box}, config);
  require(metric.candidates.empty() && metric.rejected_ambiguous == 1,
          "Spatial hash adjacency bypassed the Euclidean cluster radius");
}

void test_missing_floor_sparse_and_unsupported_detection() {
  Scene scene;
  scene.surface(-0.5, 0, 6);
  for (auto &point : scene.classified) {
    point.classification = 1;
    point.height_above_ground = std::numeric_limits<double>::quiet_NaN();
  }
  require(scene.fuse({detection()}).candidates.empty(), "Unsupported floor evidence yielded a candidate");
  for (auto &point : scene.classified) point.classification = 3;
  require(scene.fuse({detection()}).candidates.empty(), "Nonfinite ground-height evidence yielded a candidate");
  for (auto &point : scene.classified) point.height_above_ground = 1.0;
  scene.classified.resize(5);
  require(scene.fuse({detection()}).candidates.empty(), "Sparse obstacle returns yielded a candidate");
  Scene supported;
  supported.surface(-0.5, 0, 6);
  auto invalid = detection(); invalid.confidence = std::numeric_limits<double>::quiet_NaN();
  auto bad_box = detection(); bad_box.x_max = bad_box.x_min;
  const auto rejected = supported.fuse({detection("dog"), detection("person", 0.1), invalid, bad_box});
  require(rejected.candidates.empty() && rejected.rejected_invalid_detections == 4,
          "Unsupported classes, low confidence or invalid native boxes were accepted");
}

void test_provenance_and_configuration_validation() {
  Scene scene;
  scene.surface(-0.5, 0, 6);
  auto wrong = scene;
  wrong.classified[0].source_index = 9999;
  rejects([&] { wrong.fuse({detection()}); }, "Missing original source identity was accepted");
  wrong = scene;
  wrong.classified[0].point.x += 0.01;
  rejects([&] { wrong.fuse({detection()}); }, "Different scans were joined using source IDs alone");
  wrong = scene; wrong.ids[1] = wrong.ids[0];
  rejects([&] { wrong.fuse({detection()}); }, "Duplicate original source IDs were accepted");
  wrong = scene; wrong.classified.push_back(wrong.classified[0]);
  rejects([&] { wrong.fuse({detection()}); }, "Duplicate classified source IDs were accepted");
  wrong = scene; wrong.ids.pop_back();
  rejects([&] { wrong.fuse({detection()}); }, "Geometry/source arrays with unequal lengths were accepted");
  FusionConfig invalid; invalid.cluster_resolution_m = 0;
  rejects([&] { scene.fuse({detection()}, invalid); }, "Zero metric clustering radius was accepted");
  invalid = {}; invalid.max_detections = 1;
  rejects([&] { scene.fuse({detection(), detection()}, invalid); }, "Unbounded detection workload was accepted");
}

void test_visibility_projection_contract_and_claiming() {
  const auto model = camera();
  const auto identity = identity_transform();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto visible = project_visible_sensor_points({{0,0,8}, {0,0,4}, {0,0,4},
                                                       {0,0,-1}, {nan,0,3}}, model, identity);
  require(visible.points.size() == 1 && visible.points[0].point_index == 1,
          "Projection helper lost nearest-depth and first-row tie rules");
  near(visible.points[0].u, 50, "Native u projection is wrong");
  near(visible.points[0].v, 40, "Native v projection is wrong");
  auto distorted = model; distorted.d = {-1,0,0,0,0};
  const auto central = project_visible_sensor_points({{1,0,1}, {0,0,4}}, distorted, identity);
  require(central.radial_excluded_count == 1 && central.points.size() == 1,
          "Refactored helper accepted a folded radial branch");
  Scene scene; scene.surface(-0.5, 0, 6);
  const auto claimed = scene.fuse({detection("truck", 0.8), detection("car", 0.95)});
  require(claimed.candidates.size() == 1 && claimed.candidates.front().class_name == "car",
          "Overlapping detections assigned the same visible returns to conflicting classes");
}

void test_compact_and_wide_source_id_ranges() {
  Scene original;
  original.surface(-0.5, 0, 6);
  std::reverse(original.classified.begin(), original.classified.end());
  const auto reference = original.fuse({detection()});
  require(reference.candidates.size() == 1, "Reference identity fixture was unsupported");
  for (int mode = 0; mode < 3; ++mode) {
    auto scene = original;
    for (std::size_t row = 0; row < scene.ids.size(); ++row) {
      // Nonzero compact minimum, compact range ending at UINT32_MAX, and wide
      // sparse range containing both UINT32 boundaries exercise both lookups.
      const auto maximum = std::numeric_limits<std::uint32_t>::max();
      scene.ids[row] = mode == 0 ? static_cast<std::uint32_t>(50000+row*2) :
        mode == 1 ? maximum-static_cast<std::uint32_t>((scene.ids.size()-1-row)*2) :
        row == scene.ids.size()-1 ? maximum : static_cast<std::uint32_t>(row*200000003U);
    }
    for (auto &point : scene.classified) {
      const auto found = std::find(original.ids.begin(), original.ids.end(), point.source_index);
      require(found != original.ids.end(), "Fixture source provenance was lost");
      point.source_index = scene.ids[static_cast<std::size_t>(found-original.ids.begin())];
    }
    const auto result = scene.fuse({detection()});
    require(result.candidates.size() == 1 && result.visible_geometry_points == reference.visible_geometry_points &&
            result.supported_obstacle_points == reference.supported_obstacle_points,
            "Compact and wide source ranges changed visibility or association");
    const auto &actual = result.candidates.front();
    const auto &expected = reference.candidates.front();
    require(actual.classified_indices == expected.classified_indices &&
            actual.visible_support == expected.visible_support,
            "Optimized identity lookup changed original classified rows");
    near(actual.center.x, expected.center.x, "Identity range changed observed center");
    near(actual.size.x, expected.size.x, "Identity range changed observed extent");
    auto duplicate = scene; duplicate.ids[1] = duplicate.ids[0];
    rejects([&] { duplicate.fuse({detection()}); }, "Lookup accepted duplicate full IDs");
    duplicate = scene; duplicate.classified.push_back(duplicate.classified[0]);
    rejects([&] { duplicate.fuse({detection()}); }, "Lookup accepted duplicate classified IDs");
    auto mismatched = scene; mismatched.classified[0].point.y += 0.02;
    rejects([&] { mismatched.fuse({detection()}); }, "Lookup lost matching-coordinate checks");
    if (mode < 2) {
      auto missing = scene;
      missing.classified[0].source_index = scene.ids.front()+1;  // Inside an unoccupied dense slot.
      rejects([&] { missing.fuse({detection()}); }, "Dense source-range hole was treated as a valid return");
    }
  }
  Scene empty;
  require(empty.fuse({detection()}).candidates.empty(), "Empty source ranges produced a candidate");
}
}  // namespace

int main() {
  try {
    test_observed_extent_and_original_indices();
    test_full_geometry_occludes_masked_body();
    test_foreground_and_background_stay_separate();
    test_ambiguous_components_and_metric_radius();
    test_missing_floor_sparse_and_unsupported_detection();
    test_provenance_and_configuration_validation();
    test_visibility_projection_contract_and_claiming();
    test_compact_and_wide_source_id_ranges();
    std::cout << "Object fusion tests passed: observed extents, source identity, vehicle occlusion, "
                 "foreground separation, ambiguity, metric clustering, unsupported evidence, claiming "
                 "and compact/wide UINT32 source ranges\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
  return 0;
}
