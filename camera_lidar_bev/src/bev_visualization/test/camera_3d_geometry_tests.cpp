#include "bev_visualization/camera_3d_geometry.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace bev_visualization;
namespace {
void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, const char *message, double tolerance = 2e-5) {
  require(std::isfinite(actual) && std::abs(actual-expected) <= tolerance, message);
}
template<class Function> void rejects(Function function, const char *message) {
  bool rejected = false;
  try {function();} catch (const std::invalid_argument &) {rejected = true;}
  require(rejected, message);
}
struct Heads {
  int width, height;
  std::size_t plane;
  std::vector<float> heatmap, regression;
  Heads(int w, int h):width(w),height(h),plane(static_cast<std::size_t>(w)*h),
      heatmap(3*plane, 0.0001F), regression(8*plane, 0.0F) {
    for (std::size_t i = 0; i < plane; ++i) regression[7*plane+i] = 1;
  }
  void peak(int class_id, int x, int y, float score,
            const std::array<float, 8> &values = {{0, 0, 0, 0, 0, 0, 0, 1}}) {
    const auto index = static_cast<std::size_t>(y)*width+x;
    heatmap[static_cast<std::size_t>(class_id)*plane+index] = score;
    for (std::size_t channel = 0; channel < values.size(); ++channel)
      regression[channel*plane+index] = values[channel];
  }
  std::vector<Camera3DBox> decode(const cv::Matx33d &K, int image_width, int image_height,
                                double threshold = .25, std::size_t topk = 50) const {
    return decode_smoke(heatmap, regression, width, height, K, image_width, image_height, threshold, topk);
  }
};

void official_reference_all_classes_and_center() {
  Heads heads(10, 6);
  heads.peak(0, 5, 3, .9F, {{1, .25F, -.2F, .2F, -.1F, 0, 0, 1}});
  heads.peak(1, 1, 1, .8F, {{0, 0, 0, 0, 0, 0, 0, -1}});
  heads.peak(2, 8, 4, .7F, {{-.5F, 0, 0, .1F, .2F, -.2F, 1, 0}});
  const cv::Matx33d K(100, 0, 50, 0, 200, 30, 0, 0, 1);
  const auto decoded = heads.decode(K, 100, 60);
  require(decoded.size() == 3, "Car/Cyclist/Pedestrian are separate learned classes");
  // Independent golden values from upstream SMOKECoder on CPU, before its
  // KITTI-bottom serialization. No test invokes the implementation's equations.
  constexpr double xyz[3][3] = {{1.1082496643, -.4433002472, 44.3300018311},
                                {-11.2040004730, -2.8009998798, 28.0100002289},
                                {5.9549999237, .9925001264, 19.8500003815}};
  constexpr double lhw[3][3] = {{4.7390427589, 1.4748849869, 1.5299999714},
                                {1.7799999714, 1.7000000477, .5799999833},
                                {.9725504518, 2.1130268574, .5485496521}};
  constexpr double yaw[3] = {-1.5458003283, 1.1902886629, .2914579809};
  for (std::size_t i = 0; i < decoded.size(); ++i) {
    require(decoded[i].class_id == static_cast<int>(i), "Class ID and descending-score order preserved");
    near(decoded[i].score, .9-.1*i, "Learned confidence preserved");
    for (int axis = 0; axis < 3; ++axis) {
      near(decoded[i].center_xyz[axis], xyz[i][axis], "Official learned-center location");
      near(decoded[i].size_lhw[axis], lhw[i][axis], "Official exponential learned dimensions L/H/W");
    }
    near(decoded[i].rotation_y, yaw[i], "Official sine/cosine orientation convention");
  }
  require(decoded[0].center_xyz[1] < 0, "Center must not be moved down by half the learned height");
}

void centered_affine_crop_and_padding() {
  Heads crop(320, 96);
  crop.peak(0, 160, 48, .9F, {{0, .5F, .25F, 0, 0, 0, 0, 1}});
  const cv::Matx33d K(1000, 0, 512, 0, 900, 384, 0, 0, 1);
  const auto decoded = crop.decode(K, 1024, 768);
  require(decoded.size() == 1, "Wide model crop admits native center");
  // Native affine: width scale .3125 and vertical shift -72 feature pixels.
  near(decoded[0].center_xyz[0], .044816, "Crop X maps to original native pixel513.6");
  near(decoded[0].center_xyz[1], .0248977778, "Crop Y maps to original native pixel384.8");
  near(decoded[0].center_xyz[2], 28.01, "Learned depth unaffected by affine crop");
  Heads padded(32, 32);
  padded.peak(0, 16, 16, .8F);
  padded.peak(0, 16, 0, .95F);
  padded.peak(0, 16, 31, .94F);
  const cv::Matx33d centered(100, 0, 50, 0, 100, 25, 0, 0, 1);
  const auto accepted = padded.decode(centered, 100, 50);
  require(accepted.size() == 1, "Predicted centers wholly in affine padding are dropped");
  near(accepted[0].center_xyz[0], 0, "Centered native optical X");
  near(accepted[0].center_xyz[1], 0, "Centered native optical Y");
}

void local_maximum_and_global_topk() {
  Heads heads(8, 8);
  heads.peak(0, 1, 1, .9F);
  heads.peak(0, 2, 1, .8F);
  heads.peak(0, 5, 5, .7F);
  heads.peak(1, 1, 1, .85F);
  heads.peak(2, 4, 1, .95F);
  const cv::Matx33d K(100, 0, 40, 0, 100, 40, 0, 0, 1);
  const auto all = heads.decode(K, 80, 80);
  require(all.size() == 4, "Only same-class 3x3 neighbor suppressed, other classes retained");
  const auto two = heads.decode(K, 80, 80, .25, 2);
  require(two.size() == 2 && two[0].class_id == 2 && two[1].class_id == 0,
          "Top K is global across learned classes");
  Heads ties(8, 8);
  ties.peak(0, 2, 2, .9F);
  ties.peak(0, 3, 2, .9F);
  const auto tied = ties.decode(K, 80, 80);
  require(tied.size() == 2, "Official max-pool equality retains both tied neighboring maxima");
  require(tied[0].center_xyz[0] < tied[1].center_xyz[0], "Tie ordering is deterministic by flat index");
  Heads boundary(8, 8);
  boundary.peak(0, 0, 0, .9F);
  boundary.peak(2, 7, 7, .25F);
  require(boundary.decode(K, 80, 80).size() == 1, "Border peak retained and official threshold is strict greater-than");
}

void official_orientation_quadrants_and_wrap() {
  const cv::Matx33d K(50, 0, 100, 0, 100, 60, 0, 0, 1);
  // Independent upstream CPU golden yaw values for ray X/Z=+/-1.6.
  const std::array<std::array<float, 2>, 4> orientation{{{{.6F, .8F}}, {{.6F, -.8F}},
                                                        {{-.6F, -.8F}}, {{-.6F, .8F}}}};
  constexpr std::array<double, 4> expected{{.0849030018, 1.9394909143, -3.0566868782, 3.0566871166}};
  for (std::size_t i = 0; i < orientation.size(); ++i) {
    Heads heads(20, 12);
    heads.peak(0, i == 3 ? 2 : 18, 6, .9F,
               {{0, 0, 0, 0, 0, 0, orientation[i][0], orientation[i][1]}});
    const auto decoded = heads.decode(K, 200, 120);
    require(decoded.size() == 1, "Normalized orientations accepted in every cosine/sign quadrant");
    near(decoded[0].rotation_y, expected[i], "Official atan+cosine-sign+ray+PI wrapping");
    require(decoded[0].rotation_y >= -3.14159 && decoded[0].rotation_y <= 3.14159,
            "Decoded rotation lies in official wrapped interval");
  }
}

void corrupt_heads_and_schema_guards() {
  Heads heads(8, 8);
  const cv::Matx33d K(100, 0, 40, 0, 100, 40, 0, 0, 1);
  heads.peak(0, 4, 4, .9F);
  for (std::size_t channel = 0; channel < 8; ++channel) {
    auto invalid = heads;
    invalid.regression[channel*heads.plane+36] = std::numeric_limits<float>::quiet_NaN();
    require(invalid.decode(K, 80, 80).empty(), "Every nonfinite regression channel drops its prediction");
  }
  for (const float score : {std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN(), 1.1F, -.1F}) {
    auto invalid = heads; invalid.heatmap[36] = score;
    require(invalid.decode(K, 80, 80).empty(), "Nonprobability heatmap value supplies no detection");
  }
  for (const auto values : std::array<std::array<float, 8>, 5>{{
       {{-2, 0, 0, 0, 0, 0, 0, 1}}, {{0, 1000, 0, 0, 0, 0, 0, 1}},
       {{0, 0, 0, .51F, 0, 0, 0, 1}}, {{0, 0, 0, 0, 0, 0, 0, 0}},
       {{0, 0, 0, 0, 0, 0, .2F, .2F}}}}) {
    auto invalid = heads; invalid.peak(0, 4, 4, .9F, values);
    require(invalid.decode(K, 80, 80).empty(), "Invalid depth/native center/processed dimensions/orientation has no fallback");
  }
  auto short_heatmap = heads; short_heatmap.heatmap.pop_back();
  rejects([&] {short_heatmap.decode(K, 80, 80);}, "Short heatmap schema rejected");
  auto short_regression = heads; short_regression.regression.pop_back();
  rejects([&] {short_regression.decode(K, 80, 80);}, "Short regression schema rejected");
  rejects([&] {decode_smoke({}, {}, 0, 8, K, 80, 80);}, "Zero feature dimensions rejected");
  rejects([&] {heads.decode(K, 0, 80);}, "Zero native dimensions rejected");
  rejects([&] {heads.decode(K, 80, 80, -.1);}, "Negative confidence threshold rejected");
  rejects([&] {heads.decode(K, 80, 80, 1.1);}, "Threshold above probability range rejected");
  rejects([&] {heads.decode(K, 80, 80, std::numeric_limits<double>::quiet_NaN());}, "Nonfinite threshold rejected");
  rejects([&] {heads.decode(K, 80, 80, .25, 0);}, "Zero top K rejected");
  auto singular = K; singular(0, 0) = 0;
  rejects([&] {heads.decode(singular, 80, 80);}, "Singular camera intrinsics rejected");
  auto bad_K = K; bad_K(0, 2) = std::numeric_limits<double>::infinity();
  rejects([&] {heads.decode(bad_K, 80, 80);}, "Nonfinite intrinsics rejected");
}

void optical_y_rotation_corners_and_projection() {
  Camera3DBox box{0, .9, {0, 0, 10}, {4, 2, 2}, std::acos(-1.0)/2};
  const auto corners = camera_box_corners(box);
  // Positive camera-Y yaw rotates local +X towards camera -Z, not sensor XY.
  near(corners[0][0], -1, "Yaw90 local negative width becomes optical left");
  near(corners[0][1], -1, "Learned center corner extends half height upward");
  near(corners[0][2], 12, "Yaw90 negative local length extends optical forward");
  near(corners[1][2], 8, "Yaw90 positive local length extends optical backward");
  near(corners[2][1], 1, "Positive optical Y corner extends downward");
  near(corners[4][0], 1, "Local width maps into optical X after yaw");
  const cv::Matx33d K(100, 0, 50, 0, 100, 40, 0, 0, 1);
  const auto projected = project_camera_box(box, K);
  require(projected.has_value(), "Every forward corner projects into native pinhole coordinates");
  near((*projected)[0].x, 41.6666666667, "Independent pinhole corner X");
  near((*projected)[0].y, 31.6666666667, "Independent pinhole corner Y");
  near((*projected)[1].x, 37.5, "Depth-dependent camera corner projection");
  near((*projected)[1].y, 27.5, "Optical down convention maintained");
  auto near_box = box; near_box.center_xyz[2] = 2.3;
  require(!project_camera_box(near_box, K, .31), "One near-plane corner rejects the entire projection");
  near_box.center_xyz[2] = -1;
  require(!project_camera_box(near_box, K), "Behind-camera corners cannot produce visible boxes");
  auto invalid_box = box; invalid_box.size_lhw[1] = 0;
  rejects([&] {camera_box_corners(invalid_box);}, "Degenerate learned dimensions rejected");
  invalid_box = box; invalid_box.rotation_y = std::numeric_limits<double>::quiet_NaN();
  rejects([&] {camera_box_corners(invalid_box);}, "Nonfinite geometry rejected");
  rejects([&] {project_camera_box(box, K, 0);}, "Nonpositive near plane rejected");
}
}  // namespace

int main() {
  try {
    official_reference_all_classes_and_center();
    centered_affine_crop_and_padding();
    local_maximum_and_global_topk();
    official_orientation_quadrants_and_wrap();
    corrupt_heads_and_schema_guards();
    optical_y_rotation_corners_and_projection();
    std::cout << "SMOKE camera 3D geometry: 6 CPU test groups passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
