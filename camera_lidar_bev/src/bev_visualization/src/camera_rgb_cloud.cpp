#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>
#include "bev_visualization/cloud_io.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2_msgs/msg/tf_message.hpp"

namespace bev_visualization {
namespace {
using Clock = std::chrono::steady_clock;
using Image = sensor_msgs::msg::Image;
using Info = sensor_msgs::msg::CameraInfo;
using Cloud = sensor_msgs::msg::PointCloud2;
using Status = diagnostic_msgs::msg::DiagnosticStatus;

std::int64_t stamp_ns(const std_msgs::msg::Header &header) {
  return static_cast<std::int64_t>(header.stamp.sec)*1000000000LL+header.stamp.nanosec;
}
bool same_header(const std_msgs::msg::Header &a, const std_msgs::msg::Header &b) {
  return a.frame_id == b.frame_id && a.stamp.sec == b.stamp.sec && a.stamp.nanosec == b.stamp.nanosec;
}
template<std::size_t N> std::array<double, N> array_from_yaml(const YAML::Node &node, const std::string &name) {
  if (!node.IsSequence() || node.size() != N) throw std::invalid_argument(name+" must contain "+std::to_string(N)+" values");
  std::array<double, N> result{};
  for (std::size_t i = 0; i < N; ++i) {
    result[i] = node[i].as<double>();
    if (!std::isfinite(result[i])) throw std::invalid_argument(name+" contains a nonfinite value");
  }
  return result;
}
Transform transform_from_yaml(const YAML::Node &node, const std::string &name) {
  if (!node.IsSequence() || node.size() != 4) throw std::invalid_argument(name+" must be 4x4");
  Transform result{};
  for (std::size_t row = 0; row < 4; ++row) {
    const auto values = array_from_yaml<4>(node[row], name);
    std::copy(values.begin(), values.end(), result.begin()+row*4);
  }
  validate_transform(result);
  return result;
}
template<typename A, typename B> bool close_arrays(const A &a, const B &b, double tolerance = 1e-8) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::abs(a[i]-b[i]) > tolerance) return false;
  return true;
}
template<typename Message> struct Received {
  typename Message::ConstSharedPtr message;
  Clock::time_point receipt;
};
}  // namespace

class CameraRGBCloud final : public rclcpp::Node {
 public:
  CameraRGBCloud() : Node("camera_rgb_cloud") {
    const auto intrinsics_path = declare_parameter("intrinsics_file", std::string{});
    const auto extrinsics_path = declare_parameter("extrinsics_file", std::string{});
    camera_frame_ = declare_parameter("camera_frame", "camera_front_optical_frame");
    max_pair_delta_s_ = declare_parameter("max_pair_delta_s", 0.15);
    stale_after_s_ = declare_parameter("stale_after_s", 1.0);
    const auto max_publish_hz = declare_parameter("max_publish_hz", 20.0);
    max_publish_hz_ = max_publish_hz;
    if (intrinsics_path.empty() || extrinsics_path.empty())
      throw std::invalid_argument("intrinsics_file and extrinsics_file must name the native camera and B candidate YAMLs");
    if (camera_frame_.empty() || !std::isfinite(max_pair_delta_s_) || max_pair_delta_s_ <= 0 || max_pair_delta_s_ > 0.15 ||
        !std::isfinite(stale_after_s_) || stale_after_s_ <= 0 || stale_after_s_ > 2.0 ||
        !std::isfinite(max_publish_hz) || max_publish_hz <= 0 || max_publish_hz > 20.0)
      throw std::invalid_argument("Invalid frame, pairing, freshness or publication rate parameter");
    load_calibration(intrinsics_path, extrinsics_path);
    const auto image_topic = declare_parameter("image_topic", "/camera/front/image_raw");
    const auto info_topic = declare_parameter("camera_info_topic", "/camera/front/camera_info");
    const auto points_topic = declare_parameter("points_topic", "/points");
    const auto output_topic = declare_parameter("output_topic", "/camera_lidar_bev/points_rgb");
    for (const auto &topic : {image_topic, info_topic, points_topic, output_topic})
      if (topic.empty()) throw std::invalid_argument("Topic names must not be empty");
    output_pub_ = create_publisher<Cloud>(output_topic, rclcpp::QoS(1).reliable());
    diagnostics_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/camera_lidar_bev/diagnostics", rclcpp::QoS(5));
    const auto camera_qos = rclcpp::SensorDataQoS().keep_last(5);
    image_sub_ = create_subscription<Image>(image_topic, camera_qos, [this](Image::ConstSharedPtr msg) {
      image_error_ = image_layout_error(*msg);
      images_.push_back({std::move(msg), Clock::now()});
      while (images_.size() > 16) images_.pop_front();
    });
    info_sub_ = create_subscription<Info>(info_topic, camera_qos, [this](Info::ConstSharedPtr msg) {
      info_error_ = camera_info_error(*msg);
      infos_.push_back({std::move(msg), Clock::now()});
      while (infos_.size() > 24) infos_.pop_front();
    });
    cloud_sub_ = create_subscription<Cloud>(points_topic, rclcpp::QoS(1).reliable(), [this](Cloud::ConstSharedPtr msg) {
      cloud_ = {std::move(msg), Clock::now()};
      ++cloud_version_;
    });
    tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>("/tf_static", rclcpp::QoS(10).reliable().transient_local(),
      [this](tf2_msgs::msg::TFMessage::ConstSharedPtr msg) { observe_transform(*msg); });
    process_timer_ = create_wall_timer(std::chrono::duration<double>(1.0/max_publish_hz), [this]() { process(); });
    diagnostics_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() { publish_diagnostics(); });
    RCLCPP_WARN(get_logger(), "Camera RGB preview uses user-selected B calibration; geometric accuracy and sensor synchronization remain unvalidated. No TF is published.");
  }

 private:
  void load_calibration(const std::string &intrinsics_path, const std::string &extrinsics_path) {
    const auto native = YAML::LoadFile(intrinsics_path);
    if (native["distortion_model"].as<std::string>() != "plumb_bob")
      throw std::invalid_argument("Only native plumb_bob calibration is supported");
    camera_.width = native["image_width"].as<std::uint32_t>();
    camera_.height = native["image_height"].as<std::uint32_t>();
    for (const auto &entry : std::array<std::pair<const char*, std::pair<int,int>>,4>{{
      {"camera_matrix",{3,3}}, {"distortion_coefficients",{1,5}}, {"rectification_matrix",{3,3}}, {"projection_matrix",{3,4}}}}) {
      const auto matrix = native[entry.first];
      if (matrix["rows"].as<int>() != entry.second.first || matrix["cols"].as<int>() != entry.second.second)
        throw std::invalid_argument(std::string("Unexpected YAML matrix layout for ")+entry.first);
    }
    camera_.k = array_from_yaml<9>(native["camera_matrix"]["data"], "K");
    camera_.d = array_from_yaml<5>(native["distortion_coefficients"]["data"], "D");
    expected_r_ = array_from_yaml<9>(native["rectification_matrix"]["data"], "R");
    expected_p_ = array_from_yaml<12>(native["projection_matrix"]["data"], "P");
    validate_camera(camera_);
    const std::array<double,9> identity = {1,0,0,0,1,0,0,0,1};
    const std::array<double,12> native_p = {camera_.k[0],0,camera_.k[2],0,0,camera_.k[4],camera_.k[5],0,0,0,1,0};
    if (!close_arrays(expected_r_, identity) || !close_arrays(expected_p_, native_p))
      throw std::invalid_argument("Raw-image calibration must use identity R and native [K|0] P");
    const auto candidate = YAML::LoadFile(extrinsics_path);
    if (candidate["units"].as<std::string>() != "m" || candidate["direction"].as<std::string>() != "lidar_to_camera" ||
        candidate["selected_hypothesis"].as<std::string>() != "B")
      throw std::invalid_argument("Extrinsics must be the user-selected B candidate in meters");
    const auto external = candidate["transforms"]["T_camera_from_os_sensor"];
    if (external["source_frame"].as<std::string>() != "os_sensor" || external["target_frame"].as<std::string>() != camera_frame_)
      throw std::invalid_argument("Candidate source/target frames disagree with os_sensor and configured optical frame");
    camera_from_sensor_ = transform_from_yaml(external["matrix"], "T_camera_from_os_sensor");
    const auto observed = candidate["observed_driver_transform"];
    if (observed["source_frame"].as<std::string>() != "os_lidar" || observed["target_frame"].as<std::string>() != "os_sensor")
      throw std::invalid_argument("Candidate observed driver transform has unexpected frames");
    expected_sensor_from_lidar_ = transform_from_yaml(observed["matrix"], "T_os_sensor_from_os_lidar");
    const auto composed = compose(camera_from_sensor_, expected_sensor_from_lidar_);
    const auto stored_composed = transform_from_yaml(candidate["transforms"]["T_camera_from_os_lidar"]["matrix"], "T_camera_from_os_lidar");
    if (!close_arrays(composed, stored_composed)) throw std::invalid_argument("Candidate stored B composition is inconsistent");
  }

  std::string image_layout_error(const Image &image) const {
    if (image.header.frame_id != camera_frame_) return "Raw image optical frame does not match candidate";
    if (image.width != camera_.width || image.height != camera_.height) return "Raw image dimensions disagree with native calibration";
    if (image.encoding != "bgr8" && image.encoding != "rgb8") return "Raw image must use bgr8 or rgb8";
    if (static_cast<std::uint64_t>(image.width)*3 > image.step ||
        static_cast<std::uint64_t>(image.step)*image.height != image.data.size()) return "Invalid raw image row/payload layout";
    return {};
  }
  std::string camera_info_error(const Info &info) const {
    if (info.header.frame_id != camera_frame_ || info.width != camera_.width || info.height != camera_.height)
      return "CameraInfo frame/dimensions disagree with native calibration";
    if (info.distortion_model != "plumb_bob" || !close_arrays(info.k, camera_.k) || !close_arrays(info.d, camera_.d) ||
        !close_arrays(info.r, expected_r_) || !close_arrays(info.p, expected_p_)) return "CameraInfo full K/D/R/P does not match native calibration";
    if (info.binning_x > 1 || info.binning_y > 1 || info.roi.x_offset || info.roi.y_offset || info.roi.do_rectify ||
        (info.roi.width && info.roi.width != camera_.width) || (info.roi.height && info.roi.height != camera_.height))
      return "CameraInfo ROI/binning changes the calibrated native image";
    return {};
  }

  void observe_transform(const tf2_msgs::msg::TFMessage &message) {
    for (const auto &record : message.transforms) {
      if (record.header.frame_id != "os_sensor" || record.child_frame_id != "os_lidar") continue;
      const auto &q = record.transform.rotation; const auto &p = record.transform.translation;
      const auto norm2 = q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
      if (!std::isfinite(norm2) || std::abs(norm2-1.0) > 1e-6) {
        tf_error_ = "Observed driver quaternion is not finite/unit"; tf_conflict_ = true; continue;
      }
      const double norm = std::sqrt(norm2), x=q.x/norm, y=q.y/norm, z=q.z/norm, w=q.w/norm;
      Transform transform = {1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), p.x,
                            2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), p.y,
                            2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), p.z,
                            0,0,0,1};
      try { validate_transform(transform); }
      catch (const std::exception &error) { tf_error_ = error.what(); tf_conflict_ = true; continue; }
      if (!close_arrays(transform, expected_sensor_from_lidar_) || (tf_received_ && !close_arrays(transform, sensor_from_lidar_))) {
        tf_error_ = "Observed os_sensor <- os_lidar transform conflicts with selected B evidence";
        tf_conflict_ = true; continue;
      }
      sensor_from_lidar_ = transform; tf_received_ = true;
    }
  }

  template<typename Message> bool fresh(const Received<Message> &input, const Clock::time_point receipt_now,
                                        const std::int64_t ros_now) const {
    if (!input.message || std::chrono::duration<double>(receipt_now-input.receipt).count() > stale_after_s_) return false;
    const auto stamp = stamp_ns(input.message->header);
    const double age = static_cast<double>(ros_now-stamp)/1e9;
    return input.message->header.stamp.nanosec < 1000000000U && stamp > 0 && age >= -0.05 && age <= stale_after_s_;
  }
  void unavailable(const std::string &reason) {
    current_error_ = reason;
    if (has_live_output_) {
      std_msgs::msg::Header header; header.frame_id = "os_sensor"; header.stamp = now();
      output_pub_->publish(encode_rgb({}, header));
      has_live_output_ = false;
      ++cleared_outputs_;
    }
  }

  void process() {
    const auto receipt_now = Clock::now(); const auto ros_now = now().nanoseconds();
    if (tf_conflict_) { unavailable(tf_error_); return; }
    if (!tf_received_) { unavailable("Waiting for actual os_sensor <- os_lidar driver static TF"); return; }
    if (!fresh(cloud_, receipt_now, ros_now)) { unavailable("Waiting for a fresh LiDAR cloud; no cached scan is republished"); return; }
    if (cloud_.message->header.frame_id != "os_lidar") { unavailable("Input point cloud must be in os_lidar"); return; }
    if (!image_error_.empty()) { unavailable(image_error_); return; }
    if (!info_error_.empty()) { unavailable(info_error_); return; }
    if (images_.empty() || infos_.empty()) { unavailable("Waiting for exact Image/CameraInfo header pairs"); return; }
    if (!fresh(images_.back(), receipt_now, ros_now) || !fresh(infos_.back(), receipt_now, ros_now)) {
      unavailable("Camera Image/CameraInfo stream is stale"); return;
    }
    const auto cloud_stamp = stamp_ns(cloud_.message->header);
    const Received<Image> *chosen_image = nullptr;
    const Received<Info> *chosen_info = nullptr;
    double best_delta = std::numeric_limits<double>::infinity();
    for (auto image = images_.rbegin(); image != images_.rend(); ++image) {
      if (!fresh(*image, receipt_now, ros_now) || !image_layout_error(*image->message).empty()) continue;
      const auto image_stamp = stamp_ns(image->message->header);
      const double delta = std::abs(static_cast<double>(cloud_stamp-image_stamp)/1e9);
      if (delta > max_pair_delta_s_ || delta >= best_delta) continue;
      for (auto info = infos_.rbegin(); info != infos_.rend(); ++info) {
        if (same_header(info->message->header, image->message->header) &&
            fresh(*info, receipt_now, ros_now) && camera_info_error(*info->message).empty()) {
          chosen_image = &*image; chosen_info = &*info; best_delta = delta; break;
        }
      }
    }
    if (!chosen_image || !chosen_info) { unavailable("No exact native Image/CameraInfo pair within allowed LiDAR header delta"); return; }
    if (cloud_version_ == last_published_version_) return;
    try {
      const auto &image = *chosen_image->message;
      const auto xyz = decode_xyz(*cloud_.message);
      const ImageView pixels{image.data.data(), image.data.size(), image.step, image.width, image.height, image.encoding == "bgr8"};
      auto colored = colorize(xyz, camera_, pixels, camera_from_sensor_, sensor_from_lidar_);
      // Computation cannot extend the lifetime of a stale sensor sample.
      const auto complete_receipt = Clock::now(); const auto complete_ros = now().nanoseconds();
      if (!fresh(cloud_, complete_receipt, complete_ros) || !fresh(*chosen_image, complete_receipt, complete_ros) ||
          !fresh(*chosen_info, complete_receipt, complete_ros)) { unavailable("Sensor pair became stale during RGB association"); return; }
      auto header = cloud_.message->header; header.frame_id = "os_sensor";
      output_pub_->publish(encode_rgb(colored.points, header));
      last_valid_count_ = colored.points.size();
      last_stats_ = std::move(colored); last_stats_.points.clear(); last_stats_.points.shrink_to_fit();
      last_pair_delta_s_ = best_delta;
      last_published_version_ = cloud_version_; ++published_clouds_;
      has_live_output_ = true; current_error_.clear(); last_output_receipt_ = Clock::now();
      last_image_stamp_ = stamp_ns(image.header); last_cloud_stamp_ = cloud_stamp;
      if (published_clouds_ == 1) RCLCPP_INFO(get_logger(), "RGB preview publishing in os_sensor: %zu valid points, %zu camera-colored, %.6f s header delta", last_valid_count_, last_stats_.colored_count, best_delta);
    } catch (const std::exception &error) { unavailable(error.what()); }
  }

  static void add(Status &status, const std::string &key, const std::string &value) {
    diagnostic_msgs::msg::KeyValue entry; entry.key=key; entry.value=value; status.values.push_back(entry);
  }
  void publish_diagnostics() {
    diagnostic_msgs::msg::DiagnosticArray message; message.header.stamp = now();
    Status status; status.name="camera_lidar_bev/camera_rgb_cloud"; status.hardware_id="front_camera+ouster";
    status.level = Status::WARN;
    status.message = current_error_.empty() ? "Camera RGB preview active; B geometry and sensor synchronization are provisional" : current_error_;
    add(status,"calibration_status","external_candidate_unvalidated");
    add(status,"source_frame_selection","user_visual_B_os_sensor");
    add(status,"accuracy_validated","false"); add(status,"synchronization_validated","false");
    add(status,"tf_published","false"); add(status,"output_frame","os_sensor");
    add(status,"driver_tf_observed",tf_received_ ? "true" : "false");
    add(status,"publishing_fresh_cloud",has_live_output_ ? "true" : "false");
    add(status,"published_clouds",std::to_string(published_clouds_));
    add(status,"received_clouds",std::to_string(cloud_version_));
    add(status,"publication_cap_hz",std::to_string(max_publish_hz_));
    add(status,"stale_outputs_cleared",std::to_string(cleared_outputs_));
    add(status,"input_points",std::to_string(last_stats_.input_count));
    add(status,"valid_points",std::to_string(last_valid_count_));
    add(status,"colored_points",std::to_string(last_stats_.colored_count));
    add(status,"radial_domain_excluded_points",std::to_string(last_stats_.radial_excluded_count));
    add(status,"header_delta_s",std::to_string(last_pair_delta_s_));
    add(status,"image_header_ns",std::to_string(last_image_stamp_));
    add(status,"cloud_header_ns",std::to_string(last_cloud_stamp_));
    if (published_clouds_) add(status,"last_output_receipt_age_s",std::to_string(std::chrono::duration<double>(Clock::now()-last_output_receipt_).count()));
    message.status.push_back(std::move(status)); diagnostics_pub_->publish(std::move(message));
  }

  Camera camera_;
  std::array<double,9> expected_r_{};
  std::array<double,12> expected_p_{};
  Transform camera_from_sensor_{}, expected_sensor_from_lidar_{}, sensor_from_lidar_{};
  std::string camera_frame_, image_error_, info_error_, tf_error_, current_error_{"Waiting for sensor input"};
  double max_pair_delta_s_{}, stale_after_s_{}, last_pair_delta_s_{}, max_publish_hz_{};
  bool tf_received_{false}, tf_conflict_{false}, has_live_output_{false};
  std::deque<Received<Image>> images_;
  std::deque<Received<Info>> infos_;
  Received<Cloud> cloud_;
  std::uint64_t cloud_version_{}, last_published_version_{}, published_clouds_{}, cleared_outputs_{};
  std::size_t last_valid_count_{};
  std::int64_t last_image_stamp_{}, last_cloud_stamp_{};
  Clock::time_point last_output_receipt_{};
  ColorResult last_stats_;
  rclcpp::Subscription<Image>::SharedPtr image_sub_;
  rclcpp::Subscription<Info>::SharedPtr info_sub_;
  rclcpp::Subscription<Cloud>::SharedPtr cloud_sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_;
  rclcpp::Publisher<Cloud>::SharedPtr output_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
  rclcpp::TimerBase::SharedPtr process_timer_, diagnostics_timer_;
};
}  // namespace bev_visualization

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try { rclcpp::spin(std::make_shared<bev_visualization::CameraRGBCloud>()); }
  catch (const std::exception &error) {
    RCLCPP_FATAL(rclcpp::get_logger("camera_rgb_cloud"), "%s", error.what());
    rclcpp::shutdown(); return 1;
  }
  rclcpp::shutdown(); return 0;
}
