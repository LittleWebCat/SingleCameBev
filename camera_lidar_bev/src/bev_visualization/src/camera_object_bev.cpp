#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>
#include "bev_visualization/cloud_io.hpp"
#include "bev_visualization/object_fusion.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "tf2_msgs/msg/tf_message.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace bev_visualization {
namespace {
using Clock = std::chrono::steady_clock;
using Cloud = sensor_msgs::msg::PointCloud2;
using Info = sensor_msgs::msg::CameraInfo;
using Detections = vision_msgs::msg::Detection2DArray;
using Detections3D = vision_msgs::msg::Detection3DArray;
using Markers = visualization_msgs::msg::MarkerArray;
using Marker = visualization_msgs::msg::Marker;
using Status = diagnostic_msgs::msg::DiagnosticStatus;
using Field = sensor_msgs::msg::PointField;
struct PreparedOutputs {
  Detections3D detections;
  Markers markers;
  Markers vehicle_bev_markers;
  Cloud semantic;
};

std::int64_t stamp_ns(const std_msgs::msg::Header &header) {
  return static_cast<std::int64_t>(header.stamp.sec)*1000000000LL+header.stamp.nanosec;
}
bool same_header(const std_msgs::msg::Header &a, const std_msgs::msg::Header &b) {
  return a.frame_id == b.frame_id && a.stamp.sec == b.stamp.sec && a.stamp.nanosec == b.stamp.nanosec;
}
template<typename Message> struct Received {
  typename Message::ConstSharedPtr message;
  Clock::time_point receipt;
};
template<typename Message> void append(std::deque<Received<Message>> &ring,
                                       typename Message::ConstSharedPtr message,
                                       std::size_t capacity=8) {
  ring.push_back({std::move(message), Clock::now()});
  while (ring.size() > capacity) ring.pop_front();
}
template<std::size_t N> std::array<double, N> yaml_array(const YAML::Node &node, const std::string &name) {
  if (!node.IsSequence() || node.size() != N)
    throw std::invalid_argument(name+" must contain "+std::to_string(N)+" values");
  std::array<double, N> result{};
  for (std::size_t i = 0; i < N; ++i) {
    result[i] = node[i].as<double>();
    if (!std::isfinite(result[i])) throw std::invalid_argument(name+" contains a nonfinite value");
  }
  return result;
}
Transform yaml_transform(const YAML::Node &node, const std::string &name) {
  if (!node.IsSequence() || node.size() != 4) throw std::invalid_argument(name+" must be 4x4");
  Transform result{};
  for (std::size_t row = 0; row < 4; ++row) {
    const auto values = yaml_array<4>(node[row], name);
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
bool host_big_endian() {
  const std::uint16_t word = 1;
  return *reinterpret_cast<const std::uint8_t*>(&word) == 0;
}
template<typename T> T read_scalar(const std::uint8_t *bytes, bool big_endian) {
  std::array<std::uint8_t, sizeof(T)> storage{};
  std::copy(bytes, bytes+sizeof(T), storage.begin());
  if (big_endian != host_big_endian()) std::reverse(storage.begin(), storage.end());
  T value; std::memcpy(&value, storage.data(), sizeof(T)); return value;
}
template<typename T> void write_little(std::uint8_t *bytes, T value) {
  std::array<std::uint8_t, sizeof(T)> storage{};
  std::memcpy(storage.data(), &value, sizeof(T));
  if (host_big_endian()) std::reverse(storage.begin(), storage.end());
  std::copy(storage.begin(), storage.end(), bytes);
}

const Field *required_field(const Cloud &cloud, const std::string &name,
                            std::uint8_t type, std::uint32_t bytes) {
  const Field *result = nullptr;
  for (const auto &field : cloud.fields) {
    if (field.name != name) continue;
    if (result || field.count != 1 || field.datatype != type ||
        static_cast<std::uint64_t>(field.offset)+bytes > cloud.point_step)
      throw std::invalid_argument(name+" must be a unique scalar of the required type within point_step");
    result = &field;
  }
  if (!result) throw std::invalid_argument("Cloud is missing "+name);
  return result;
}
void check_cloud_layout(const Cloud &cloud, std::size_t max_points, bool classified) {
  const auto count = static_cast<std::uint64_t>(cloud.width)*cloud.height;
  if (cloud.header.frame_id != "os_sensor") throw std::invalid_argument("Both input clouds must use os_sensor");
  if (!count || count > max_points || !cloud.point_step ||
      static_cast<std::uint64_t>(cloud.width)*cloud.point_step > cloud.row_step ||
      static_cast<std::uint64_t>(cloud.row_step)*cloud.height != cloud.data.size())
    throw std::invalid_argument("Input cloud is empty or has invalid/bounded point and row layout");
  required_field(cloud, "source_index", Field::UINT32, 4);
  if (classified) {
    required_field(cloud, "classification", Field::UINT8, 1);
    required_field(cloud, "height_above_ground", Field::FLOAT32, 4);
  }
}
std::vector<std::uint32_t> decode_indices(const Cloud &cloud) {
  const auto *field = required_field(cloud, "source_index", Field::UINT32, 4);
  std::vector<std::uint32_t> result;
  result.reserve(static_cast<std::size_t>(cloud.width)*cloud.height);
  for (std::uint32_t row = 0; row < cloud.height; ++row)
    for (std::uint32_t column = 0; column < cloud.width; ++column) {
      const auto *bytes = cloud.data.data()+static_cast<std::size_t>(row)*cloud.row_step+
                          static_cast<std::size_t>(column)*cloud.point_step;
      result.push_back(read_scalar<std::uint32_t>(bytes+field->offset, cloud.is_bigendian));
    }
  return result;
}
std::vector<ClassifiedPoint> decode_classified(const Cloud &cloud) {
  const auto xyz = decode_xyz(cloud);
  const auto indices = decode_indices(cloud);
  const auto *classification = required_field(cloud, "classification", Field::UINT8, 1);
  const auto *height = required_field(cloud, "height_above_ground", Field::FLOAT32, 4);
  std::vector<ClassifiedPoint> result; result.reserve(xyz.size());
  std::size_t index = 0;
  for (std::uint32_t row = 0; row < cloud.height; ++row)
    for (std::uint32_t column = 0; column < cloud.width; ++column, ++index) {
      const auto *bytes = cloud.data.data()+static_cast<std::size_t>(row)*cloud.row_step+
                          static_cast<std::size_t>(column)*cloud.point_step;
      const auto label = bytes[classification->offset];
      const auto above_ground = read_scalar<float>(bytes+height->offset, cloud.is_bigendian);
      if (label < 1 || label > 3 || (label != 1 && !std::isfinite(above_ground)))
        throw std::invalid_argument("Classified cloud contains an invalid label or ground-height evidence");
      result.push_back({xyz[index], indices[index], label, above_ground});
    }
  return result;
}

bool supported_class(const std::string &name) {
  return name == "person" || name == "bicycle" || name == "car" || name == "motorcycle" ||
         name == "bus" || name == "truck";
}
bool vehicle_class(const std::string &name) {
  return name == "car" || name == "motorcycle" || name == "bus" || name == "truck";
}
std::uint16_t class_id(const std::string &name) {
  if (name == "person") return 0;
  if (name == "bicycle") return 1;
  if (name == "car") return 2;
  if (name == "motorcycle") return 3;
  if (name == "bus") return 5;
  if (name == "truck") return 7;
  throw std::invalid_argument("Unexpected fused class name");
}
std::uint32_t palette(const std::string &name) {
  if (name == "person") return 0x00ff00ff;
  if (name == "bicycle") return 0x0000ffff;
  return 0x00ffa500;
}
Cloud empty_semantic(const std_msgs::msg::Header &header) {
  Cloud result; result.header=header; result.height=1; result.point_step=28;
  result.is_bigendian=false; result.is_dense=true;
  for (const auto &entry : std::array<std::pair<const char*,std::uint8_t>,7>{{
      {"x",Field::FLOAT32},{"y",Field::FLOAT32},{"z",Field::FLOAT32},{"rgb",Field::FLOAT32},
      {"source_index",Field::UINT32},{"class_id",Field::UINT16},{"confidence",Field::FLOAT32}}}) {
    Field field; field.name=entry.first; field.datatype=entry.second; field.count=1;
    field.offset=static_cast<std::uint32_t>(result.fields.size())*4;
    result.fields.push_back(field);
  }
  return result;
}
Marker delete_all(const std_msgs::msg::Header &header) {
  Marker result; result.header=header; result.action=Marker::DELETEALL; return result;
}
void marker_color(Marker &marker, std::uint32_t rgb) {
  marker.color.r=static_cast<float>((rgb >> 16)&255)/255.0F;
  marker.color.g=static_cast<float>((rgb >> 8)&255)/255.0F;
  marker.color.b=static_cast<float>(rgb&255)/255.0F;
  marker.color.a=1.0F;
}
geometry_msgs::msg::Point marker_point(double x, double y, double z) {
  geometry_msgs::msg::Point point; point.x=x; point.y=y; point.z=z; return point;
}
}  // namespace

class CameraObjectBEV final : public rclcpp::Node {
 public:
  CameraObjectBEV() : Node("camera_object_bev") {
    camera_frame_=declare_parameter("camera_frame", "camera_front_optical_frame");
    model_name_=declare_parameter("model_name", "camera_detector");
    const auto intrinsics_path=declare_parameter("intrinsics_file", std::string{});
    const auto extrinsics_path=declare_parameter("extrinsics_file", std::string{});
    const auto allow_provisional=declare_parameter("allow_provisional_calibration", false);
    max_pair_delta_s_=declare_parameter("max_pair_delta_s", 0.15);
    stale_after_s_=declare_parameter("stale_after_s", 0.75);
    max_publish_hz_=declare_parameter("max_publish_hz", 20.0);
    const auto max_cloud_points=declare_parameter("max_cloud_points", 500000);
    if (camera_frame_.empty() || model_name_.empty() || intrinsics_path.empty() || extrinsics_path.empty() ||
        !std::isfinite(max_pair_delta_s_) || max_pair_delta_s_ <= 0 || max_pair_delta_s_ > 0.15 ||
        !std::isfinite(stale_after_s_) || stale_after_s_ <= 0 || stale_after_s_ > 0.75 ||
        !std::isfinite(max_publish_hz_) || max_publish_hz_ <= 0 || max_publish_hz_ > 20 ||
        max_cloud_points <= 0 || max_cloud_points > 2000000)
      throw std::invalid_argument("Invalid native calibration, frame, rate, freshness, pairing or point bound");
    max_cloud_points_=static_cast<std::size_t>(max_cloud_points);
    config_.min_confidence=declare_parameter("min_confidence", config_.min_confidence);
    const auto min_support=declare_parameter("min_support_points", static_cast<int>(config_.min_support_points));
    const auto min_pixels=declare_parameter("min_visible_pixels", static_cast<int>(config_.min_visible_pixels));
    const auto min_central=declare_parameter("min_central_points", static_cast<int>(config_.min_central_points));
    if (min_support <= 0 || min_pixels <= 0 || min_central <= 0)
      throw std::invalid_argument("Object support counts must be positive");
    config_.min_support_points=static_cast<std::size_t>(min_support);
    config_.min_visible_pixels=static_cast<std::size_t>(min_pixels);
    config_.min_central_points=static_cast<std::size_t>(min_central);
    config_.cluster_resolution_m=declare_parameter("cluster_resolution_m", config_.cluster_resolution_m);
    config_.max_optical_depth_m=declare_parameter("max_optical_depth_m", config_.max_optical_depth_m);
    config_.max_observed_extent_m=declare_parameter("max_observed_extent_m", config_.max_observed_extent_m);
    config_.max_observed_height_m=declare_parameter("max_observed_height_m", config_.max_observed_height_m);
    config_.ambiguity_depth_m=declare_parameter("ambiguity_depth_m", config_.ambiguity_depth_m);
    config_.ambiguity_support_ratio=declare_parameter("ambiguity_support_ratio", config_.ambiguity_support_ratio);
    validate_fusion_config(config_);
    load_calibration(intrinsics_path, extrinsics_path, allow_provisional);
    const auto detections_topic=declare_parameter("detections_topic", "/camera_lidar_bev/detections_2d");
    const auto info_topic=declare_parameter("camera_info_topic", "/camera/front/camera_info");
    const auto full_topic=declare_parameter("full_points_topic", "/camera_lidar_bev/points_rgb");
    const auto classified_topic=declare_parameter("classified_points_topic", "/camera_lidar_bev/ground_bev_points");
    const auto detections3d_topic=declare_parameter("detections_3d_topic", "/camera_lidar_bev/detections_3d");
    const auto markers_topic=declare_parameter("markers_topic", "/camera_lidar_bev/object_markers");
    const auto vehicle_bev_markers_topic=declare_parameter("vehicle_bev_markers_topic", "/camera_lidar_bev/vehicle_bev_markers");
    const auto semantic_topic=declare_parameter("semantic_points_topic", "/camera_lidar_bev/semantic_points");
    for (const auto &topic : {detections_topic, info_topic, full_topic, classified_topic,
                             detections3d_topic, markers_topic, vehicle_bev_markers_topic, semantic_topic})
      if (topic.empty()) throw std::invalid_argument("Topic names must not be empty");
    detections_pub_=create_publisher<Detections3D>(detections3d_topic, rclcpp::QoS(1).reliable());
    markers_pub_=create_publisher<Markers>(markers_topic, rclcpp::QoS(1).reliable());
    vehicle_bev_markers_pub_=create_publisher<Markers>(vehicle_bev_markers_topic, rclcpp::QoS(1).reliable());
    semantic_pub_=create_publisher<Cloud>(semantic_topic, rclcpp::QoS(1).reliable());
    diagnostics_pub_=create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/camera_lidar_bev/diagnostics", rclcpp::QoS(5));
    detections_sub_=create_subscription<Detections>(detections_topic, rclcpp::QoS(1).reliable(),
      [this](Detections::ConstSharedPtr message) {
        append<Detections>(detections_ring_,std::move(message),16); ++detections_version_;
      });
    info_sub_=create_subscription<Info>(info_topic, rclcpp::SensorDataQoS().keep_last(32),
      [this](Info::ConstSharedPtr message) { append<Info>(infos_, std::move(message),32); });
    full_sub_=create_subscription<Cloud>(full_topic, rclcpp::SensorDataQoS().keep_last(1),
      [this](Cloud::ConstSharedPtr message) { append<Cloud>(full_clouds_, std::move(message)); });
    classified_sub_=create_subscription<Cloud>(classified_topic, rclcpp::SensorDataQoS().keep_last(1),
      [this](Cloud::ConstSharedPtr message) { append<Cloud>(classified_clouds_, std::move(message)); });
    tf_sub_=create_subscription<tf2_msgs::msg::TFMessage>("/tf_static", rclcpp::QoS(10).reliable().transient_local(),
      [this](tf2_msgs::msg::TFMessage::ConstSharedPtr message) { observe_transform(*message); });
    process_timer_=create_wall_timer(std::chrono::duration<double>(1.0/max_publish_hz_), [this]() { process(); });
    diagnostics_timer_=create_wall_timer(std::chrono::seconds(1), [this]() { publish_diagnostics(); });
    RCLCPP_WARN(get_logger(), "Camera labels + observed LiDAR extents use provisional B calibration. No complete 3D box, heading, tracking or TF is inferred.");
  }

 private:
  void load_calibration(const std::string &intrinsics_path, const std::string &extrinsics_path, bool allow_provisional) {
    const auto native=YAML::LoadFile(intrinsics_path);
    if (native["distortion_model"].as<std::string>() != "plumb_bob")
      throw std::invalid_argument("Only native plumb_bob calibration is supported");
    camera_.width=native["image_width"].as<std::uint32_t>();
    camera_.height=native["image_height"].as<std::uint32_t>();
    for (const auto &entry : std::array<std::pair<const char*,std::pair<int,int>>,4>{{
        {"camera_matrix",{3,3}}, {"distortion_coefficients",{1,5}},
        {"rectification_matrix",{3,3}}, {"projection_matrix",{3,4}}}}) {
      const auto matrix=native[entry.first];
      if (matrix["rows"].as<int>() != entry.second.first || matrix["cols"].as<int>() != entry.second.second)
        throw std::invalid_argument(std::string("Unexpected YAML matrix layout for ")+entry.first);
    }
    camera_.k=yaml_array<9>(native["camera_matrix"]["data"], "K");
    camera_.d=yaml_array<5>(native["distortion_coefficients"]["data"], "D");
    expected_r_=yaml_array<9>(native["rectification_matrix"]["data"], "R");
    expected_p_=yaml_array<12>(native["projection_matrix"]["data"], "P");
    validate_camera(camera_);
    const std::array<double,9> identity={1,0,0,0,1,0,0,0,1};
    const std::array<double,12> native_p={camera_.k[0],0,camera_.k[2],0,0,camera_.k[4],camera_.k[5],0,0,0,1,0};
    if (!close_arrays(expected_r_, identity) || !close_arrays(expected_p_, native_p))
      throw std::invalid_argument("Raw-image calibration must use identity R and native [K|0] P");
    const auto candidate=YAML::LoadFile(extrinsics_path);
    if (candidate["units"].as<std::string>() != "m" || candidate["direction"].as<std::string>() != "lidar_to_camera" ||
        candidate["selected_hypothesis"].as<std::string>() != "B")
      throw std::invalid_argument("Extrinsics must be the selected B candidate in meters");
    if (!allow_provisional || candidate["calibration_status"].as<std::string>() != "external_candidate_unvalidated")
      throw std::invalid_argument("This preview requires explicit allow_provisional_calibration for the unvalidated B candidate");
    const auto external=candidate["transforms"]["T_camera_from_os_sensor"];
    if (external["source_frame"].as<std::string>() != "os_sensor" ||
        external["target_frame"].as<std::string>() != camera_frame_)
      throw std::invalid_argument("Candidate frames disagree with os_sensor and configured optical frame");
    camera_from_sensor_=yaml_transform(external["matrix"], "T_camera_from_os_sensor");
    const auto observed=candidate["observed_driver_transform"];
    if (observed["source_frame"].as<std::string>() != "os_lidar" || observed["target_frame"].as<std::string>() != "os_sensor")
      throw std::invalid_argument("Candidate driver transform has unexpected frames");
    expected_sensor_from_lidar_=yaml_transform(observed["matrix"], "T_os_sensor_from_os_lidar");
    const auto composed=candidate["transforms"]["T_camera_from_os_lidar"];
    if (composed["source_frame"].as<std::string>() != "os_lidar" || composed["target_frame"].as<std::string>() != camera_frame_ ||
        !close_arrays(compose(camera_from_sensor_, expected_sensor_from_lidar_),
                      yaml_transform(composed["matrix"], "T_camera_from_os_lidar")))
      throw std::invalid_argument("Candidate stored B composition is inconsistent");
  }
  std::string camera_info_error(const Info &info) const {
    if (info.header.frame_id != camera_frame_ || info.width != camera_.width || info.height != camera_.height)
      return "CameraInfo frame/dimensions disagree with native calibration";
    if (info.distortion_model != "plumb_bob" || !close_arrays(info.k, camera_.k) || !close_arrays(info.d, camera_.d) ||
        !close_arrays(info.r, expected_r_) || !close_arrays(info.p, expected_p_))
      return "CameraInfo full K/D/R/P does not match native calibration";
    if (info.binning_x > 1 || info.binning_y > 1 || info.roi.x_offset || info.roi.y_offset || info.roi.do_rectify ||
        (info.roi.width && info.roi.width != camera_.width) || (info.roi.height && info.roi.height != camera_.height))
      return "CameraInfo ROI/binning changes the native image";
    return {};
  }
  void observe_transform(const tf2_msgs::msg::TFMessage &message) {
    for (const auto &record : message.transforms) {
      if (record.header.frame_id != "os_sensor" || record.child_frame_id != "os_lidar") continue;
      const auto &q=record.transform.rotation; const auto &p=record.transform.translation;
      const auto norm2=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
      if (!std::isfinite(norm2) || std::abs(norm2-1.0) > 1e-6) {
        tf_conflict_=true; tf_error_="Driver quaternion is not finite/unit"; continue;
      }
      const double norm=std::sqrt(norm2), x=q.x/norm, y=q.y/norm, z=q.z/norm, w=q.w/norm;
      const Transform transform={1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),p.x,
                                 2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),p.y,
                                 2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y),p.z,
                                 0,0,0,1};
      try { validate_transform(transform); }
      catch (const std::exception &error) { tf_conflict_=true; tf_error_=error.what(); continue; }
      if (!close_arrays(transform, expected_sensor_from_lidar_)) {
        tf_conflict_=true; tf_error_="Observed driver TF conflicts with selected B evidence"; continue;
      }
      tf_received_=true;
    }
  }
  template<typename Message> bool fresh(const Received<Message> &input, Clock::time_point receipt_now,
                                       std::int64_t ros_now) const {
    if (!input.message || std::chrono::duration<double>(receipt_now-input.receipt).count() > stale_after_s_) return false;
    const auto &header=input.message->header;
    const auto stamp=stamp_ns(header);
    const auto age=static_cast<double>(ros_now-stamp)/1e9;
    return header.stamp.nanosec < 1000000000U && stamp > 0 && age >= -0.05 && age <= stale_after_s_;
  }
  std_msgs::msg::Header output_header(const std_msgs::msg::Header &source) const {
    auto header=source; header.frame_id="os_sensor"; return header;
  }
  void publish_empty(const std_msgs::msg::Header &header) {
    Detections3D detections; detections.header=header; detections_pub_->publish(std::move(detections));
    Markers markers; markers.markers.push_back(delete_all(header));
    markers_pub_->publish(markers); vehicle_bev_markers_pub_->publish(std::move(markers));
    semantic_pub_->publish(empty_semantic(header));
    last_candidate_count_=0; last_semantic_points_=0;
  }
  void unavailable(const std::string &reason) {
    // Clear once on each transition and whenever a live result stops being valid.
    if (has_live_output_ || !initial_clear_sent_ || reason != current_error_) {
      std_msgs::msg::Header header; header.frame_id="os_sensor"; header.stamp=now();
      publish_empty(header); ++cleared_outputs_; initial_clear_sent_=true;
    }
    has_live_output_=false; current_error_=reason;
  }
  std::vector<CameraDetection> decode_detections(const Detections &message) const {
    if (message.header.frame_id != camera_frame_) throw std::invalid_argument("Detection image frame must match native optical frame");
    if (message.detections.size() > config_.max_detections)
      throw std::invalid_argument("Detection array exceeds bounded maximum");
    std::vector<CameraDetection> result; result.reserve(message.detections.size());
    for (const auto &detection : message.detections) {
      if (!same_header(detection.header, message.header))
        throw std::invalid_argument("Detection child header must exactly match its source image header");
      const auto &bbox=detection.bbox;
      if (!std::isfinite(bbox.center.position.x) || !std::isfinite(bbox.center.position.y) ||
          !std::isfinite(bbox.center.theta) || std::abs(bbox.center.theta) > 1e-8 ||
          !std::isfinite(bbox.size_x) || !std::isfinite(bbox.size_y) || bbox.size_x <= 0 || bbox.size_y <= 0)
        throw std::invalid_argument("Detection boxes must be finite axis-aligned native-image rectangles");
      const vision_msgs::msg::ObjectHypothesis *best=nullptr;
      for (const auto &hypothesis : detection.results) {
        const auto &item=hypothesis.hypothesis;
        if (!std::isfinite(item.score) || item.score < 0 || item.score > 1)
          throw std::invalid_argument("Camera confidence must be finite in [0,1]");
        if (supported_class(item.class_id) && (!best || item.score > best->score)) best=&item;
      }
      if (!best || best->score < config_.min_confidence) continue;
      const double x_min=bbox.center.position.x-bbox.size_x/2, x_max=bbox.center.position.x+bbox.size_x/2;
      const double y_min=bbox.center.position.y-bbox.size_y/2, y_max=bbox.center.position.y+bbox.size_y/2;
      if (x_min < -1e-6 || y_min < -1e-6 || x_max > camera_.width+1e-6 || y_max > camera_.height+1e-6)
        throw std::invalid_argument("Detection box extends outside native calibrated image");
      result.push_back({best->class_id, best->score, std::max(0.0,x_min), std::max(0.0,y_min),
                        std::min(static_cast<double>(camera_.width),x_max),
                        std::min(static_cast<double>(camera_.height),y_max)});
    }
    return result;
  }
  PreparedOutputs prepare_candidates(const FusionResult &result, const std_msgs::msg::Header &header) {
    PreparedOutputs output;
    auto &detections=output.detections; detections.header=header;
    auto &markers=output.markers; markers.markers.push_back(delete_all(header));
    auto &vehicle_bev_markers=output.vehicle_bev_markers; vehicle_bev_markers.markers.push_back(delete_all(header));
    auto &semantic=output.semantic; semantic=empty_semantic(header);
    std::size_t point_count=0;
    for (const auto &candidate : result.candidates) point_count+=candidate.classified_indices.size();
    if (point_count > std::numeric_limits<std::uint32_t>::max()/semantic.point_step)
      throw std::invalid_argument("Semantic cloud exceeds row_step bound");
    semantic.width=static_cast<std::uint32_t>(point_count);
    semantic.row_step=semantic.width*semantic.point_step;
    semantic.data.resize(semantic.row_step,0);
    std::unordered_set<std::uint32_t> seen; seen.reserve(point_count);
    std::size_t destination=0;
    for (std::size_t i=0; i < result.candidates.size(); ++i) {
      const auto &candidate=result.candidates[i];
      vision_msgs::msg::Detection3D detection; detection.header=header;
      detection.bbox.center.position.x=candidate.center.x;
      detection.bbox.center.position.y=candidate.center.y;
      detection.bbox.center.position.z=candidate.center.z;
      detection.bbox.center.orientation.w=1.0;
      detection.bbox.size.x=candidate.size.x; detection.bbox.size.y=candidate.size.y; detection.bbox.size.z=candidate.size.z;
      vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
      hypothesis.hypothesis.class_id=candidate.class_name; hypothesis.hypothesis.score=candidate.confidence;
      hypothesis.pose.pose=detection.bbox.center;
      detection.results.push_back(std::move(hypothesis));
      // Empty IDs explicitly avoid implying tracking consistency.
      detections.detections.push_back(std::move(detection));
      const auto rgb=palette(candidate.class_name);
      Marker box; box.header=header; box.ns="observed_object_extent"; box.id=static_cast<int>(i*2);
      box.type=Marker::LINE_LIST; box.action=Marker::ADD; box.pose.orientation.w=1.0;
      box.scale.x=0.035; box.lifetime=rclcpp::Duration::from_seconds(0.5); marker_color(box,rgb);
      std::array<geometry_msgs::msg::Point,8> corners;
      for (int corner=0; corner < 8; ++corner)
        corners[corner]=marker_point(candidate.center.x+(corner&1 ? 0.5 : -0.5)*candidate.size.x,
                                    candidate.center.y+(corner&2 ? 0.5 : -0.5)*candidate.size.y,
                                    candidate.center.z+(corner&4 ? 0.5 : -0.5)*candidate.size.z);
      for (int corner=0; corner < 8; ++corner) for (int axis=0; axis < 3; ++axis)
        if (!(corner&(1<<axis))) {
          box.points.push_back(corners[corner]); box.points.push_back(corners[corner|(1<<axis)]);
        }
      markers.markers.push_back(std::move(box));
      Marker label; label.header=header; label.ns="camera_object_labels"; label.id=static_cast<int>(i*2+1);
      label.type=Marker::TEXT_VIEW_FACING; label.action=Marker::ADD; label.pose.orientation.w=1.0;
      label.pose.position=marker_point(candidate.center.x,candidate.center.y,candidate.center.z+candidate.size.z/2+0.25);
      label.scale.z=0.3; label.lifetime=rclcpp::Duration::from_seconds(0.5); marker_color(label,rgb);
      std::ostringstream text; text << candidate.class_name << ' ' << std::fixed << std::setprecision(2) << candidate.confidence;
      label.text=text.str();
      if (vehicle_class(candidate.class_name)) {
        // Flat display plane in os_sensor; XY bounds remain the observed returns.
        // This does not infer a ground height, complete vehicle size or heading.
        const double xmin=candidate.center.x-candidate.size.x/2;
        const double xmax=candidate.center.x+candidate.size.x/2;
        const double ymin=candidate.center.y-candidate.size.y/2;
        const double ymax=candidate.center.y+candidate.size.y/2;
        Marker footprint; footprint.header=header; footprint.ns="observed_vehicle_bev_extent";
        footprint.id=static_cast<int>(i*2); footprint.type=Marker::LINE_STRIP;
        footprint.action=Marker::ADD; footprint.pose.orientation.w=1.0;
        footprint.scale.x=0.07; footprint.lifetime=rclcpp::Duration::from_seconds(0.5);
        marker_color(footprint,rgb);
        footprint.points={marker_point(xmin,ymin,0.15), marker_point(xmax,ymin,0.15),
                          marker_point(xmax,ymax,0.15), marker_point(xmin,ymax,0.15),
                          marker_point(xmin,ymin,0.15)};
        vehicle_bev_markers.markers.push_back(std::move(footprint));
        Marker footprint_label=label; footprint_label.ns="vehicle_bev_labels";
        footprint_label.pose.position=marker_point(candidate.center.x,candidate.center.y,0.17);
        footprint_label.scale.z=0.35;
        vehicle_bev_markers.markers.push_back(std::move(footprint_label));
      }
      markers.markers.push_back(std::move(label));
      for (const auto index : candidate.classified_indices) {
        if (index >= cached_classified_.size()) throw std::invalid_argument("Fusion returned an out-of-range classified row");
        const auto &point=cached_classified_[index];
        if (!seen.insert(point.source_index).second) throw std::invalid_argument("Fusion assigned one return to multiple objects");
        auto *bytes=semantic.data.data()+destination++*semantic.point_step;
        write_little(bytes,static_cast<float>(point.point.x));
        write_little(bytes+4,static_cast<float>(point.point.y));
        write_little(bytes+8,static_cast<float>(point.point.z));
        write_little(bytes+12,rgb); write_little(bytes+16,point.source_index);
        write_little(bytes+20,class_id(candidate.class_name));
        write_little(bytes+24,static_cast<float>(candidate.confidence));
      }
    }
    return output;
  }
  void mark_processed(const Detections &message, double processing_ms) {
    processed_version_=detections_version_;
    last_processed_stamp_=std::max(last_processed_stamp_,stamp_ns(message.header));
    last_image_stamp_=stamp_ns(message.header); last_processing_ms_=processing_ms;
    last_output_receipt_=Clock::now(); has_live_output_=true; current_error_.clear(); ++processed_arrays_;
  }
  void process() {
    const auto start=Clock::now(); const auto ros_now=now().nanoseconds();
    if (tf_conflict_) { unavailable(tf_error_); return; }
    if (!tf_received_) { unavailable("Waiting for actual driver static TF"); return; }
    if (detections_ring_.empty() || !fresh(detections_ring_.back(),start,ros_now)) {
      unavailable("Camera detections are stale or unavailable"); return;
    }
    const auto &latest=detections_ring_.back();
    if (has_live_output_ && (std::chrono::duration<double>(start-last_output_receipt_).count() > stale_after_s_ ||
        static_cast<double>(ros_now-last_image_stamp_)/1e9 > stale_after_s_ ||
        (last_cloud_stamp_ > 0 && static_cast<double>(ros_now-last_cloud_stamp_)/1e9 > stale_after_s_)))
      unavailable("Last object output became stale");
    const auto latest_stamp=stamp_ns(latest.message->header);
    if (detections_version_ != checked_latest_version_) {
      checked_latest_version_=detections_version_;
      if (last_seen_latest_stamp_ && latest_stamp <= last_seen_latest_stamp_) {
        latest_header_invalid_=true;
      } else {
        last_seen_latest_stamp_=latest_stamp; latest_header_invalid_=false;
      }
    }
    if (latest_header_invalid_) {
      unavailable("Repeated or nonmonotonic camera detection timestamp"); return;
    }
    try {
      const auto latest_boxes=decode_detections(*latest.message);
      auto find_info=[this,start,ros_now](const Detections &message) -> const Received<Info>* {
        for (auto item=infos_.rbegin(); item != infos_.rend(); ++item) {
          if (!same_header(item->message->header,message.header) || !fresh(*item,start,ros_now)) continue;
          const auto error=camera_info_error(*item->message);
          if (!error.empty()) throw std::invalid_argument(error);
          return &*item;
        }
        return nullptr;
      };
      // Newest empty camera evidence clears immediately. It never waits for a
      // LiDAR scan or decodes the large point payloads.
      if (latest_boxes.empty()) {
        if (latest_stamp <= last_empty_stamp_) return;
        const auto *info=find_info(*latest.message);
        if (!info) { unavailable("Waiting for exact native CameraInfo/detection image header pair"); return; }
        last_detection_count_=latest.message->detections.size(); last_supported_boxes_=0;
        last_source_age_s_=static_cast<double>(ros_now-latest_stamp)/1e9;
        last_result_=FusionResult{}; last_decode_ms_=0; last_fusion_ms_=0; last_pair_delta_s_=0; last_cloud_stamp_=0;
        publish_empty(output_header(latest.message->header));
        last_empty_stamp_=latest_stamp;
        mark_processed(*latest.message,std::chrono::duration<double,std::milli>(Clock::now()-start).count());
        return;
      }
      if (full_clouds_.empty() || classified_clouds_.empty() ||
          !fresh(full_clouds_.back(),start,ros_now) || !fresh(classified_clouds_.back(),start,ros_now)) {
        unavailable("Waiting for fresh full geometry and classified cloud streams"); return;
      }
      check_cloud_layout(*full_clouds_.back().message,max_cloud_points_,false);
      check_cloud_layout(*classified_clouds_.back().message,max_cloud_points_,true);
      const Received<Cloud> *classified=nullptr, *full=nullptr;
      const Received<Detections> *selected=nullptr;
      const Received<Info> *info=nullptr;
      std::vector<CameraDetection> boxes;
      double best_delta=std::numeric_limits<double>::infinity();
      // Anchor the newest classified scan, then select its closest fresh camera
      // image. Images may support multiple distinct scans when their timestamps
      // are the nearest match; each scan itself is published at most once.
      // A newer image 145 ms away must not displace a 5 ms match.
      classified=&classified_clouds_.back();
      const auto cloud_stamp=stamp_ns(classified->message->header);
      for (auto source=full_clouds_.rbegin(); source != full_clouds_.rend(); ++source)
        if (same_header(source->message->header,classified->message->header) && fresh(*source,start,ros_now)) {
          full=&*source; break;
        }
      if (!full) {
        unavailable("Waiting for full geometry with exactly the newest classified scan header"); return;
      }
      if (cloud_stamp <= last_used_cloud_stamp_) return;
      for (auto detection=detections_ring_.rbegin(); detection != detections_ring_.rend(); ++detection) {
        const auto detection_stamp=stamp_ns(detection->message->header);
        if (detection_stamp <= last_empty_stamp_ || !fresh(*detection,start,ros_now)) continue;
        const double delta=std::abs(static_cast<double>(cloud_stamp-detection_stamp)/1e9);
        // Reverse iteration gives the freshest image precedence on a tie.
        if (delta > max_pair_delta_s_ || delta >= best_delta) continue;
        auto detection_boxes=decode_detections(*detection->message);
        if (detection_boxes.empty()) continue;
        const auto *candidate_info=find_info(*detection->message);
        if (!candidate_info) continue;
        selected=&*detection; info=candidate_info; boxes=std::move(detection_boxes); best_delta=delta;
      }
      if (!selected || !classified || !full || !info) {
        unavailable("Waiting for exact full/classified cloud pair within a fresh image time limit"); return;
      }
      const auto &message=*selected->message;
      const auto detection_stamp=stamp_ns(message.header);
      last_detection_count_=message.detections.size(); last_supported_boxes_=boxes.size();
      last_source_age_s_=static_cast<double>(ros_now-detection_stamp)/1e9;
      const auto decode_start=Clock::now();
      if (cached_full_message_ != full->message || cached_classified_message_ != classified->message) {
        check_cloud_layout(*full->message,max_cloud_points_,false);
        check_cloud_layout(*classified->message,max_cloud_points_,true);
        cached_full_=decode_xyz(*full->message); cached_indices_=decode_indices(*full->message);
        cached_classified_=decode_classified(*classified->message);
        cached_full_message_=full->message; cached_classified_message_=classified->message;
      }
      last_decode_ms_=std::chrono::duration<double,std::milli>(Clock::now()-decode_start).count();
      const auto fusion_start=Clock::now();
      const auto result=fuse_camera_objects(cached_full_,cached_indices_,cached_classified_,boxes,camera_,camera_from_sensor_,config_);
      last_fusion_ms_=std::chrono::duration<double,std::milli>(Clock::now()-fusion_start).count();
      auto output=prepare_candidates(result,output_header(classified->message->header));
      const auto complete=Clock::now(); const auto complete_ros=now().nanoseconds();
      if (!fresh(*selected,complete,complete_ros) || !fresh(*info,complete,complete_ros) ||
          !fresh(*classified,complete,complete_ros) || !fresh(*full,complete,complete_ros)) {
        unavailable("Object inputs became stale during association"); return;
      }
      last_candidate_count_=result.candidates.size(); last_semantic_points_=output.semantic.width;
      detections_pub_->publish(std::move(output.detections)); markers_pub_->publish(std::move(output.markers));
      vehicle_bev_markers_pub_->publish(std::move(output.vehicle_bev_markers));
      semantic_pub_->publish(std::move(output.semantic));
      last_result_=result; last_result_.candidates.clear();
      last_pair_delta_s_=best_delta; last_cloud_stamp_=stamp_ns(classified->message->header);
      last_used_cloud_stamp_=last_cloud_stamp_;
      last_source_age_s_=static_cast<double>(complete_ros-detection_stamp)/1e9;
      mark_processed(message,std::chrono::duration<double,std::milli>(Clock::now()-start).count());
    } catch (const std::exception &error) {
      processed_version_=detections_version_; last_processed_stamp_=std::max(last_processed_stamp_,latest_stamp);
      unavailable(error.what());
    }
  }
  static void add(Status &status, const std::string &key, const std::string &value) {
    diagnostic_msgs::msg::KeyValue entry; entry.key=key; entry.value=value; status.values.push_back(std::move(entry));
  }
  void publish_diagnostics() {
    diagnostic_msgs::msg::DiagnosticArray message; message.header.stamp=now();
    Status status; status.name="camera_object_bev"; status.hardware_id="front_camera+ouster"; status.level=Status::WARN;
    status.message=current_error_.empty() ? "Camera labels and observed extents active; geometry/synchronization provisional" : current_error_;
    add(status,"model",model_name_); add(status,"calibration_status","external_candidate_unvalidated");
    add(status,"accuracy_validated","false"); add(status,"synchronization_validated","false");
    add(status,"tf_published","false"); add(status,"output_frame","os_sensor");
    add(status,"observed_extent","true"); add(status,"heading_estimated","false"); add(status,"tracking_enabled","false");
    add(status,"occupancy_modified","false"); add(status,"driver_tf_observed",tf_received_ ? "true" : "false");
    add(status,"publishing_fresh_output",has_live_output_ ? "true" : "false");
    add(status,"received_arrays",std::to_string(detections_version_)); add(status,"processed_arrays",std::to_string(processed_arrays_));
    add(status,"cleared_outputs",std::to_string(cleared_outputs_)); add(status,"publication_cap_hz",std::to_string(max_publish_hz_));
    add(status,"camera_detections",std::to_string(last_detection_count_)); add(status,"supported_camera_boxes",std::to_string(last_supported_boxes_));
    add(status,"object_candidates",std::to_string(last_candidate_count_)); add(status,"semantic_points",std::to_string(last_semantic_points_));
    add(status,"visible_geometry_points",std::to_string(last_result_.visible_geometry_points));
    add(status,"supported_obstacle_points",std::to_string(last_result_.supported_obstacle_points));
    add(status,"rejected_invalid_detections",std::to_string(last_result_.rejected_invalid_detections));
    add(status,"rejected_unsupported",std::to_string(last_result_.rejected_unsupported));
    add(status,"rejected_ambiguous",std::to_string(last_result_.rejected_ambiguous));
    add(status,"header_delta_s",std::to_string(last_pair_delta_s_)); add(status,"source_age_at_processing_s",std::to_string(last_source_age_s_));
    add(status,"image_header_ns",std::to_string(last_image_stamp_)); add(status,"cloud_header_ns",std::to_string(last_cloud_stamp_));
    add(status,"processing_ms",std::to_string(last_processing_ms_)); add(status,"decode_ms",std::to_string(last_decode_ms_));
    add(status,"fusion_ms",std::to_string(last_fusion_ms_));
    if (processed_arrays_) add(status,"last_output_receipt_age_s",std::to_string(std::chrono::duration<double>(Clock::now()-last_output_receipt_).count()));
    message.status.push_back(std::move(status)); diagnostics_pub_->publish(std::move(message));
  }

  Camera camera_;
  Transform camera_from_sensor_{}, expected_sensor_from_lidar_{};
  std::array<double,9> expected_r_{};
  std::array<double,12> expected_p_{};
  FusionConfig config_;
  std::string camera_frame_, model_name_, tf_error_, current_error_{"Waiting for inputs"};
  double max_pair_delta_s_{}, stale_after_s_{}, max_publish_hz_{};
  double last_pair_delta_s_{}, last_source_age_s_{}, last_processing_ms_{}, last_decode_ms_{}, last_fusion_ms_{};
  std::size_t max_cloud_points_{}, last_detection_count_{}, last_supported_boxes_{}, last_candidate_count_{}, last_semantic_points_{};
  bool tf_received_{false}, tf_conflict_{false}, has_live_output_{false}, initial_clear_sent_{false}, latest_header_invalid_{false};
  std::uint64_t detections_version_{}, processed_version_{}, checked_latest_version_{}, processed_arrays_{}, cleared_outputs_{};
  std::int64_t last_processed_stamp_{}, last_image_stamp_{}, last_cloud_stamp_{}, last_used_cloud_stamp_{},
               last_seen_latest_stamp_{}, last_empty_stamp_{};
  Clock::time_point last_output_receipt_{};
  std::deque<Received<Detections>> detections_ring_;
  std::deque<Received<Info>> infos_;
  std::deque<Received<Cloud>> full_clouds_, classified_clouds_;
  Cloud::ConstSharedPtr cached_full_message_, cached_classified_message_;
  std::vector<XYZ> cached_full_;
  std::vector<std::uint32_t> cached_indices_;
  std::vector<ClassifiedPoint> cached_classified_;
  FusionResult last_result_;
  rclcpp::Subscription<Detections>::SharedPtr detections_sub_;
  rclcpp::Subscription<Info>::SharedPtr info_sub_;
  rclcpp::Subscription<Cloud>::SharedPtr full_sub_, classified_sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_;
  rclcpp::Publisher<Detections3D>::SharedPtr detections_pub_;
  rclcpp::Publisher<Markers>::SharedPtr markers_pub_, vehicle_bev_markers_pub_;
  rclcpp::Publisher<Cloud>::SharedPtr semantic_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
  rclcpp::TimerBase::SharedPtr process_timer_, diagnostics_timer_;
};
}  // namespace bev_visualization

int main(int argc, char **argv) {
  rclcpp::init(argc,argv);
  try { rclcpp::spin(std::make_shared<bev_visualization::CameraObjectBEV>()); }
  catch (const std::exception &error) {
    RCLCPP_FATAL(rclcpp::get_logger("camera_object_bev"), "%s", error.what());
    rclcpp::shutdown(); return 1;
  }
  rclcpp::shutdown(); return 0;
}
