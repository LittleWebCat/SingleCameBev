#include "bev_geometry/ground.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"

namespace bev_geometry {
namespace {
using Cloud = sensor_msgs::msg::PointCloud2;
using Field = sensor_msgs::msg::PointField;
using SteadyClock = std::chrono::steady_clock;
using Status = diagnostic_msgs::msg::DiagnosticStatus;

std::int64_t stamp_ns(const std_msgs::msg::Header &header) {
  return static_cast<std::int64_t>(header.stamp.sec) * 1000000000LL + header.stamp.nanosec;
}
bool host_big_endian() {
  const std::uint16_t value = 1;
  return *reinterpret_cast<const std::uint8_t *>(&value) == 0;
}
template<typename T> T read_scalar(const std::uint8_t *bytes, bool big_endian) {
  std::array<std::uint8_t, sizeof(T)> storage{};
  std::copy(bytes, bytes + sizeof(T), storage.begin());
  if (big_endian != host_big_endian()) std::reverse(storage.begin(), storage.end());
  T result;
  std::memcpy(&result, storage.data(), sizeof(T));
  return result;
}
template<typename T> void write_little(std::uint8_t *bytes, T value) {
  std::array<std::uint8_t, sizeof(T)> storage{};
  std::memcpy(storage.data(), &value, sizeof(T));
  if (host_big_endian()) std::reverse(storage.begin(), storage.end());
  std::copy(storage.begin(), storage.end(), bytes);
}

struct InputPoints {
  std::vector<Point> xyz;
  std::vector<std::uint32_t> source_index;
};

// Preserve the original organized scan index, including row padding and endian
// layout. The RGB precursor already supplies source_index; generic XYZ fixtures
// receive their row-major input index instead.
InputPoints decode_cloud(const Cloud &cloud) {
  const auto count = static_cast<std::uint64_t>(cloud.width) * cloud.height;
  if (!count || !cloud.point_step || count > 2000000U ||
      static_cast<std::uint64_t>(cloud.width) * cloud.point_step > cloud.row_step ||
      static_cast<std::uint64_t>(cloud.row_step) * cloud.height != cloud.data.size()) {
    throw std::invalid_argument("Invalid PointCloud2 point/row/payload layout");
  }
  std::array<const Field *, 3> fields{};
  const Field *source_index = nullptr;
  for (const auto &field : cloud.fields) {
    const int axis = field.name == "x" ? 0 : field.name == "y" ? 1 : field.name == "z" ? 2 : -1;
    if (axis >= 0) {
      if (fields[axis] || field.count != 1 ||
          (field.datatype != Field::FLOAT32 && field.datatype != Field::FLOAT64)) {
        throw std::invalid_argument("XYZ fields must be unique scalar FLOAT32/FLOAT64");
      }
      const auto size = field.datatype == Field::FLOAT32 ? 4U : 8U;
      if (static_cast<std::uint64_t>(field.offset) + size > cloud.point_step) {
        throw std::invalid_argument("XYZ field exceeds point_step");
      }
      fields[axis] = &field;
    } else if (field.name == "source_index") {
      if (source_index || field.count != 1 || field.datatype != Field::UINT32 ||
          static_cast<std::uint64_t>(field.offset) + 4U > cloud.point_step) {
        throw std::invalid_argument("source_index must be a unique scalar UINT32 inside point_step");
      }
      source_index = &field;
    }
  }
  if (!fields[0] || !fields[1] || !fields[2]) throw std::invalid_argument("Point cloud is missing XYZ");
  InputPoints output;
  output.xyz.reserve(count);
  output.source_index.reserve(count);
  for (std::uint32_t row = 0; row < cloud.height; ++row) {
    for (std::uint32_t column = 0; column < cloud.width; ++column) {
      const auto *bytes = cloud.data.data() + static_cast<std::size_t>(row) * cloud.row_step +
        static_cast<std::size_t>(column) * cloud.point_step;
      std::array<double, 3> xyz{};
      for (int axis = 0; axis < 3; ++axis) {
        const auto *field = fields[axis];
        xyz[axis] = field->datatype == Field::FLOAT32 ?
          static_cast<double>(read_scalar<float>(bytes + field->offset, cloud.is_bigendian)) :
          read_scalar<double>(bytes + field->offset, cloud.is_bigendian);
        // Classified outputs are FLOAT32. Reject finite FLOAT64 values that
        // would turn into infinity during serialization, retaining their index
        // in the Invalid count rather than publishing nonfinite coordinates.
        if (std::isfinite(xyz[axis]) && std::abs(xyz[axis]) > std::numeric_limits<float>::max()) {
          xyz[axis] = std::numeric_limits<double>::quiet_NaN();
        }
      }
      output.xyz.push_back({xyz[0], xyz[1], xyz[2]});
      output.source_index.push_back(source_index ?
        read_scalar<std::uint32_t>(bytes + source_index->offset, cloud.is_bigendian) :
        static_cast<std::uint32_t>(output.xyz.size() - 1));
    }
  }
  return output;
}

Cloud empty_cloud(const std_msgs::msg::Header &header) {
  Cloud cloud;
  cloud.header = header;
  cloud.height = 1;
  cloud.is_bigendian = false;
  cloud.point_step = 28;
  cloud.is_dense = false;  // Unknown ground heights intentionally carry NaN.
  for (const auto &item : std::array<std::pair<const char *, std::uint8_t>, 7>{{
    {"x", Field::FLOAT32}, {"y", Field::FLOAT32}, {"z", Field::FLOAT32},
    {"rgb", Field::FLOAT32}, {"source_index", Field::UINT32},
    {"height_above_ground", Field::FLOAT32}, {"classification", Field::UINT8}}}) {
    Field field;
    field.name = item.first;
    field.datatype = item.second;
    field.offset = static_cast<std::uint32_t>(cloud.fields.size()) * 4U;
    field.count = 1;
    cloud.fields.push_back(std::move(field));
  }
  return cloud;
}

Cloud labeled_cloud(const InputPoints &input, const Result &result, Label wanted,
                    const std_msgs::msg::Header &header, bool combined = false) {
  Cloud cloud = empty_cloud(header);
  const auto included = [wanted, combined](Label label) {
    return combined ? (label == Label::Ground || label == Label::Obstacle || label == Label::Unknown) : label == wanted;
  };
  const auto count = static_cast<std::size_t>(std::count_if(result.labels.begin(), result.labels.end(), included));
  if (count > std::numeric_limits<std::uint32_t>::max() / cloud.point_step) {
    throw std::invalid_argument("Classified cloud exceeds UINT32 row_step");
  }
  cloud.width = static_cast<std::uint32_t>(count);
  cloud.row_step = cloud.width * cloud.point_step;
  cloud.data.resize(cloud.row_step, 0);
  std::size_t destination = 0;
  for (std::size_t i = 0; i < result.labels.size(); ++i) {
    const auto label = result.labels[i];
    if (!included(label)) continue;
    const std::uint32_t color = label == Label::Ground ? 0x0000ff00U :
      label == Label::Obstacle ? 0x00ff3030U : 0x00999999U;
    const auto &point = input.xyz[i];
    auto *bytes = cloud.data.data() + destination++ * cloud.point_step;
    write_little(bytes, static_cast<float>(point.x));
    write_little(bytes + 4, static_cast<float>(point.y));
    write_little(bytes + 8, static_cast<float>(point.z));
    write_little(bytes + 12, color);
    write_little(bytes + 16, input.source_index[i]);
    write_little(bytes + 20, static_cast<float>(result.heights_above_ground[i]));
    bytes[24] = static_cast<std::uint8_t>(label);
  }
  return cloud;
}

nav_msgs::msg::OccupancyGrid occupancy_message(const bev_core::Grid &grid,
                                              const std_msgs::msg::Header &header) {
  nav_msgs::msg::OccupancyGrid message;
  message.header = header;
  message.info.map_load_time = header.stamp;
  message.info.resolution = static_cast<float>(grid.spec().resolution_m);
  message.info.width = static_cast<std::uint32_t>(grid.forward_cells());
  message.info.height = static_cast<std::uint32_t>(grid.lateral_cells());
  message.info.origin.position.x = grid.spec().x_min_m;
  message.info.origin.position.y = grid.spec().y_min_m;
  message.info.origin.orientation.w = 1.0;
  message.data.resize(grid.cell_count(), -1);
  // bev_core varies lateral fastest, while ROS OccupancyGrid varies x fastest.
  for (std::size_t x = 0; x < grid.forward_cells(); ++x) {
    for (std::size_t y = 0; y < grid.lateral_cells(); ++y) {
      message.data[y * grid.forward_cells() + x] =
        static_cast<std::int8_t>(grid.cell(bev_core::CellIndex{x, y}).occupancy);
    }
  }
  return message;
}

sensor_msgs::msg::Image occupancy_image(const bev_core::Grid &grid,
                                      const std_msgs::msg::Header &header) {
  sensor_msgs::msg::Image message;
  message.header = header;
  message.width = static_cast<std::uint32_t>(grid.lateral_cells());
  message.height = static_cast<std::uint32_t>(grid.forward_cells());
  message.encoding = "rgb8";
  message.is_bigendian = false;
  message.step = message.width * 3U;
  message.data.resize(static_cast<std::size_t>(message.step) * message.height);
  // Each pixel is one observed grid cell: forward is up and sensor-left is
  // image-left. Keep unknown cells unknown, including occlusion behind objects.
  for (std::size_t forward = 0; forward < grid.forward_cells(); ++forward) {
    for (std::size_t lateral = 0; lateral < grid.lateral_cells(); ++lateral) {
      const auto value = grid.cell(bev_core::CellIndex{forward, lateral}).occupancy;
      const std::array<std::uint8_t, 3> color = value == bev_core::Occupancy::Free ?
        std::array<std::uint8_t, 3>{50, 170, 70} : value == bev_core::Occupancy::Occupied ?
        std::array<std::uint8_t, 3>{230, 60, 50} : std::array<std::uint8_t, 3>{48, 48, 48};
      const auto row = grid.forward_cells() - 1 - forward;
      const auto column = grid.lateral_cells() - 1 - lateral;
      auto *pixel = message.data.data() + row * message.step + column * 3;
      std::copy(color.begin(), color.end(), pixel);
    }
  }
  return message;
}
}  // namespace

class GroundBEVNode final : public rclcpp::Node {
 public:
  GroundBEVNode() : Node("ground_bev_node") {
    input_topic_ = declare_parameter("input_topic", "/camera_lidar_bev/points_rgb");
    expected_frame_ = declare_parameter("expected_frame", "os_sensor");
    stale_after_s_ = declare_parameter("stale_after_s", 0.75);
    max_publish_hz_ = declare_parameter("max_publish_hz", 20.0);
    if (input_topic_.empty() || expected_frame_.empty() ||
        !std::isfinite(stale_after_s_) || stale_after_s_ <= 0 || stale_after_s_ > 2 ||
        !std::isfinite(max_publish_hz_) || max_publish_hz_ <= 0 || max_publish_hz_ > 20) {
      throw std::invalid_argument("Invalid input, frame, freshness or publication rate parameter");
    }
    read_config();
    validate_config(config_);
    auto cloud_publisher = [this](const char *parameter, const char *fallback) {
      const auto topic = declare_parameter(parameter, fallback);
      if (topic.empty()) throw std::invalid_argument("Output topic must not be empty");
      return create_publisher<Cloud>(topic, rclcpp::QoS(1).reliable());
    };
    ground_pub_ = cloud_publisher("ground_topic", "/camera_lidar_bev/ground_points");
    obstacle_pub_ = cloud_publisher("obstacle_topic", "/camera_lidar_bev/obstacle_points");
    unknown_pub_ = cloud_publisher("unknown_topic", "/camera_lidar_bev/unknown_points");
    classified_pub_ = cloud_publisher("classified_topic", "/camera_lidar_bev/ground_bev_points");
    const auto grid_topic = declare_parameter("occupancy_topic", "/camera_lidar_bev/occupancy");
    const auto grid_image_topic = declare_parameter("occupancy_image_topic", "/camera_lidar_bev/occupancy_image");
    const auto diagnostic_topic = declare_parameter("diagnostics_topic", "/camera_lidar_bev/diagnostics");
    if (grid_topic.empty() || grid_image_topic.empty() || diagnostic_topic.empty()) {
      throw std::invalid_argument("Output topic must not be empty");
    }
    occupancy_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(grid_topic,
      rclcpp::QoS(1).reliable().transient_local());
    occupancy_image_pub_ = create_publisher<sensor_msgs::msg::Image>(grid_image_topic,
      rclcpp::QoS(1).reliable().transient_local());
    diagnostic_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(diagnostic_topic, rclcpp::QoS(5));
    input_sub_ = create_subscription<Cloud>(input_topic_, rclcpp::QoS(1).reliable(),
      [this](Cloud::ConstSharedPtr message) {
        input_ = std::move(message);
        receipt_ = SteadyClock::now();
        ++received_version_;
      });
    process_timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / max_publish_hz_),
      [this]() { process(); });
    diagnostic_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() { publish_diagnostics(); });
    unavailable("Waiting for a fresh point cloud in " + expected_frame_);
    RCLCPP_WARN(get_logger(), "Stationary ground/obstacle preview in %s; ground, camera calibration, motion and synchronization remain provisional. No TF is published.", expected_frame_.c_str());
  }

 private:
  void read_config() {
    auto value = [this](const char *name, double &target) { target = declare_parameter(name, target); };
    value("ground_height_m", config_.ground_height_m);
    value("ground_height_tolerance_m", config_.ground_height_tolerance_m);
    value("auto_ground_height_min_m", config_.auto_ground_height_min_m);
    value("auto_ground_height_max_m", config_.auto_ground_height_max_m);
    value("max_slope_deg", config_.max_slope_deg);
    value("fit_min_range_m", config_.fit_min_range_m);
    value("fit_max_range_m", config_.fit_max_range_m);
    value("min_range_m", config_.min_range_m);
    value("max_range_m", config_.max_range_m);
    value("ground_distance_m", config_.ground_distance_m);
    value("obstacle_min_height_m", config_.obstacle_min_height_m);
    value("obstacle_max_height_m", config_.obstacle_max_height_m);
    value("patch_size_m", config_.patch_size_m);
    value("local_support_radius_m", config_.local_support_radius_m);
    value("min_ground_span_m", config_.min_ground_span_m);
    auto count = [this](const char *name, std::size_t &target) {
      const auto requested = declare_parameter<std::int64_t>(name, static_cast<std::int64_t>(target));
      if (requested <= 0) throw std::invalid_argument(std::string(name) + " must be positive");
      target = static_cast<std::size_t>(requested);
    };
    count("min_ground_points", config_.min_ground_points);
    count("min_ground_patches", config_.min_ground_patches);
    config_.self_mask_enabled = declare_parameter("self_mask_enabled", config_.self_mask_enabled);
    value("self_x_min_m", config_.self_x_min_m); value("self_x_max_m", config_.self_x_max_m);
    value("self_y_min_m", config_.self_y_min_m); value("self_y_max_m", config_.self_y_max_m);
    value("self_z_min_m", config_.self_z_min_m); value("self_z_max_m", config_.self_z_max_m);
    value("grid_x_min_m", config_.grid.x_min_m); value("grid_x_max_m", config_.grid.x_max_m);
    value("grid_y_min_m", config_.grid.y_min_m); value("grid_y_max_m", config_.grid.y_max_m);
    value("grid_resolution_m", config_.grid.resolution_m);
  }

  void unavailable(const std::string &reason) {
    current_error_ = reason;
    if (outputs_cleared_) return;
    std_msgs::msg::Header header;
    header.frame_id = expected_frame_;
    header.stamp = now();
    const auto empty = empty_cloud(header);
    ground_pub_->publish(empty);
    obstacle_pub_->publish(empty);
    unknown_pub_->publish(empty);
    classified_pub_->publish(empty);
    const bev_core::Grid unknown_grid(config_.grid);
    occupancy_pub_->publish(occupancy_message(unknown_grid, header));
    occupancy_image_pub_->publish(occupancy_image(unknown_grid, header));
    outputs_cleared_ = true;
    ++cleared_outputs_;
    last_ground_ = GroundModel{};
    counts_.fill(0);
    cloud_outputs_active_ = 0;
    image_output_active_ = false;
    grid_free_ = grid_occupied_ = 0;
    grid_unknown_ = unknown_grid.cell_count();
  }

  void process() {
    const auto started = SteadyClock::now();
    const auto ros_now = now().nanoseconds();
    if (!input_) { unavailable("Waiting for point cloud input"); return; }
    if (input_->header.frame_id != expected_frame_) {
      unavailable("Input frame is " + input_->header.frame_id + "; expected " + expected_frame_); return;
    }
    const auto source_stamp = stamp_ns(input_->header);
    const auto age = static_cast<double>(ros_now - source_stamp) / 1e9;
    const auto receipt_age = std::chrono::duration<double>(started - receipt_).count();
    if (input_->header.stamp.nanosec >= 1000000000U || source_stamp <= 0 ||
        age < -0.05 || age > stale_after_s_ || receipt_age > stale_after_s_) {
      unavailable("Point cloud is stale or its header timestamp is invalid; cached scans are not republished"); return;
    }
    if (!input_->width || !input_->height) { unavailable("Point cloud source cleared its output"); return; }
    if (received_version_ == processed_version_) return;
    processed_version_ = received_version_;
    if (last_source_stamp_ && source_stamp <= last_source_stamp_) {
      unavailable("Repeated or nonmonotonic source timestamp; no scan was republished"); return;
    }
    try {
      const auto points = decode_cloud(*input_);
      const auto decoded_at = SteadyClock::now();
      const auto result = classify_and_grid(points.xyz, config_);
      const auto classified_at = SteadyClock::now();
      if (result.labels.size() != points.xyz.size() || result.heights_above_ground.size() != points.xyz.size()) {
        throw std::runtime_error("Core classification/height evidence did not preserve input indices");
      }
      // Check again after classification and serialization: a scan that was
      // fresh at callback entry must not become live evidence after a slow fit.
      // RViz normally subscribes to the combined cloud. Do not walk and encode
      // the full scan again for partition topics without any current consumers.
      auto encode_if_subscribed = [&](const rclcpp::Publisher<Cloud>::SharedPtr &publisher,
                                      Label label, bool combined = false) -> std::optional<Cloud> {
        if (publisher->get_subscription_count() + publisher->get_intra_process_subscription_count() == 0) {
          return std::nullopt;
        }
        return labeled_cloud(points, result, label, input_->header, combined);
      };
      const auto ground_message = encode_if_subscribed(ground_pub_, Label::Ground);
      const auto obstacle_message = encode_if_subscribed(obstacle_pub_, Label::Obstacle);
      const auto unknown_message = encode_if_subscribed(unknown_pub_, Label::Unknown);
      const auto classified_message = encode_if_subscribed(classified_pub_, Label::Unknown, true);
      const auto grid_message = occupancy_message(result.grid, input_->header);
      std::optional<sensor_msgs::msg::Image> grid_image;
      const bool image_has_subscribers = occupancy_image_pub_->get_subscription_count() +
        occupancy_image_pub_->get_intra_process_subscription_count() > 0;
      if (image_has_subscribers) grid_image = occupancy_image(result.grid, input_->header);
      const auto serialized_at = SteadyClock::now();
      const auto completed_age = static_cast<double>(now().nanoseconds() - source_stamp) / 1e9;
      const auto completed_receipt_age = std::chrono::duration<double>(SteadyClock::now() - receipt_).count();
      if (completed_age < -0.05 || completed_age > stale_after_s_ || completed_receipt_age > stale_after_s_) {
        unavailable("Point cloud became stale during ground classification; no cached scan was published");
        return;
      }
      const auto publish_started = SteadyClock::now();
      if (ground_message) ground_pub_->publish(*ground_message);
      if (obstacle_message) obstacle_pub_->publish(*obstacle_message);
      if (unknown_message) unknown_pub_->publish(*unknown_message);
      if (classified_message) classified_pub_->publish(*classified_message);
      occupancy_pub_->publish(grid_message);
      if (grid_image) occupancy_image_pub_->publish(*grid_image);
      const auto published_at = SteadyClock::now();
      counts_.fill(0);
      for (const auto label : result.labels) ++counts_.at(static_cast<std::size_t>(label));
      grid_free_ = grid_occupied_ = grid_unknown_ = 0;
      for (const auto &cell : result.grid.cells()) {
        if (cell.occupancy == bev_core::Occupancy::Free) ++grid_free_;
        else if (cell.occupancy == bev_core::Occupancy::Occupied) ++grid_occupied_;
        else ++grid_unknown_;
      }
      last_ground_ = result.ground;
      last_source_stamp_ = source_stamp;
      last_source_age_s_ = completed_age;
      decode_ms_ = std::chrono::duration<double, std::milli>(decoded_at - started).count();
      core_ms_ = std::chrono::duration<double, std::milli>(classified_at - decoded_at).count();
      serialize_ms_ = std::chrono::duration<double, std::milli>(serialized_at - classified_at).count();
      publish_ms_ = std::chrono::duration<double, std::milli>(published_at - publish_started).count();
      cloud_outputs_active_ = ground_message.has_value() + obstacle_message.has_value() +
        unknown_message.has_value() + classified_message.has_value();
      image_output_active_ = grid_image.has_value();
      processing_ms_ = std::chrono::duration<double, std::milli>(SteadyClock::now() - started).count();
      outputs_cleared_ = false;
      ++processed_scans_;
      current_error_ = result.ground.valid ? "" : result.ground.reason;
    } catch (const std::exception &error) {
      unavailable(std::string("Rejected point cloud: ") + error.what());
    }
  }

  void publish_diagnostics() {
    diagnostic_msgs::msg::DiagnosticArray message;
    message.header.stamp = now();
    Status status;
    status.name = "ground_bev";
    status.hardware_id = expected_frame_ + ":stationary_preview";
    status.level = Status::WARN;
    status.message = current_error_.empty() ?
      "Provisional stationary ground/obstacle BEV; camera calibration and motion remain unvalidated" : current_error_;
    auto item = [&status](const std::string &key, const auto &value) {
      diagnostic_msgs::msg::KeyValue pair;
      pair.key = key;
      if constexpr (std::is_same_v<std::decay_t<decltype(value)>, std::string>) pair.value = value;
      else pair.value = std::to_string(value);
      status.values.push_back(std::move(pair));
    };
    item("provisional", std::string("true"));
    item("stationary_mode", std::string("true"));
    item("tf_published", std::string("false"));
    item("expected_frame", expected_frame_);
    item("input_topic", input_topic_);
    item("plane_valid", last_ground_.valid ? std::string("true") : std::string("false"));
    item("ground_height_prior_used", last_ground_.used_height_prior ? std::string("true") : std::string("false"));
    item("ground_height_m", config_.ground_height_m);
    item("self_mask_enabled", config_.self_mask_enabled ? std::string("true") : std::string("false"));
    item("self_x_min_m", config_.self_x_min_m); item("self_x_max_m", config_.self_x_max_m);
    item("self_y_min_m", config_.self_y_min_m); item("self_y_max_m", config_.self_y_max_m);
    item("self_z_min_m", config_.self_z_min_m); item("self_z_max_m", config_.self_z_max_m);
    item("plane_a", last_ground_.a); item("plane_b", last_ground_.b); item("plane_c", last_ground_.c);
    item("ground_reason", last_ground_.reason);
    item("support_points", last_ground_.support_points); item("support_patches", last_ground_.support_patches);
    item("ground_rms_m", last_ground_.rms_m); item("slope_deg", last_ground_.slope_deg);
    item("input_points", counts_[0] + counts_[1] + counts_[2] + counts_[3] + counts_[4]);
    item("invalid_points", counts_[0]); item("unknown_points", counts_[1]);
    item("ground_points", counts_[2]); item("obstacle_points", counts_[3]); item("masked_points", counts_[4]);
    item("grid_free_cells", grid_free_); item("grid_occupied_cells", grid_occupied_); item("grid_unknown_cells", grid_unknown_);
    item("processed_scans", processed_scans_); item("cleared_outputs", cleared_outputs_);
    item("processing_ms", processing_ms_);
    item("decode_ms", decode_ms_); item("core_ms", core_ms_);
    item("serialize_ms", serialize_ms_); item("publish_ms", publish_ms_);
    item("cloud_outputs_active", cloud_outputs_active_);
    item("occupancy_image_active", image_output_active_ ? std::string("true") : std::string("false"));
    item("occupancy_image_subscribers", occupancy_image_pub_->get_subscription_count() +
      occupancy_image_pub_->get_intra_process_subscription_count());
    item("source_age_s", input_ ? static_cast<double>(now().nanoseconds() - stamp_ns(input_->header)) / 1e9 :
      std::numeric_limits<double>::quiet_NaN());
    item("source_age_at_processing_s", last_source_age_s_);
    item("outputs_cleared", outputs_cleared_ ? std::string("true") : std::string("false"));
    item("current_error", current_error_);
    message.status.push_back(std::move(status));
    diagnostic_pub_->publish(message);
  }

  Config config_;
  std::string input_topic_, expected_frame_, current_error_;
  double stale_after_s_{0.75}, max_publish_hz_{20.0};
  Cloud::ConstSharedPtr input_;
  SteadyClock::time_point receipt_{};
  std::uint64_t received_version_{0}, processed_version_{0}, processed_scans_{0}, cleared_outputs_{0};
  std::int64_t last_source_stamp_{0};
  bool outputs_cleared_{false};
  GroundModel last_ground_;
  std::array<std::size_t, 5> counts_{};
  std::size_t grid_free_{0}, grid_occupied_{0}, grid_unknown_{0};
  double processing_ms_{0}, last_source_age_s_{0};
  double decode_ms_{0}, core_ms_{0}, serialize_ms_{0}, publish_ms_{0};
  std::size_t cloud_outputs_active_{0};
  bool image_output_active_{false};
  rclcpp::Subscription<Cloud>::SharedPtr input_sub_;
  rclcpp::Publisher<Cloud>::SharedPtr ground_pub_, obstacle_pub_, unknown_pub_, classified_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr occupancy_image_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostic_pub_;
  rclcpp::TimerBase::SharedPtr process_timer_, diagnostic_timer_;
};
}  // namespace bev_geometry

int main(int argc, char **argv) {
  try {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<bev_geometry::GroundBEVNode>());
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception &error) {
    RCLCPP_FATAL(rclcpp::get_logger("ground_bev_node"), "%s", error.what());
    if (rclcpp::ok()) rclcpp::shutdown();
    return 1;
  }
}
