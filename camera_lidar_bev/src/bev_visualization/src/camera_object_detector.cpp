#include "bev_visualization/yolo_detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

namespace bev_visualization {
namespace {
using Clock = std::chrono::steady_clock;
using Image = sensor_msgs::msg::Image;
using Detections = vision_msgs::msg::Detection2DArray;
using Status = diagnostic_msgs::msg::DiagnosticStatus;
std::int64_t stamp_ns(const std_msgs::msg::Header &header) {
  return static_cast<std::int64_t>(header.stamp.sec)*1000000000LL+header.stamp.nanosec;
}
double elapsed_ms(Clock::time_point started) {
  return std::chrono::duration<double, std::milli>(Clock::now()-started).count();
}
Image bgr_message(const cv::Mat &bgr, const std_msgs::msg::Header &header) {
  Image image;
  image.header = header;
  image.height = static_cast<std::uint32_t>(bgr.rows);
  image.width = static_cast<std::uint32_t>(bgr.cols);
  image.encoding = "bgr8";
  image.is_bigendian = false;
  image.step = image.width*3;
  image.data.resize(static_cast<std::size_t>(image.step)*image.height);
  for (std::uint32_t row = 0; row < image.height; ++row)
    std::copy_n(bgr.ptr<std::uint8_t>(static_cast<int>(row)), image.step,
                image.data.data()+static_cast<std::size_t>(row)*image.step);
  return image;
}
cv::Scalar class_color(int class_id) {
  if (class_id == 0) return {255, 0, 255};
  if (class_id == 1) return {255, 255, 0};
  return {0, 165, 255};
}
}  // namespace

class CameraObjectDetector final : public rclcpp::Node {
 public:
  CameraObjectDetector() : Node("camera_object_detector") {
    engine_file_ = declare_parameter("engine_file", std::string{});
    camera_frame_ = declare_parameter("camera_frame", "camera_front_optical_frame");
    width_ = declare_parameter("image_width", 1024);
    height_ = declare_parameter("image_height", 768);
    const auto image_topic = declare_parameter("image_topic", "/camera/front/image_raw");
    const auto detections_topic = declare_parameter("detections_topic", "/camera_lidar_bev/detections_2d");
    const auto labeled_image_topic = declare_parameter("labeled_image_topic", "/camera_lidar_bev/camera_labeled");
    max_publish_hz_ = declare_parameter("max_publish_hz", 20.0);
    stale_after_s_ = declare_parameter("stale_after_s", 0.75);
    YoloConfig config;
    config.confidence = declare_parameter("confidence", 0.35);
    config.nms_iou = declare_parameter("nms_iou", 0.5);
    const int max_detections = declare_parameter("max_detections", 100);
    if (max_detections <= 0) throw std::invalid_argument("max_detections must be positive");
    config.max_detections = static_cast<std::size_t>(max_detections);
    const auto classes = declare_parameter<std::vector<std::string>>("classes",
      {"person", "bicycle", "car", "motorcycle", "bus", "truck"});
    config.allowed_class_ids.clear();
    for (const auto &name : classes) config.allowed_class_ids.push_back(road_class_id(name));
    validate_yolo_config(config);
    if (engine_file_.empty() || camera_frame_.empty() || image_topic.empty() ||
        detections_topic.empty() || labeled_image_topic.empty() ||
        width_ <= 0 || height_ <= 0 || width_ > 4096 || height_ > 4096 ||
        !std::isfinite(max_publish_hz_) || max_publish_hz_ <= 0 || max_publish_hz_ > 20.0 ||
        !std::isfinite(stale_after_s_) || stale_after_s_ <= 0 || stale_after_s_ > 1.0) {
      throw std::invalid_argument("Invalid engine, camera, topic, dimension, rate or freshness parameter");
    }
    detector_ = std::make_unique<YoloDetector>(engine_file_, std::move(config));
    detections_pub_ = create_publisher<Detections>(detections_topic, rclcpp::QoS(1).reliable());
    image_pub_ = create_publisher<Image>(labeled_image_topic, rclcpp::SensorDataQoS().keep_last(1));
    diagnostics_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "/camera_lidar_bev/diagnostics", rclcpp::QoS(5).reliable());
    image_sub_ = create_subscription<Image>(image_topic, rclcpp::SensorDataQoS().keep_last(1),
      [this](Image::ConstSharedPtr image) {
        ++received_frames_;
        if (image_ && version_ != processed_version_) ++mailbox_dropped_frames_;
        image_ = std::move(image);
        receipt_ = Clock::now();
        ++version_;
      });
    process_timer_ = create_wall_timer(std::chrono::duration<double>(1.0/max_publish_hz_), [this]() { process(); });
    diagnostics_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() { publish_diagnostics(); });
    RCLCPP_INFO(get_logger(), "Native TensorRT COCO road-user labels: %s, 640x640 batch 1, %.1f Hz maximum",
      engine_file_.c_str(), max_publish_hz_);
  }

 private:
  std::string image_error(const Image &image) const {
    if (image.header.frame_id != camera_frame_) return "Camera optical frame mismatch";
    if (image.width != static_cast<std::uint32_t>(width_) || image.height != static_cast<std::uint32_t>(height_))
      return "Camera native image dimensions mismatch";
    if (image.encoding != "bgr8" && image.encoding != "rgb8") return "Image encoding must be bgr8 or rgb8";
    if (static_cast<std::uint64_t>(image.width)*3 > image.step ||
        static_cast<std::uint64_t>(image.step)*image.height != image.data.size())
      return "Invalid image row or payload layout";
    return {};
  }

  bool fresh(const Image &image, Clock::time_point receipt) const {
    const auto stamp = stamp_ns(image.header);
    const auto age = static_cast<double>(now().nanoseconds()-stamp)/1e9;
    const auto receipt_age = std::chrono::duration<double>(Clock::now()-receipt).count();
    return image.header.stamp.nanosec < 1000000000U && stamp > 0 && age >= 0.0 &&
      age <= stale_after_s_ && receipt_age >= 0.0 && receipt_age <= stale_after_s_;
  }

  void clear_outputs(const std::string &reason, bool invalid = false) {
    last_error_ = reason;
    error_is_invalid_ = invalid;
    if (outputs_cleared_) return;
    Detections empty;
    empty.header.stamp = now();
    empty.header.frame_id = camera_frame_;
    cv::Mat cleared(height_, width_, CV_8UC3, cv::Scalar(35, 35, 35));
    cv::putText(cleared, "Camera labels unavailable", {20, 40}, cv::FONT_HERSHEY_SIMPLEX,
      0.8, cv::Scalar(120, 180, 255), 2, cv::LINE_AA);
    cv::putText(cleared, reason.substr(0, 90), {20, 75}, cv::FONT_HERSHEY_SIMPLEX,
      0.55, cv::Scalar(190, 190, 190), 1, cv::LINE_AA);
    auto image = bgr_message(cleared, empty.header);
    detections_pub_->publish(empty);
    image_pub_->publish(std::move(image));
    outputs_cleared_ = true;
    ++cleared_outputs_;
    last_detection_count_ = 0;
  }

  void process() {
    if (!image_) { clear_outputs("Waiting for camera image"); return; }
    const auto image = image_;
    const auto receipt = receipt_;
    const auto source_version = version_;
    const auto layout_error = image_error(*image);
    if (!layout_error.empty()) {
      if (source_version != processed_version_) ++rejected_frames_;
      processed_version_ = source_version;
      clear_outputs(layout_error, true); return;
    }
    if (!fresh(*image, receipt)) {
      if (source_version != processed_version_) ++stale_dropped_frames_;
      processed_version_ = source_version;
      clear_outputs("Camera image is stale or timestamp is in the future"); return;
    }
    if (source_version == processed_version_) return;
    processed_version_ = source_version;
    const auto source_stamp = stamp_ns(image->header);
    if (source_stamp == last_processed_stamp_) { ++duplicate_frames_; return; }
    if (source_stamp < last_processed_stamp_) {
      ++rejected_frames_;
      clear_outputs("Camera image timestamps moved backwards", true); return;
    }
    const auto started = Clock::now();
    source_age_at_start_s_ = static_cast<double>(now().nanoseconds()-source_stamp)/1e9;
    try {
      const auto decode_started = Clock::now();
      cv::Mat native(height_, width_, CV_8UC3, const_cast<std::uint8_t *>(image->data.data()), image->step);
      cv::Mat bgr;
      if (image->encoding == "rgb8") cv::cvtColor(native, bgr, cv::COLOR_RGB2BGR);
      else bgr = native;
      decode_ms_ = elapsed_ms(decode_started);
      const auto inference = detector_->infer(bgr);
      ++inferred_frames_;
      preprocess_ms_ = inference.preprocess_ms;
      inference_ms_ = inference.inference_ms;
      gpu_compute_ms_ = inference.gpu_compute_ms;
      postprocess_ms_ = inference.postprocess_ms;
      const auto annotate_started = Clock::now();
      Detections detections;
      detections.header = image->header;
      detections.detections.reserve(inference.detections.size());
      cv::Mat labeled = bgr.clone();
      for (const auto &item : inference.detections) {
        vision_msgs::msg::Detection2D detection;
        detection.header = image->header;
        detection.bbox.center.position.x = (item.x_min+item.x_max)/2;
        detection.bbox.center.position.y = (item.y_min+item.y_max)/2;
        detection.bbox.center.theta = 0;
        detection.bbox.size_x = item.x_max-item.x_min;
        detection.bbox.size_y = item.y_max-item.y_min;
        vision_msgs::msg::ObjectHypothesisWithPose result;
        result.hypothesis.class_id = coco_class_name(item.class_id);
        result.hypothesis.score = item.score;
        // No pose or persistent identity is inferred by the 2D detector.
        detection.results.push_back(std::move(result));
        detections.detections.push_back(std::move(detection));
        const cv::Scalar color = class_color(item.class_id);
        const cv::Point p0(std::clamp(static_cast<int>(std::floor(item.x_min)), 0, width_-1),
          std::clamp(static_cast<int>(std::floor(item.y_min)), 0, height_-1));
        const cv::Point p1(std::clamp(static_cast<int>(std::ceil(item.x_max)), 0, width_-1),
          std::clamp(static_cast<int>(std::ceil(item.y_max)), 0, height_-1));
        cv::rectangle(labeled, p0, p1, color, 2, cv::LINE_AA);
        char label[128];
        std::snprintf(label, sizeof(label), "%s %.2f", coco_class_name(item.class_id), item.score);
        const int label_y = std::max(20, p0.y-6);
        cv::putText(labeled, label, {p0.x, label_y}, cv::FONT_HERSHEY_SIMPLEX, 0.65,
          cv::Scalar(10, 10, 10), 4, cv::LINE_AA);
        cv::putText(labeled, label, {p0.x, label_y}, cv::FONT_HERSHEY_SIMPLEX, 0.65,
          color, 2, cv::LINE_AA);
      }
      char timing[160];
      std::snprintf(timing, sizeof(timing), "Camera labels | %zu detections", inference.detections.size());
      cv::putText(labeled, timing, {15, height_-20}, cv::FONT_HERSHEY_SIMPLEX, 0.65,
        cv::Scalar(10, 10, 10), 4, cv::LINE_AA);
      cv::putText(labeled, timing, {15, height_-20}, cv::FONT_HERSHEY_SIMPLEX, 0.65,
        cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
      auto labeled_image = bgr_message(labeled, image->header);
      annotate_ms_ = elapsed_ms(annotate_started);
      processing_ms_ = elapsed_ms(started);
      source_age_at_publish_s_ = static_cast<double>(now().nanoseconds()-source_stamp)/1e9;
      if (!fresh(*image, receipt)) {
        ++stale_dropped_frames_;
        clear_outputs("Camera image expired during inference"); return;
      }
      const auto publish_started = Clock::now();
      detections_pub_->publish(std::move(detections));
      image_pub_->publish(std::move(labeled_image));
      publish_ms_ = elapsed_ms(publish_started);
      last_processed_stamp_ = source_stamp;
      last_detection_count_ = inference.detections.size();
      ++processed_frames_;
      outputs_cleared_ = false;
      last_error_.clear();
      error_is_invalid_ = false;
    } catch (const std::exception &error) {
      ++inference_errors_;
      clear_outputs(std::string("Detector failure: ")+error.what(), true);
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Camera detector failure: %s", error.what());
    }
  }

  void publish_diagnostics() {
    diagnostic_msgs::msg::DiagnosticArray report;
    report.header.stamp = now();
    Status status;
    status.name = "camera_object_detector";
    status.hardware_id = "front_camera_tensor_rt";
    status.level = outputs_cleared_ ? (error_is_invalid_ ? Status::ERROR : Status::WARN) : Status::OK;
    status.message = outputs_cleared_ ? last_error_ : "COCO road-user 2D camera detections; native TensorRT";
    const auto field = [&status](const std::string &name, const auto &value) {
      diagnostic_msgs::msg::KeyValue entry;
      entry.key = name; entry.value = std::to_string(value); status.values.push_back(std::move(entry));
    };
    diagnostic_msgs::msg::KeyValue engine;
    engine.key = "engine_file"; engine.value = engine_file_; status.values.push_back(std::move(engine));
    field("received_frames", received_frames_); field("processed_frames", processed_frames_);
    field("inferred_frames", inferred_frames_); field("mailbox_dropped_frames", mailbox_dropped_frames_);
    field("stale_dropped_frames", stale_dropped_frames_); field("rejected_frames", rejected_frames_);
    field("duplicate_frames", duplicate_frames_); field("inference_errors", inference_errors_);
    field("cleared_outputs", cleared_outputs_); field("outputs_cleared", outputs_cleared_);
    field("detections", last_detection_count_); field("max_publish_hz", max_publish_hz_);
    field("source_stamp_ns", last_processed_stamp_); field("source_age_at_start_s", source_age_at_start_s_);
    field("source_age_at_publish_s", source_age_at_publish_s_);
    field("source_age_s", image_ ? static_cast<double>(now().nanoseconds()-stamp_ns(image_->header))/1e9 : -1.0);
    field("receipt_age_s", image_ ? std::chrono::duration<double>(Clock::now()-receipt_).count() : -1.0);
    field("decode_ms", decode_ms_); field("preprocess_ms", preprocess_ms_);
    field("inference_ms", inference_ms_); field("gpu_compute_ms", gpu_compute_ms_);
    field("postprocess_ms", postprocess_ms_); field("annotate_ms", annotate_ms_);
    field("processing_ms", processing_ms_); field("publish_ms", publish_ms_);
    report.status.push_back(std::move(status));
    diagnostics_pub_->publish(std::move(report));
  }

  std::string engine_file_, camera_frame_, last_error_;
  int width_{}, height_{};
  double max_publish_hz_{}, stale_after_s_{};
  std::unique_ptr<YoloDetector> detector_;
  Image::ConstSharedPtr image_;
  Clock::time_point receipt_{};
  std::uint64_t version_{}, processed_version_{}, received_frames_{}, processed_frames_{}, inferred_frames_{};
  std::uint64_t mailbox_dropped_frames_{}, stale_dropped_frames_{}, rejected_frames_{}, duplicate_frames_{};
  std::uint64_t inference_errors_{}, cleared_outputs_{};
  std::int64_t last_processed_stamp_{};
  std::size_t last_detection_count_{};
  bool outputs_cleared_{false}, error_is_invalid_{false};
  double decode_ms_{}, preprocess_ms_{}, inference_ms_{}, gpu_compute_ms_{}, postprocess_ms_{};
  double annotate_ms_{}, processing_ms_{}, publish_ms_{}, source_age_at_start_s_{}, source_age_at_publish_s_{};
  rclcpp::Publisher<Detections>::SharedPtr detections_pub_;
  rclcpp::Publisher<Image>::SharedPtr image_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
  rclcpp::Subscription<Image>::SharedPtr image_sub_;
  rclcpp::TimerBase::SharedPtr process_timer_, diagnostics_timer_;
};

}  // namespace bev_visualization

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<bev_visualization::CameraObjectDetector>());
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception &error) {
    RCLCPP_FATAL(rclcpp::get_logger("camera_object_detector"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
}
