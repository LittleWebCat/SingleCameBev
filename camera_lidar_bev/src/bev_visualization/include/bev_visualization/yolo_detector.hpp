#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace bev_visualization {

struct YoloConfig {
  double confidence{0.35};
  double nms_iou{0.5};
  std::size_t max_detections{100};
  std::vector<int> allowed_class_ids{0, 1, 2, 3, 5, 7};
};

struct Letterbox {
  int source_width{}, source_height{};
  int resized_width{}, resized_height{};
  int pad_left{}, pad_top{};
  double scale_x{}, scale_y{};
};

struct YoloDetection {
  int class_id{};
  double score{};
  double x_min{}, y_min{}, x_max{}, y_max{};
};

constexpr int kYoloInputSize = 640;
constexpr std::size_t kYoloAnchors = 8400;
constexpr std::size_t kYoloClasses = 80;
constexpr std::size_t kYoloInputElements = 3U * kYoloInputSize * kYoloInputSize;
constexpr std::size_t kYoloOutputElements = (4U + kYoloClasses) * kYoloAnchors;

void validate_yolo_config(const YoloConfig &config);
const char *coco_class_name(int class_id);
int road_class_id(const std::string &name);
Letterbox make_letterbox(int source_width, int source_height);
// BGR uint8 input -> RGB float32 NCHW, normalized by 255, with 114 padding.
void prepare_yolo_input(const cv::Mat &bgr, const Letterbox &letterbox,
                        float *output, std::size_t count);
// Strict YOLO11 COCO [1,84,8400] channel-major output, no objectness channel.
std::vector<YoloDetection> decode_yolo_output(const float *output, std::size_t count,
                                            const Letterbox &letterbox,
                                            const YoloConfig &config);

struct YoloInference {
  std::vector<YoloDetection> detections;
  double preprocess_ms{}, inference_ms{}, gpu_compute_ms{}, postprocess_ms{};
};

class YoloDetector {
 public:
  explicit YoloDetector(const std::string &engine_file, YoloConfig config);
  ~YoloDetector();
  YoloDetector(const YoloDetector &) = delete;
  YoloDetector &operator=(const YoloDetector &) = delete;
  YoloInference infer(const cv::Mat &bgr);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace bev_visualization
