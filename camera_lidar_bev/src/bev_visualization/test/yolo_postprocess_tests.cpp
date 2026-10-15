#include "bev_visualization/yolo_detector.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace bev_visualization;
namespace {
void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
void near(double actual, double expected, const char *message) { require(std::abs(actual-expected) < 1e-4, message); }
template<typename Function> void rejects(Function function, const char *message) {
  bool rejected = false;
  try { function(); } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, message);
}
void anchor(std::vector<float> &output, std::size_t index, int id, float score,
            float cx, float cy, float width, float height) {
  output[index] = cx; output[kYoloAnchors+index] = cy;
  output[2*kYoloAnchors+index] = width; output[3*kYoloAnchors+index] = height;
  output[(4+static_cast<std::size_t>(id))*kYoloAnchors+index] = score;
}

void aspect_ratio_and_channels() {
  const auto box = make_letterbox(1024, 768);
  require(box.resized_width == 640 && box.resized_height == 480 && box.pad_top == 80 && box.pad_left == 0,
    "native 4:3 must be letterboxed, not stretched");
  near(box.scale_x, 0.625, "horizontal gain"); near(box.scale_y, 0.625, "vertical gain");
  cv::Mat bgr(768, 1024, CV_8UC3, cv::Scalar(20, 40, 60));
  std::vector<float> input(kYoloInputElements);
  prepare_yolo_input(bgr, box, input.data(), input.size());
  constexpr std::size_t plane = 640U*640U;
  near(input[0], 114.0/255, "padding normalization");
  near(input[80U*640], 60.0/255, "BGR to RGB red channel");
  near(input[plane+80U*640], 40.0/255, "green channel");
  near(input[2*plane+80U*640], 20.0/255, "blue channel");
  require(make_letterbox(768,1024).pad_left == 80, "portrait horizontal padding");
  const auto odd = make_letterbox(1001, 773);
  require(odd.resized_width == 640 && odd.resized_height == 494 && odd.pad_top == 73,
    "odd aspect ratio rounded resize");
  rejects([&] { prepare_yolo_input(bgr, box, input.data(), input.size()-1); }, "short input buffer rejected");
  rejects([] { make_letterbox(0, 768); }, "zero dimensions rejected");
}

void undo_letterbox_and_classwise_nms() {
  const auto box = make_letterbox(1024, 768);
  std::vector<float> raw(kYoloOutputElements, 0.0F);
  anchor(raw, 0, 2, .95F, 320, 320, 200, 100);
  anchor(raw, 1, 2, .8F, 321, 320, 200, 100);
  anchor(raw, 2, 0, .9F, 320, 320, 200, 100);
  anchor(raw, 3, 5, .2F, 500, 400, 80, 80);
  const auto detections = decode_yolo_output(raw.data(), raw.size(), box, YoloConfig{});
  require(detections.size() == 2, "same-class overlap suppressed, different classes retained");
  require(detections[0].class_id == 2 && detections[1].class_id == 0, "class IDs and score order");
  near(detections[0].x_min, 352, "unletterbox left"); near(detections[0].x_max, 672, "unletterbox right");
  near(detections[0].y_min, 304, "unletterbox top"); near(detections[0].y_max, 464, "unletterbox bottom");
  require(std::string(coco_class_name(2)) == "car" && road_class_id("truck") == 7,
    "explicit COCO model labels");
  YoloConfig one;
  one.max_detections = 1;
  require(decode_yolo_output(raw.data(), raw.size(), box, one).size() == 1, "detection count bounded");
}

void clipping_nonfinite_and_probability_guards() {
  const auto box = make_letterbox(1024, 768);
  std::vector<float> raw(kYoloOutputElements, 0.0F);
  anchor(raw, 0, 0, .9F, 0, 80, 100, 100);  // Straddles native left/top boundaries.
  anchor(raw, 1, 2, .9F, 300, 300, -100, 100);
  anchor(raw, 2, 2, .9F, std::numeric_limits<float>::infinity(), 300, 100, 100);
  anchor(raw, 3, 2, std::numeric_limits<float>::quiet_NaN(), 300, 300, 100, 100);
  anchor(raw, 4, 2, 1.1F, 300, 300, 100, 100);
  anchor(raw, 5, 4, .9F, 300, 300, 100, 100);  // Excluded airplane, not relabeled.
  anchor(raw, 6, 2, .9F, 300, 10, 100, 20);  // Entirely in top padding.
  const auto detections = decode_yolo_output(raw.data(), raw.size(), box, YoloConfig{});
  require(detections.size() == 1, "invalid, excluded, and padding-only predictions rejected");
  near(detections[0].x_min, 0, "left clipped"); near(detections[0].y_min, 0, "top clipped");
  near(detections[0].x_max, 80, "right boundary scaled"); near(detections[0].y_max, 80, "bottom boundary scaled");
  rejects([&] { decode_yolo_output(raw.data(), raw.size()-1, box, YoloConfig{}); }, "wrong output layout rejected");
  YoloConfig invalid;
  invalid.allowed_class_ids.push_back(2);
  rejects([&] { validate_yolo_config(invalid); }, "duplicate allowed class rejected");
  rejects([] { road_class_id("vehicle"); }, "unknown class name rejected");
}
}  // namespace

int main() {
  try {
    aspect_ratio_and_channels();
    undo_letterbox_and_classwise_nms();
    clipping_nonfinite_and_probability_guards();
    std::cout << "YOLO CPU preprocessing/postprocessing: 3 test groups passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
