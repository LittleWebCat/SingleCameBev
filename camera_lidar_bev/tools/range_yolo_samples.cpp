// Camera vehicle proposals for inspecting raw LiDAR support in saved images.
// Runs the existing YOLO core once per image, without ROS or any 3D model.
#include "bev_visualization/yolo_detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;

void write_new_file(const fs::path &path, const std::string &content) {
  std::FILE *file = std::fopen(path.c_str(), "wx");
  if (!file) throw std::runtime_error("Cannot create new output: " + path.string());
  const auto written = std::fwrite(content.data(), 1, content.size(), file);
  const int closed = std::fclose(file);
  if (written != content.size() || closed != 0)
    throw std::runtime_error("Could not finish writing output: " + path.string());
}
}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4)
      throw std::runtime_error("Usage: range_yolo_samples MANIFEST ENGINE OUTPUT_JSON");
    const fs::path manifest_path = fs::absolute(argv[1]);
    const fs::path engine_path = fs::absolute(argv[2]);
    const fs::path output_path = fs::absolute(argv[3]);
    if (fs::exists(output_path))
      throw std::runtime_error("Refusing to overwrite output: " + output_path.string());
    std::ifstream input(manifest_path);
    if (!input) throw std::runtime_error("Cannot open manifest: " + manifest_path.string());
    Json manifest;
    input >> manifest;

    bev_visualization::YoloConfig config;
    config.confidence = 0.35;
    config.nms_iou = 0.5;
    config.allowed_class_ids = {2, 3, 5, 7};
    bev_visualization::YoloDetector detector(engine_path.string(), config);

    Json output = {
      {"manifest", manifest_path.string()},
      {"engine", engine_path.string()},
      {"model", "YOLO11n camera 2D"},
      {"confidence_threshold", config.confidence},
      {"nms_iou", config.nms_iou},
      {"allowed_class_ids", config.allowed_class_ids},
      {"inferences_per_image", 1},
      {"purpose", "Vehicle image ROIs for raw LiDAR point inspection"},
      {"warning", "Model proposals are not ground-truth labels or range measurements."},
      {"samples", Json::array()}
    };
    for (const auto &sample : manifest.at("samples")) {
      fs::path image_path = sample.at("image").get<std::string>();
      if (image_path.is_relative()) image_path = manifest_path.parent_path() / image_path;
      const auto image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (image.empty()) throw std::runtime_error("Cannot read image: " + image_path.string());
      const auto prediction = detector.infer(image);
      Json detections = Json::array();
      for (const auto &box : prediction.detections) {
        detections.push_back({
          {"class_id", box.class_id},
          {"class_name", bev_visualization::coco_class_name(box.class_id)},
          {"score", box.score},
          {"native_xyxy_px", {box.x_min, box.y_min, box.x_max, box.y_max}}
        });
      }
      output["samples"].push_back({
        {"sample_id", sample.at("sample_id")},
        {"image", image_path.string()},
        {"width", image.cols},
        {"height", image.rows},
        {"detections", detections}
      });
      std::cout << sample.at("sample_id").dump() << ": " << detections.size()
                << " vehicle proposals" << std::endl;
    }
    write_new_file(output_path, output.dump(2) + "\n");
    std::cout << "Saved " << output_path << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "range_yolo_samples: " << error.what() << std::endl;
    return 1;
  }
}
