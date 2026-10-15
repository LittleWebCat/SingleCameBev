// Compare existing native perception cores on extracted recording samples.
// No ROS initialization, sensor driver, DDS, model export or accuracy test runs.
// Compile against the current built libraries, then run only when the GPU is free.
// The following engine class is copied verbatim from the existing native camera
// 3D node; no node class or ROS runtime is instantiated in this standalone tool.
#include "bev_visualization/camera_3d_geometry.hpp"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
namespace bev_visualization {
namespace {
constexpr int input_width = 1280, input_height = 384;
constexpr int feature_width = 320, feature_height = 96;
constexpr std::size_t input_elements = 3*input_width*input_height;
constexpr std::size_t heatmap_elements = 3*feature_width*feature_height;
constexpr std::size_t regression_elements = 8*feature_width*feature_height;
void cuda_check(cudaError_t result, const char *operation) {
  if(result!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(result));
}
class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char *message) noexcept override {
    if(severity<=Severity::kERROR){try{error=message ? message : "TensorRT error";}catch(...){}}
  }
  std::string error;
};
struct Buffer {
  void *device{};float *host{};std::size_t count{};
  explicit Buffer(std::size_t n):count(n) {
    cuda_check(cudaMalloc(&device,n*sizeof(float)),"cudaMalloc");
    const auto result=cudaMallocHost(reinterpret_cast<void **>(&host),n*sizeof(float));
    if(result!=cudaSuccess){cudaFree(device);device=nullptr;cuda_check(result,"cudaMallocHost");}
  }
  ~Buffer(){if(device)cudaFree(device);if(host)cudaFreeHost(host);}
  Buffer(const Buffer &)=delete;
};
struct Stream {
  cudaStream_t value{};
  Stream(){cuda_check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking),"cudaStreamCreate");}
  ~Stream(){if(value)cudaStreamDestroy(value);}
};
struct Event {
  cudaEvent_t value{};
  Event(){cuda_check(cudaEventCreate(&value),"cudaEventCreate");}
  ~Event(){if(value)cudaEventDestroy(value);}
};
bool shape_is(const nvinfer1::Dims &d,std::initializer_list<int> expected) {
  if(d.nbDims!=static_cast<int>(expected.size()))return false;
  int i=0;for(const int n:expected)if(d.d[i++]!=n)return false;
  return true;
}
class SmokeEngine {
 public:
  explicit SmokeEngine(const std::string &path) {
    if(!initLibNvInferPlugins(&logger_,""))throw std::runtime_error("TensorRT plugins unavailable");
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file)throw std::runtime_error("SMOKE engine is missing: "+path);
    const auto size=file.tellg();
    if(size<=0 || size>1024LL*1024*1024)throw std::runtime_error("Invalid SMOKE engine size");
    std::vector<char> bytes(static_cast<std::size_t>(size));file.seekg(0);file.read(bytes.data(),size);
    if(!file)throw std::runtime_error("Cannot read SMOKE engine");
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if(!runtime_)throw std::runtime_error("Cannot create TensorRT runtime");
    engine_.reset(runtime_->deserializeCudaEngine(bytes.data(),bytes.size()));
    if(!engine_)throw std::runtime_error("Cannot deserialize SMOKE: "+logger_.error);
    if(engine_->getNbIOTensors()!=3)throw std::runtime_error("SMOKE engine must have exactly three tensors");
    validate("image",nvinfer1::TensorIOMode::kINPUT,{1,3,384,1280});
    validate("heatmap",nvinfer1::TensorIOMode::kOUTPUT,{1,3,96,320});
    validate("regression",nvinfer1::TensorIOMode::kOUTPUT,{1,8,96,320});
    context_.reset(engine_->createExecutionContext());
    if(!context_)throw std::runtime_error("Cannot create SMOKE context");
    if(!context_->setTensorAddress("image",input_.device) ||
       !context_->setTensorAddress("heatmap",heatmap_.device) ||
       !context_->setTensorAddress("regression",regression_.device))
      throw std::runtime_error("Cannot assign SMOKE tensor buffers");
  }
  void prepare(const cv::Mat &rectified) {
    const double scale=static_cast<double>(input_width)/rectified.cols;
    const double shift=(input_height-rectified.rows*scale)/2;
    const cv::Matx23d affine(scale,0,0,0,scale,shift);
    cv::Mat cropped;cv::warpAffine(rectified,cropped,affine,{input_width,input_height},cv::INTER_LINEAR);
    constexpr std::array<float,3> mean{{.485F,.456F,.406F}},stdev{{.229F,.224F,.225F}};
    constexpr std::size_t plane=input_width*input_height;
    // The official SMOKE checkpoint normalizes BGR (not RGB) in this order.
    for(int y=0;y<input_height;++y){const auto *row=cropped.ptr<cv::Vec3b>(y);
      for(int x=0;x<input_width;++x){const auto i=static_cast<std::size_t>(y)*input_width+x;
        for(std::size_t c=0;c<3;++c)input_.host[c*plane+i]=(row[x][c]/255.F-mean[c])/stdev[c];}}
  }
  void infer(std::vector<float> &heatmap,std::vector<float> &regression) {
    cuda_check(cudaMemcpyAsync(input_.device,input_.host,input_elements*sizeof(float),cudaMemcpyHostToDevice,stream_.value),"SMOKE upload");
    cuda_check(cudaEventRecord(begin_.value,stream_.value),"CUDA start event");
    if(!context_->enqueueV3(stream_.value))throw std::runtime_error("SMOKE inference failed: "+logger_.error);
    cuda_check(cudaEventRecord(end_.value,stream_.value),"CUDA end event");
    cuda_check(cudaMemcpyAsync(heatmap_.host,heatmap_.device,heatmap_elements*sizeof(float),cudaMemcpyDeviceToHost,stream_.value),"SMOKE heatmap download");
    cuda_check(cudaMemcpyAsync(regression_.host,regression_.device,regression_elements*sizeof(float),cudaMemcpyDeviceToHost,stream_.value),"SMOKE regression download");
    cuda_check(cudaStreamSynchronize(stream_.value),"SMOKE synchronize");
    float ms=0;cuda_check(cudaEventElapsedTime(&ms,begin_.value,end_.value),"CUDA timing");gpu_ms=ms;
    heatmap.assign(heatmap_.host,heatmap_.host+heatmap_elements);
    regression.assign(regression_.host,regression_.host+regression_elements);
  }
  double gpu_ms{};
 private:
  void validate(const char *name,nvinfer1::TensorIOMode mode,std::initializer_list<int> shape) {
    if(engine_->getTensorIOMode(name)!=mode || engine_->getTensorDataType(name)!=nvinfer1::DataType::kFLOAT ||
       !shape_is(engine_->getTensorShape(name),shape))throw std::runtime_error(std::string("SMOKE tensor contract mismatch: ")+name);
  }
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  Stream stream_;Event begin_,end_;
  Buffer input_{input_elements},heatmap_{heatmap_elements},regression_{regression_elements};
};
}
}

#include "bev_visualization/camera_box_association.hpp"
#include "bev_visualization/yolo_detector.hpp"
#include "bev_lidar_detection/detector.hpp"
#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <numeric>

namespace {
using Json = nlohmann::json;
using Timer = std::chrono::steady_clock;
using Point = bev_lidar_detection::PointXYZI;
using LidarBox = bev_lidar_detection::Box3D;
double millis(Timer::time_point start) {
  return std::chrono::duration<double, std::milli>(Timer::now() - start).count();
}

struct Sample {
  std::string id;
  std::int64_t image_stamp{}, lidar_stamp{};
  cv::Mat image;
  std::vector<Point> points;
};

std::vector<Point> read_cloud(const std::string &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("Cannot read extracted XYZI cloud: " + path);
  const auto size = input.tellg();
  if (size < 0 || size % sizeof(Point) != 0)
    throw std::runtime_error("Extracted cloud must contain float32 XYZI records");
  std::vector<Point> result(static_cast<std::size_t>(size) / sizeof(Point));
  input.seekg(0);
  input.read(reinterpret_cast<char *>(result.data()), size);
  if (!input) throw std::runtime_error("Could not finish reading cloud: " + path);
  return result;
}

template<std::size_t N>
std::array<double, N> values(const YAML::Node &node) {
  std::array<double, N> result{};
  for (std::size_t i = 0; i < N; ++i) result[i] = node[i].as<double>();
  return result;
}

Json timing_summary(std::vector<double> values_ms) {
  std::sort(values_ms.begin(), values_ms.end());
  if (values_ms.empty()) return Json::object();
  const auto percentile = [&](double q) {
    const double position = q * (values_ms.size() - 1);
    const auto index = static_cast<std::size_t>(position);
    const auto next = std::min(index + 1, values_ms.size() - 1);
    return values_ms[index] + (values_ms[next] - values_ms[index]) * (position - index);
  };
  const double median = percentile(.5);
  return {{"count", values_ms.size()}, {"median_ms", median},
          {"p95_ms", percentile(.95)}, {"max_ms", values_ms.back()},
          {"mean_ms", std::accumulate(values_ms.begin(), values_ms.end(), 0.) / values_ms.size()},
          {"processing_capacity_hz", median > 0 ? 1000. / median : 0.}};
}

const char *lidar_class(int id) { return id == 0 ? "car" : "person"; }
bool own_body(const LidarBox &box) {
  return box.x >= -2.131 && box.x <= 2.350 && box.y >= -.905 && box.y <= .885 &&
         box.z >= -1.800 && box.z <= 0.;
}
std::vector<LidarBox> selected_lidar(std::vector<LidarBox> boxes) {
  boxes.erase(std::remove_if(boxes.begin(), boxes.end(), [](const auto &box) {
    return box.class_id < 0 || box.class_id > 1 || box.score < .35F || box.score > 1 ||
           box.x < 0 || box.x > 60 || box.size_x > 15 || box.size_y > 6 ||
           box.size_z > 5 || own_body(box);
  }), boxes.end());
  return boxes;
}
Json camera_box_json(const bev_visualization::Camera3DBox &box) {
  return {{"class", box.class_id == 0 ? "car" : "person"}, {"model_class_id", box.class_id},
          {"score", box.score}, {"camera_xyz_m", {box.center_xyz[0], box.center_xyz[1], box.center_xyz[2]}},
          {"size_lhw_m", {box.size_lhw[0], box.size_lhw[1], box.size_lhw[2]}},
          {"camera_rotation_y_rad", box.rotation_y}};
}
Json lidar_box_json(const LidarBox &box) {
  return {{"class", lidar_class(box.class_id)}, {"model_class_id", box.class_id},
          {"score", box.score}, {"os_sensor_xyz_m", {box.x, box.y, box.z}},
          {"size_lwh_m", {box.size_x, box.size_y, box.size_z}}, {"yaw_rad", box.yaw}};
}
Json projected_json(const bev_visualization::ProjectedBox &p) {
  return {{"visible", p.visible}, {"native_xyxy_px", {p.x_min, p.y_min, p.x_max, p.y_max}},
          {"native_center_px", {p.center_u, p.center_v}}};
}
}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3) {
      std::cerr << "Usage: perception_speed_compare manifest.yaml output.json\n";
      return 2;
    }
    cv::setNumThreads(2);
    const auto manifest = YAML::LoadFile(argv[1]);
    const auto calibration = manifest["calibration"];
    bev_visualization::Camera camera;
    camera.width = calibration["width"].as<unsigned>();
    camera.height = calibration["height"].as<unsigned>();
    camera.k = values<9>(calibration["K"]);
    camera.d = values<5>(calibration["D"]);
    const auto camera_from_sensor = values<16>(calibration["T"]);
    cv::Matx33d intrinsic;
    std::copy(camera.k.begin(), camera.k.end(), intrinsic.val);
    cv::Mat map_x, map_y;
    cv::initUndistortRectifyMap(cv::Mat(intrinsic), cv::Mat(camera.d), cv::Mat(), cv::Mat(intrinsic),
                              {static_cast<int>(camera.width), static_cast<int>(camera.height)},
                              CV_32FC1, map_x, map_y);
    std::vector<Sample> samples;
    for (const auto entry : manifest["samples"]) {
      Sample sample;
      sample.id = entry["sample_id"].as<std::string>();
      sample.image_stamp = entry["image_stamp_ns"].as<std::int64_t>();
      sample.lidar_stamp = entry["lidar_stamp_ns"].as<std::int64_t>();
      sample.image = cv::imread(entry["image"].as<std::string>(), cv::IMREAD_COLOR);
      if (sample.image.empty()) throw std::runtime_error("Cannot read sample image: " + sample.id);
      sample.points = read_cloud(entry["cloud"].as<std::string>());
      samples.push_back(std::move(sample));
    }
    if (samples.empty()) throw std::runtime_error("Manifest has no extracted samples");
    const auto project = std::filesystem::absolute(std::filesystem::path(__FILE__)).parent_path().parent_path();
    const auto model_path = [&](const char *key, const char *fallback) {
      return manifest["engines"] && manifest["engines"][key]
        ? manifest["engines"][key].as<std::string>() : (project / fallback).string();
    };
    const auto smoke_path = model_path("camera_3d", "models/camera_3d_001/smoke_fp16_raw.engine");
    const auto lidar_path = model_path("lidar", "models/lidar_objects_001/pointpillar_fp16.plan");
    const auto yolo_path = model_path("yolo", "models/camera_objects_001/yolo11n_fp16_raw.engine");
    bev_visualization::SmokeEngine smoke(smoke_path);
    bev_lidar_detection::Detector lidar(lidar_path, .35F, 200);
    bev_visualization::YoloConfig yolo_config;
    yolo_config.allowed_class_ids = {0, 2, 3, 5, 7};
    bev_visualization::YoloDetector yolo(yolo_path, yolo_config);
    const bev_visualization::AssociationConfig association_config{.20, .35, .05, .30};
    std::vector<float> heatmap, regression;
    cv::Mat rectified;
    std::vector<double> smoke_total, smoke_gpu, lidar_total, lidar_core, yolo_total, yolo_gpu, match_total, combined_total;
    Json rows = Json::array(), timing_rows = Json::array();
    std::size_t total_smoke_cars{}, total_smoke_people{}, total_lidar_cars{}, total_lidar_people{},
                total_visible{}, total_matches{}, matched_cars{}, matched_people{}, camera_2d_count{};

    const auto run = [&](const Sample &sample, bool measured, int repeat, Json *sample_report) {
      auto start = Timer::now();
      cv::remap(sample.image, rectified, map_x, map_y, cv::INTER_LINEAR);
      smoke.prepare(rectified);
      smoke.infer(heatmap, regression);
      auto camera_boxes = bev_visualization::decode_smoke(heatmap, regression, 320, 96, intrinsic,
                            sample.image.cols, sample.image.rows, .25, 50);
      camera_boxes.erase(std::remove_if(camera_boxes.begin(), camera_boxes.end(), [](const auto &box) {
        return (box.class_id != 0 && box.class_id != 2) || box.center_xyz[2] > 60 ||
               box.center_xyz[2] < .5 || std::abs(box.center_xyz[0]) > 40;
      }), camera_boxes.end());
      const double camera_ms = millis(start), camera_gpu_ms = smoke.gpu_ms;
      const auto combined_start = Timer::now();
      start = Timer::now();
      const auto lidar_boxes = selected_lidar(lidar.infer(sample.points));
      const double lidar_ms = millis(start), lidar_core_ms = lidar.lastStats().processing_ms;
      start = Timer::now();
      const auto labels = yolo.infer(sample.image);
      const double yolo_ms = millis(start);
      std::vector<bev_visualization::OrientedBox> oriented;
      for (const auto &box : lidar_boxes)
        oriented.push_back({{box.x, box.y, box.z}, {box.size_x, box.size_y, box.size_z},
                            {0, 0, std::sin(box.yaw / 2.), std::cos(box.yaw / 2.)}, lidar_class(box.class_id)});
      std::vector<bev_visualization::CameraDetection> camera_labels;
      for (const auto &label : labels.detections)
        camera_labels.push_back({bev_visualization::coco_class_name(label.class_id), label.score,
                                label.x_min, label.y_min, label.x_max, label.y_max});
      start = Timer::now();
      const auto matches = bev_visualization::associate_camera_boxes(oriented, camera_labels,
                              camera, camera_from_sensor, association_config);
      const double association_ms = millis(start), combined_ms = millis(combined_start);
      std::size_t match_count{}, visible{}, cars_matched{}, people_matched{};
      for (std::size_t i = 0; i < matches.size(); ++i) {
        if (matches[i].projection.visible) ++visible;
        if (matches[i].camera_index >= 0) {
          ++match_count;
          if (lidar_boxes[i].class_id == 0) ++cars_matched; else ++people_matched;
        }
      }
      if (measured) {
        smoke_total.push_back(camera_ms); smoke_gpu.push_back(camera_gpu_ms);
        lidar_total.push_back(lidar_ms); lidar_core.push_back(lidar_core_ms);
        yolo_total.push_back(yolo_ms); yolo_gpu.push_back(labels.gpu_compute_ms);
        match_total.push_back(association_ms); combined_total.push_back(combined_ms);
        timing_rows.push_back({{"sample_id", sample.id}, {"repeat", repeat},
          {"camera_3d_processing_ms", camera_ms}, {"camera_3d_gpu_ms", camera_gpu_ms},
          {"lidar_processing_ms", lidar_ms}, {"lidar_core_ms", lidar_core_ms},
          {"yolo_processing_ms", yolo_ms}, {"yolo_gpu_ms", labels.gpu_compute_ms},
          {"yolo_preprocess_ms", labels.preprocess_ms}, {"yolo_inference_transfer_ms", labels.inference_ms},
          {"yolo_postprocess_ms", labels.postprocess_ms}, {"matching_ms", association_ms},
          {"lidar_yolo_matching_processing_ms", combined_ms},
          {"camera_3d_boxes", camera_boxes.size()}, {"lidar_boxes", lidar_boxes.size()},
          {"camera_2d_boxes", camera_labels.size()}, {"matches", match_count}});
      }
      if (sample_report) {
        Json camera_results = Json::array(), lidar_results = Json::array(), label_results = Json::array(), match_results = Json::array();
        for (const auto &box : camera_boxes) {
          camera_results.push_back(camera_box_json(box));
          if (box.class_id == 0) ++total_smoke_cars; else ++total_smoke_people;
        }
        for (const auto &box : lidar_boxes) {
          lidar_results.push_back(lidar_box_json(box));
          if (box.class_id == 0) ++total_lidar_cars; else ++total_lidar_people;
        }
        for (const auto &label : camera_labels)
          label_results.push_back({{"class", label.class_name}, {"score", label.confidence},
                                   {"native_xyxy_px", {label.x_min, label.y_min, label.x_max, label.y_max}}});
        for (std::size_t i = 0; i < matches.size(); ++i)
          match_results.push_back({{"lidar_index", i}, {"camera_index", matches[i].camera_index},
                                   {"iou", matches[i].iou}, {"projection", projected_json(matches[i].projection)}});
        total_visible += visible; total_matches += match_count; matched_cars += cars_matched;
        matched_people += people_matched; camera_2d_count += camera_labels.size();
        *sample_report = {{"sample_id", sample.id}, {"image_stamp_ns", sample.image_stamp},
          {"lidar_stamp_ns", sample.lidar_stamp},
          {"pair_delta_ms", std::abs(static_cast<double>(sample.image_stamp - sample.lidar_stamp)) / 1e6},
          {"input_points", sample.points.size()}, {"camera_3d_boxes", camera_results},
          {"lidar_boxes", lidar_results}, {"camera_2d_boxes", label_results}, {"associations", match_results},
          {"projectable_lidar_boxes", visible}, {"matched_boxes", match_count}};
      }
    };
    for (int i = 0; i < 3; ++i) run(samples.front(), false, i, nullptr);
    for (const auto &sample : samples) {
      Json row;
      for (int repeat = 0; repeat < 3; ++repeat) run(sample, true, repeat, repeat == 0 ? &row : nullptr);
      rows.push_back(std::move(row));
    }
    const auto lidar_count = total_lidar_cars + total_lidar_people;
    Json report = {{"kind", "recording_processing_speed_and_camera_lidar_matching"},
      {"sample_count", samples.size()}, {"warmup_per_model", 3}, {"repeats_per_sample", 3},
      {"scope", "Sequential native core processing; excludes capture, disk reads, point-cloud extraction, ROS/DDS, output serialization and visualization. Matching is projected association, not ground-truth accuracy."},
      {"engines", {{"camera_3d", smoke_path}, {"lidar", lidar_path}, {"yolo", yolo_path}}},
      {"thresholds", {{"camera_3d_confidence", .25}, {"lidar_confidence", .35}, {"camera_2d_confidence", .35},
                      {"matching_iou", .20}, {"matching_ambiguity_margin", .05}, {"forward_range_m", 60}}},
      {"timing", {{"camera_3d_processing", timing_summary(smoke_total)}, {"camera_3d_gpu", timing_summary(smoke_gpu)},
                  {"lidar_processing", timing_summary(lidar_total)}, {"lidar_core", timing_summary(lidar_core)},
                  {"camera_2d_processing", timing_summary(yolo_total)}, {"camera_2d_gpu", timing_summary(yolo_gpu)},
                  {"matching", timing_summary(match_total)}, {"lidar_camera_2d_matching_processing", timing_summary(combined_total)}}},
      {"unique_sample_totals", {{"camera_3d_car_boxes", total_smoke_cars}, {"camera_3d_person_boxes", total_smoke_people},
        {"lidar_car_boxes", total_lidar_cars}, {"lidar_person_boxes", total_lidar_people}, {"camera_2d_boxes", camera_2d_count},
        {"projectable_lidar_boxes", total_visible}, {"matched_boxes", total_matches},
        {"matched_car_boxes", matched_cars}, {"matched_person_boxes", matched_people},
        {"matched_fraction_of_all_lidar_boxes", lidar_count ? static_cast<double>(total_matches) / lidar_count : 0.},
        {"matched_fraction_of_projectable_lidar_boxes", total_visible ? static_cast<double>(total_matches) / total_visible : 0.}}},
      {"samples", rows}, {"timing_runs", timing_rows}};
    std::ofstream output(argv[2]);
    if (!output) throw std::runtime_error("Cannot write comparison output");
    output << report.dump(2) << '\n';
    std::cout << Json({{"output", argv[2]}, {"timing", report["timing"]},
                      {"unique_sample_totals", report["unique_sample_totals"]}}).dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "perception_speed_compare: " << error.what() << '\n';
    return 1;
  }
}
