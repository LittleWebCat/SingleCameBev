#include "bev_visualization/yolo_detector.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace bev_visualization {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
constexpr std::array<const char *, 80> kCocoNames{{
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
  "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
  "toothbrush"}};

bool road_id(int id) { return id == 0 || id == 1 || id == 2 || id == 3 || id == 5 || id == 7; }

void check_letterbox(const Letterbox &box) {
  const auto expected = make_letterbox(box.source_width, box.source_height);
  if (box.resized_width != expected.resized_width || box.resized_height != expected.resized_height ||
      box.pad_left != expected.pad_left || box.pad_top != expected.pad_top ||
      box.scale_x != expected.scale_x || box.scale_y != expected.scale_y) {
    throw std::invalid_argument("Letterbox does not match native image dimensions");
  }
}

double iou(const YoloDetection &a, const YoloDetection &b) {
  const double width = std::max(0.0, std::min(a.x_max, b.x_max) - std::max(a.x_min, b.x_min));
  const double height = std::max(0.0, std::min(a.y_max, b.y_max) - std::max(a.y_min, b.y_min));
  const double overlap = width * height;
  const double total = (a.x_max-a.x_min)*(a.y_max-a.y_min) +
    (b.x_max-b.x_min)*(b.y_max-b.y_min) - overlap;
  return total > 0.0 ? overlap / total : 0.0;
}

void cuda_check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char *message) noexcept override {
    if (severity <= Severity::kERROR) {
      try { error = message ? message : "TensorRT error"; } catch (...) {}
    }
  }
  std::string error;
};

struct DeviceBuffer {
  void *pointer{};
  explicit DeviceBuffer(std::size_t bytes) { cuda_check(cudaMalloc(&pointer, bytes), "cudaMalloc"); }
  ~DeviceBuffer() { if (pointer) cudaFree(pointer); }
  DeviceBuffer(const DeviceBuffer &) = delete;
};
struct HostBuffer {
  float *pointer{};
  explicit HostBuffer(std::size_t elements) {
    cuda_check(cudaMallocHost(reinterpret_cast<void **>(&pointer), elements*sizeof(float)), "cudaMallocHost");
  }
  ~HostBuffer() { if (pointer) cudaFreeHost(pointer); }
  HostBuffer(const HostBuffer &) = delete;
};
struct Stream {
  cudaStream_t value{};
  Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "cudaStreamCreate"); }
  ~Stream() { if (value) cudaStreamDestroy(value); }
};
struct Event {
  cudaEvent_t value{};
  Event() { cuda_check(cudaEventCreate(&value), "cudaEventCreate"); }
  ~Event() { if (value) cudaEventDestroy(value); }
};

bool shape_is(const nvinfer1::Dims &shape, std::initializer_list<int> expected) {
  if (shape.nbDims != static_cast<int>(expected.size())) return false;
  std::size_t i = 0;
  for (const int size : expected) if (shape.d[i++] != size) return false;
  return true;
}
}  // namespace

const char *coco_class_name(int class_id) {
  if (class_id < 0 || class_id >= static_cast<int>(kCocoNames.size()))
    throw std::invalid_argument("COCO class ID is out of range");
  return kCocoNames[static_cast<std::size_t>(class_id)];
}

int road_class_id(const std::string &name) {
  for (const int id : {0, 1, 2, 3, 5, 7}) if (name == coco_class_name(id)) return id;
  throw std::invalid_argument("Unsupported road-user class: " + name);
}

void validate_yolo_config(const YoloConfig &config) {
  if (!std::isfinite(config.confidence) || config.confidence <= 0.0 || config.confidence > 1.0 ||
      !std::isfinite(config.nms_iou) || config.nms_iou < 0.0 || config.nms_iou > 1.0 ||
      config.max_detections == 0 || config.max_detections > 300 || config.allowed_class_ids.empty()) {
    throw std::invalid_argument("Invalid YOLO confidence, NMS, class or detection limit");
  }
  std::set<int> seen;
  for (const int id : config.allowed_class_ids) {
    if (!road_id(id) || !seen.insert(id).second)
      throw std::invalid_argument("YOLO classes must be unique validated COCO road-user IDs");
  }
}

Letterbox make_letterbox(int width, int height) {
  if (width <= 0 || height <= 0 || width > 16384 || height > 16384)
    throw std::invalid_argument("Invalid native image size");
  const double gain = std::min(static_cast<double>(kYoloInputSize)/width,
                               static_cast<double>(kYoloInputSize)/height);
  Letterbox box;
  box.source_width = width; box.source_height = height;
  box.resized_width = std::max(1, static_cast<int>(std::round(width*gain)));
  box.resized_height = std::max(1, static_cast<int>(std::round(height*gain)));
  box.pad_left = (kYoloInputSize-box.resized_width)/2;
  box.pad_top = (kYoloInputSize-box.resized_height)/2;
  box.scale_x = static_cast<double>(box.resized_width)/width;
  box.scale_y = static_cast<double>(box.resized_height)/height;
  return box;
}

void prepare_yolo_input(const cv::Mat &bgr, const Letterbox &box, float *output, std::size_t count) {
  check_letterbox(box);
  if (!output || count != kYoloInputElements || bgr.type() != CV_8UC3 ||
      bgr.cols != box.source_width || bgr.rows != box.source_height) {
    throw std::invalid_argument("YOLO requires a matching BGR8 image and exact input buffer");
  }
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(box.resized_width, box.resized_height), 0, 0, cv::INTER_LINEAR);
  std::fill(output, output+count, 114.0F/255.0F);
  constexpr std::size_t plane = kYoloInputSize*kYoloInputSize;
  for (int y = 0; y < resized.rows; ++y) {
    const auto *row = resized.ptr<cv::Vec3b>(y);
    for (int x = 0; x < resized.cols; ++x) {
      const auto index = static_cast<std::size_t>(y+box.pad_top)*kYoloInputSize+x+box.pad_left;
      output[index] = row[x][2]/255.0F;
      output[plane+index] = row[x][1]/255.0F;
      output[2*plane+index] = row[x][0]/255.0F;
    }
  }
}

std::vector<YoloDetection> decode_yolo_output(const float *output, std::size_t count,
                                            const Letterbox &box, const YoloConfig &config) {
  check_letterbox(box);
  validate_yolo_config(config);
  if (!output || count != kYoloOutputElements)
    throw std::invalid_argument("Expected exact YOLO11 COCO [1,84,8400] output");
  std::array<bool, kYoloClasses> allowed{};
  for (const int id : config.allowed_class_ids) allowed[static_cast<std::size_t>(id)] = true;
  std::vector<YoloDetection> candidates;
  candidates.reserve(256);
  for (std::size_t anchor = 0; anchor < kYoloAnchors; ++anchor) {
    const double cx = output[anchor], cy = output[kYoloAnchors+anchor];
    const double width = output[2*kYoloAnchors+anchor], height = output[3*kYoloAnchors+anchor];
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(width) || !std::isfinite(height) ||
        width <= 0.0 || height <= 0.0) continue;
    int class_id = -1;
    double score = 0.0;
    bool valid_scores = true;
    for (std::size_t id = 0; id < kYoloClasses; ++id) {
      const double value = output[(4+id)*kYoloAnchors+anchor];
      if (!std::isfinite(value) || value < 0.0 || value > 1.0) { valid_scores = false; break; }
      if (value > score) { score = value; class_id = static_cast<int>(id); }
    }
    if (!valid_scores || score < config.confidence || class_id < 0 || !allowed[class_id]) continue;
    YoloDetection detection{class_id, score,
      std::clamp((cx-width/2-box.pad_left)/box.scale_x, 0.0, static_cast<double>(box.source_width)),
      std::clamp((cy-height/2-box.pad_top)/box.scale_y, 0.0, static_cast<double>(box.source_height)),
      std::clamp((cx+width/2-box.pad_left)/box.scale_x, 0.0, static_cast<double>(box.source_width)),
      std::clamp((cy+height/2-box.pad_top)/box.scale_y, 0.0, static_cast<double>(box.source_height))};
    if (detection.x_max <= detection.x_min || detection.y_max <= detection.y_min) continue;
    candidates.push_back(detection);
  }
  std::stable_sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) { return a.score > b.score; });
  // Bound CPU NMS latency even if every anchor passes the confidence threshold.
  if (candidates.size() > 1000) candidates.resize(1000);
  std::vector<YoloDetection> kept;
  kept.reserve(std::min(config.max_detections, candidates.size()));
  for (const auto &candidate : candidates) {
    bool suppressed = false;
    for (const auto &previous : kept) {
      if (candidate.class_id == previous.class_id && iou(candidate, previous) > config.nms_iou) {
        suppressed = true; break;
      }
    }
    if (!suppressed) kept.push_back(candidate);
    if (kept.size() == config.max_detections) break;
  }
  return kept;
}

struct YoloDetector::Impl {
  YoloConfig config;
  Logger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::unique_ptr<Stream> stream;
  std::unique_ptr<Event> start_event, stop_event;
  std::unique_ptr<DeviceBuffer> device_input, device_output;
  std::unique_ptr<HostBuffer> host_input, host_output;

  Impl(const std::string &file, YoloConfig options) : config(std::move(options)) {
    validate_yolo_config(config);
    std::ifstream input(file, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open TensorRT engine: " + file);
    const auto bytes = input.tellg();
    if (bytes <= 0 || bytes > 128*1024*1024) throw std::runtime_error("Invalid YOLO engine file size");
    std::vector<char> plan(static_cast<std::size_t>(bytes));
    input.seekg(0);
    if (!input.read(plan.data(), static_cast<std::streamsize>(plan.size())))
      throw std::runtime_error("Incomplete TensorRT engine file");
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime) throw std::runtime_error("TensorRT runtime creation failed: " + logger.error);
    engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine) throw std::runtime_error("TensorRT engine deserialization failed: " + logger.error);
    if (engine->getNbIOTensors() != 2 || engine->getNbOptimizationProfiles() != 1)
      throw std::runtime_error("YOLO engine must contain exactly two fixed I/O tensors and one profile");
    for (int index = 0; index < engine->getNbIOTensors(); ++index) {
      const std::string name = engine->getIOTensorName(index);
      const bool is_input = engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
      const bool expected = is_input ? name == "images" && shape_is(engine->getTensorShape(name.c_str()), {1,3,640,640}) :
        name == "output0" && shape_is(engine->getTensorShape(name.c_str()), {1,84,8400});
      if (!expected || engine->getTensorDataType(name.c_str()) != nvinfer1::DataType::kFLOAT ||
          engine->getTensorLocation(name.c_str()) != nvinfer1::TensorLocation::kDEVICE ||
          engine->getTensorFormat(name.c_str()) != nvinfer1::TensorFormat::kLINEAR ||
          engine->getTensorVectorizedDim(name.c_str()) != -1 || engine->isShapeInferenceIO(name.c_str())) {
        throw std::runtime_error("YOLO engine must be fixed FLOAT32 images[1,3,640,640] / output0[1,84,8400] COCO");
      }
    }
    context.reset(engine->createExecutionContext());
    if (!context) throw std::runtime_error("TensorRT execution context creation failed: " + logger.error);
    stream = std::make_unique<Stream>();
    start_event = std::make_unique<Event>(); stop_event = std::make_unique<Event>();
    device_input = std::make_unique<DeviceBuffer>(kYoloInputElements*sizeof(float));
    device_output = std::make_unique<DeviceBuffer>(kYoloOutputElements*sizeof(float));
    host_input = std::make_unique<HostBuffer>(kYoloInputElements);
    host_output = std::make_unique<HostBuffer>(kYoloOutputElements);
    if (!context->setTensorAddress("images", device_input->pointer) ||
        !context->setTensorAddress("output0", device_output->pointer))
      throw std::runtime_error("TensorRT tensor address setup failed");
  }

  ~Impl() { if (stream) cudaStreamSynchronize(stream->value); }

  YoloInference infer(const cv::Mat &bgr) {
    YoloInference result;
    auto started = Clock::now();
    const auto letterbox = make_letterbox(bgr.cols, bgr.rows);
    prepare_yolo_input(bgr, letterbox, host_input->pointer, kYoloInputElements);
    result.preprocess_ms = elapsed_ms(started);
    started = Clock::now();
    cuda_check(cudaMemcpyAsync(device_input->pointer, host_input->pointer,
      kYoloInputElements*sizeof(float), cudaMemcpyHostToDevice, stream->value), "YOLO upload");
    cuda_check(cudaEventRecord(start_event->value, stream->value), "YOLO start event");
    if (!context->enqueueV3(stream->value)) throw std::runtime_error("TensorRT enqueue failed: " + logger.error);
    cuda_check(cudaEventRecord(stop_event->value, stream->value), "YOLO stop event");
    cuda_check(cudaMemcpyAsync(host_output->pointer, device_output->pointer,
      kYoloOutputElements*sizeof(float), cudaMemcpyDeviceToHost, stream->value), "YOLO download");
    cuda_check(cudaStreamSynchronize(stream->value), "YOLO synchronization");
    result.inference_ms = elapsed_ms(started);
    float compute_ms{};
    cuda_check(cudaEventElapsedTime(&compute_ms, start_event->value, stop_event->value), "YOLO event timing");
    result.gpu_compute_ms = compute_ms;
    started = Clock::now();
    result.detections = decode_yolo_output(host_output->pointer, kYoloOutputElements, letterbox, config);
    result.postprocess_ms = elapsed_ms(started);
    return result;
  }
};

YoloDetector::YoloDetector(const std::string &engine_file, YoloConfig config)
  : impl_(std::make_unique<Impl>(engine_file, std::move(config))) {}
YoloDetector::~YoloDetector() = default;
YoloInference YoloDetector::infer(const cv::Mat &bgr) { return impl_->infer(bgr); }

}  // namespace bev_visualization
