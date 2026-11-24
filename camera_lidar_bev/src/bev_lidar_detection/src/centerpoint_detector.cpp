#include "bev_lidar_detection/centerpoint_detector.hpp"
#include "centerpoint_vfe.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace bev_lidar_detection {
namespace {
// Must match tools/centerpoint_pillar.py and models/lidar_objects_002/provenance.json.
constexpr float range_min_x = -51.2f, range_min_y = -51.2f;
constexpr float voxel_x = 0.2f, voxel_y = 0.2f;
constexpr int grid = 512;
constexpr int bev_channels = 64;
constexpr int feature_stride = 4;
constexpr int feature_size = grid / feature_stride;  // 128
constexpr int feature_pixels = feature_size * feature_size;
constexpr int num_classes = 10;
constexpr int reg_channels = 10;   // dx dy z log(l) log(w) log(h) cos sin vx vy
constexpr float score_floor = 0.1f;
constexpr std::size_t topk_per_head = 500;
constexpr float nms_iou = 0.2f;
constexpr std::size_t nms_post_max = 83;
constexpr float center_limit = 61.2f, center_limit_z = 10.0f;
constexpr double pi = 3.14159265358979323846;
// nuScenes classes in heatmap order, renamed to this project's vocabulary.
const std::vector<std::string> class_names{
  "car", "truck", "construction_vehicle", "bus", "trailer",
  "barrier", "motorcycle", "bicycle", "person", "traffic_cone"};
constexpr std::array<std::array<int, 2>, 6> head_classes{{{0, 1}, {1, 3}, {3, 5}, {5, 6}, {6, 8}, {8, 10}}};

void check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

class Logger final : public nvinfer1::ILogger {
  void log(Severity severity, const char *message) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
  }
};

template <class T> struct HostBuffer {
  T *ptr = nullptr;
  void allocate(std::size_t n) { check(cudaMallocHost(reinterpret_cast<void **>(&ptr), n * sizeof(T)), "cudaMallocHost"); }
  ~HostBuffer() { if (ptr) cudaFreeHost(ptr); }
};
template <class T> struct DeviceBuffer {
  T *ptr = nullptr;
  void allocate(std::size_t n) { check(cudaMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T)), "cudaMalloc"); }
  ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
};

// BEV polygon of a box, counter-clockwise.
using Polygon = std::vector<std::array<double, 2>>;
Polygon corners(const Box3D &b) {
  const double c = std::cos(b.yaw), s = std::sin(b.yaw), hx = b.size_x / 2.0, hy = b.size_y / 2.0;
  Polygon p;
  for (const auto &k : std::array<std::array<double, 2>, 4>{{{hx, hy}, {-hx, hy}, {-hx, -hy}, {hx, -hy}}})
    p.push_back({b.x + c * k[0] - s * k[1], b.y + s * k[0] + c * k[1]});
  return p;
}
double area(const Polygon &p) {
  double a = 0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    const auto &u = p[i], &v = p[(i + 1) % p.size()];
    a += u[0] * v[1] - v[0] * u[1];
  }
  return std::abs(a) / 2.0;
}
}  // namespace

double bev_iou(const Box3D &a, const Box3D &b) {
  // Sutherland-Hodgman clip of a's rectangle against b's convex rectangle.
  Polygon out = corners(a);
  const Polygon clip = corners(b);
  for (std::size_t i = 0; i < clip.size() && !out.empty(); ++i) {
    const auto &p0 = clip[i], &p1 = clip[(i + 1) % clip.size()];
    const auto side = [&](const std::array<double, 2> &q) {
      return (p1[0] - p0[0]) * (q[1] - p0[1]) - (p1[1] - p0[1]) * (q[0] - p0[0]);
    };
    Polygon in;
    in.swap(out);
    for (std::size_t j = 0; j < in.size(); ++j) {
      const auto &cur = in[j], &prev = in[(j + in.size() - 1) % in.size()];
      const double sc = side(cur), sp = side(prev);
      const auto cross = [&] {
        const double t = sp / (sp - sc);
        return std::array<double, 2>{prev[0] + (cur[0] - prev[0]) * t, prev[1] + (cur[1] - prev[1]) * t};
      };
      if (sc >= 0) {
        if (sp < 0) out.push_back(cross());
        out.push_back(cur);
      } else if (sp >= 0) {
        out.push_back(cross());
      }
    }
  }
  if (out.size() < 3) return 0.0;
  const double inter = area(out);
  return inter / std::max(double(a.size_x) * a.size_y + double(b.size_x) * b.size_y - inter, 1e-9);
}

class CenterPointDetector::Impl {
 public:
  explicit Impl(const CenterPointConfig &c) : config_(c) {
    if (!std::isfinite(c.score_threshold) || c.score_threshold < score_floor || c.score_threshold >= 1 ||
        c.max_output == 0 || !std::isfinite(c.z_offset) || std::abs(c.z_offset) > 3 ||
        !std::isfinite(c.intensity_scale) || c.intensity_scale <= 0)
      throw std::invalid_argument("Invalid CenterPoint configuration");
    std::ifstream f(c.engine_path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open CenterPoint engine " + c.engine_path);
    const std::vector<char> blob{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("TensorRT runtime creation failed");
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) throw std::runtime_error("Failed to deserialize CenterPoint engine " + c.engine_path);
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("TensorRT execution context creation failed");
    verify_contract();
    load_pfn_weights(c.pfn_weights_path.empty()
        ? c.engine_path.substr(0, c.engine_path.find_last_of('/') + 1) + "pfn_weights.bin" : c.pfn_weights_path);
    points_.allocate(CenterPointDetector::max_points);
    ages_.allocate(CenterPointDetector::max_points);
    d_ages_.allocate(CenterPointDetector::max_points);
    offsets_.allocate(grid * grid + 1);
    flat_index_.allocate(grid * grid);
    d_points_.allocate(CenterPointDetector::max_points);
    d_offsets_.allocate(grid * grid + 1);
    d_flat_index_.allocate(grid * grid);
    d_canvas_.allocate(static_cast<std::size_t>(bev_channels) * grid * grid);
    d_heatmap_.allocate(num_classes * feature_pixels);
    d_reg_.allocate(head_classes.size() * reg_channels * feature_pixels);
    heatmap_.allocate(num_classes * feature_pixels);
    reg_.allocate(head_classes.size() * reg_channels * feature_pixels);
    check(cudaStreamCreate(&stream_), "cudaStreamCreate");
    check(cudaMemsetAsync(d_canvas_.ptr, 0, static_cast<std::size_t>(bev_channels) * grid * grid * sizeof(float), stream_),
          "canvas clear");
    check(cudaStreamSynchronize(stream_), "canvas clear");
    if (!context_->setTensorAddress("spatial_features", d_canvas_.ptr) ||
        !context_->setTensorAddress("heatmap", d_heatmap_.ptr) ||
        !context_->setTensorAddress("reg", d_reg_.ptr))
      throw std::runtime_error("Failed to bind CenterPoint tensors");
    cell_.assign(grid * grid, -1);
    pillar_count_.resize(grid * grid);
    touched_.reserve(grid * grid);
    model_points_.reserve(CenterPointDetector::max_points);
    model_ages_.reserve(CenterPointDetector::max_points);
    point_pillar_.reserve(CenterPointDetector::max_points);
  }
  ~Impl() { if (stream_) cudaStreamDestroy(stream_); }

  std::vector<Box3D> infer(const std::vector<PointXYZI> &points, const std::vector<float> *ages) {
    const auto start = Clock::now();
    stats_ = {};
    stats_.input_points = points.size();
    if (points.size() > CenterPointDetector::max_points)
      throw std::invalid_argument("CenterPoint input exceeds its point limit");
    if (ages && ages->size() != points.size())
      throw std::invalid_argument("CenterPoint sweep ages must match the points");
    ages_used_ = ages != nullptr;
    const std::size_t pillars = pillarize(points, ages);
    const auto pre_end = Clock::now();
    stats_.preprocess_ms = ms(start, pre_end);
    std::vector<Box3D> out;
    if (pillars) {
      run_network(pillars);
      const auto net_end = Clock::now();
      stats_.inference_ms = ms(pre_end, net_end);
      out = decode();
      stats_.postprocess_ms = ms(net_end, Clock::now());
    }
    stats_.output_boxes = out.size();
    stats_.processing_ms = ms(start, Clock::now());
    return out;
  }

  DetectorStats stats_;

 private:
  using Clock = std::chrono::steady_clock;
  static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  }
  struct Deleter {
    template <class T> void operator()(T *p) const { delete p; }
  };

  void verify_contract() {
    const auto expect = [&](const char *name, nvinfer1::TensorIOMode mode, nvinfer1::DataType type,
                            std::vector<int64_t> dims) {
      if (engine_->getTensorIOMode(name) != mode || engine_->getTensorDataType(name) != type)
        throw std::runtime_error(std::string("CenterPoint engine tensor mismatch: ") + name);
      const auto d = engine_->getTensorShape(name);
      if (d.nbDims != static_cast<int>(dims.size())) throw std::runtime_error(std::string("CenterPoint rank mismatch: ") + name);
      for (int i = 0; i < d.nbDims; ++i)
        if (d.d[i] != dims[i]) throw std::runtime_error(std::string("CenterPoint shape mismatch: ") + name);
    };
    using M = nvinfer1::TensorIOMode;
    using T = nvinfer1::DataType;
    if (engine_->getNbIOTensors() != 3) throw std::runtime_error("CenterPoint engine must have 3 I/O tensors");
    expect("spatial_features", M::kINPUT, T::kFLOAT, {1, bev_channels, grid, grid});
    expect("heatmap", M::kOUTPUT, T::kFLOAT, {1, num_classes, feature_size, feature_size});
    expect("reg", M::kOUTPUT, T::kFLOAT, {1, static_cast<int64_t>(head_classes.size()) * reg_channels, feature_size, feature_size});
  }

  void load_pfn_weights(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != static_cast<std::streamoff>(centerpoint_pfn_weight_count * sizeof(float)))
      throw std::runtime_error("Missing or wrong-size CenterPoint PFN weights " + path);
    std::vector<float> w(centerpoint_pfn_weight_count);
    f.seekg(0);
    f.read(reinterpret_cast<char *>(w.data()), static_cast<std::streamsize>(w.size() * sizeof(float)));
    if (!std::all_of(w.begin(), w.end(), [](float v) { return std::isfinite(v); }))
      throw std::runtime_error("Non-finite CenterPoint PFN weights " + path);
    d_weights_.allocate(w.size());
    check(cudaMemcpy(d_weights_.ptr, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice), "PFN weights upload");
  }

  // Groups in-range points by pillar (CSR order, first-seen pillar order) in the model frame.
  std::size_t pillarize(const std::vector<PointXYZI> &points, const std::vector<float> *ages) {
    model_points_.clear();
    model_ages_.clear();
    point_pillar_.clear();
    touched_.clear();
    for (std::size_t n = 0; n < points.size(); ++n) {
      const auto &p = points[n];
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || !std::isfinite(p.intensity)) continue;
      // os_sensor (x fwd, y left) -> nuScenes LIDAR_TOP (x right, y fwd).
      const float x = -p.y, y = p.x, z = p.z + config_.z_offset;
      const int cx = static_cast<int>(std::floor((x - range_min_x) / voxel_x));
      const int cy = static_cast<int>(std::floor((y - range_min_y) / voxel_y));
      if (cx < 0 || cx >= grid || cy < 0 || cy >= grid) continue;
      const int flat = cy * grid + cx;
      std::int32_t &id = cell_[flat];
      if (id < 0) {
        id = static_cast<std::int32_t>(touched_.size());
        touched_.push_back(flat);
        pillar_count_[id] = 0;
      }
      ++pillar_count_[id];
      model_points_.push_back({x, y, z, p.intensity * config_.intensity_scale});
      if (ages) model_ages_.push_back((*ages)[n]);
      point_pillar_.push_back(id);
    }
    const std::size_t pillars = touched_.size();
    offsets_.ptr[0] = 0;
    for (std::size_t id = 0; id < pillars; ++id) {
      offsets_.ptr[id + 1] = offsets_.ptr[id] + static_cast<int>(pillar_count_[id]);
      pillar_count_[id] = 0;  // reused as the fill cursor below
      flat_index_.ptr[id] = touched_[id];
      cell_[touched_[id]] = -1;
    }
    for (std::size_t i = 0; i < model_points_.size(); ++i) {
      const auto id = static_cast<std::size_t>(point_pillar_[i]);
      const int slot = offsets_.ptr[id] + static_cast<int>(pillar_count_[id]++);
      points_.ptr[slot] = model_points_[i];
      if (ages) ages_.ptr[slot] = model_ages_[i];
    }
    stats_.accepted_points = model_points_.size();
    stats_.input_pillars = stats_.accepted_pillars = pillars;
    return pillars;
  }

  void run_network(std::size_t pillars) {
    const int p = static_cast<int>(pillars);
    check(cudaMemcpyAsync(d_points_.ptr, points_.ptr, model_points_.size() * sizeof(float4),
                          cudaMemcpyHostToDevice, stream_), "points upload");
    check(cudaMemcpyAsync(d_offsets_.ptr, offsets_.ptr, (pillars + 1) * sizeof(int),
                          cudaMemcpyHostToDevice, stream_), "offsets upload");
    check(cudaMemcpyAsync(d_flat_index_.ptr, flat_index_.ptr, pillars * sizeof(int),
                          cudaMemcpyHostToDevice, stream_), "flat_index upload");
    if (ages_used_)
      check(cudaMemcpyAsync(d_ages_.ptr, ages_.ptr, model_points_.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream_), "ages upload");
    check(launch_centerpoint_vfe(d_points_.ptr, ages_used_ ? d_ages_.ptr : nullptr, d_offsets_.ptr, d_flat_index_.ptr, p, d_weights_.ptr,
                                 d_canvas_.ptr, stream_), "CenterPoint VFE");
    if (!context_->enqueueV3(stream_)) throw std::runtime_error("CenterPoint TensorRT enqueue failed");
    check(cudaMemcpyAsync(heatmap_.ptr, d_heatmap_.ptr, num_classes * feature_pixels * sizeof(float),
                          cudaMemcpyDeviceToHost, stream_), "heatmap download");
    check(cudaMemcpyAsync(reg_.ptr, d_reg_.ptr, head_classes.size() * reg_channels * feature_pixels * sizeof(float),
                          cudaMemcpyDeviceToHost, stream_), "reg download");
    // Leave the canvas zeroed for the next scan.
    check(launch_centerpoint_clear(d_flat_index_.ptr, p, d_canvas_.ptr, stream_), "canvas clear");
    check(cudaStreamSynchronize(stream_), "CenterPoint stream");
  }

  // OpenPCDet CenterHead.generate_predicted_boxes + class-agnostic rotated NMS per head.
  std::vector<Box3D> decode() {
    std::vector<Box3D> out;
    struct Candidate { float score; int cls; int pixel; };
    std::vector<Candidate> candidates;
    for (std::size_t h = 0; h < head_classes.size(); ++h) {
      candidates.clear();
      for (int c = head_classes[h][0]; c < head_classes[h][1]; ++c) {
        const float *hm = heatmap_.ptr + c * feature_pixels;
        for (int i = 0; i < feature_pixels; ++i)
          if (hm[i] > score_floor) candidates.push_back({hm[i], c, i});
      }
      const auto by_score = [](const Candidate &a, const Candidate &b) { return a.score > b.score; };
      if (candidates.size() > topk_per_head) {
        std::partial_sort(candidates.begin(), candidates.begin() + topk_per_head, candidates.end(), by_score);
        candidates.resize(topk_per_head);
      } else {
        std::sort(candidates.begin(), candidates.end(), by_score);
      }
      const float *r = reg_.ptr + h * reg_channels * feature_pixels;
      const auto at = [&](int ch, int px) { return r[ch * feature_pixels + px]; };
      std::vector<Box3D> boxes;  // model frame
      for (const auto &k : candidates) {
        const int px = k.pixel;
        const float x = ((px % feature_size) + at(0, px)) * feature_stride * voxel_x + range_min_x;
        const float y = ((px / feature_size) + at(1, px)) * feature_stride * voxel_y + range_min_y;
        const float z = at(2, px);
        if (!(std::abs(x) <= center_limit && std::abs(y) <= center_limit && std::abs(z) <= center_limit_z)) continue;
        const auto size = [&](int ch) { return std::exp(std::min(at(ch, px), 10.0f)); };
        Box3D b{x, y, z, size(3), size(4), size(5), std::atan2(at(7, px), at(6, px)), k.score, k.cls};
        if (std::isfinite(b.size_x) && std::isfinite(b.size_y) && std::isfinite(b.size_z) && std::isfinite(b.yaw))
          boxes.push_back(b);
      }
      std::vector<bool> suppressed(boxes.size(), false);
      std::size_t kept = 0;
      for (std::size_t i = 0; i < boxes.size() && kept < nms_post_max; ++i) {
        if (suppressed[i]) continue;
        ++kept;
        const float ri = std::hypot(boxes[i].size_x, boxes[i].size_y) / 2;
        for (std::size_t j = i + 1; j < boxes.size(); ++j) {
          if (suppressed[j]) continue;
          // Circumscribed circles that do not touch cannot overlap.
          const float reach = ri + std::hypot(boxes[j].size_x, boxes[j].size_y) / 2;
          const float dx = boxes[i].x - boxes[j].x, dy = boxes[i].y - boxes[j].y;
          if (dx * dx + dy * dy < reach * reach && bev_iou(boxes[i], boxes[j]) > nms_iou) suppressed[j] = true;
        }
        const auto &b = boxes[i];
        if (b.score < config_.score_threshold) continue;
        // Back to os_sensor: x = y_model, y = -x_model, yaw - 90 degrees.
        out.push_back({b.y, -b.x, b.z - config_.z_offset, b.size_x, b.size_y, b.size_z,
                       static_cast<float>(std::remainder(b.yaw - pi / 2, 2 * pi)), b.score, b.class_id});
      }
    }
    std::stable_sort(out.begin(), out.end(), [](const Box3D &a, const Box3D &b) { return a.score > b.score; });
    if (out.size() > config_.max_output) out.resize(config_.max_output);
    return out;
  }

  CenterPointConfig config_;
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime, Deleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, Deleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, Deleter> context_;
  cudaStream_t stream_ = nullptr;
  HostBuffer<float> heatmap_, reg_;
  HostBuffer<float4> points_;
  HostBuffer<float> ages_;
  DeviceBuffer<float> d_ages_;
  bool ages_used_ = false;
  std::vector<float> model_ages_;
  HostBuffer<int> offsets_, flat_index_;
  DeviceBuffer<float> d_weights_, d_canvas_, d_heatmap_, d_reg_;
  DeviceBuffer<float4> d_points_;
  DeviceBuffer<int> d_offsets_, d_flat_index_;
  std::vector<std::int32_t> cell_, point_pillar_;
  std::vector<int> touched_;
  std::vector<std::uint32_t> pillar_count_;
  std::vector<float4> model_points_;
};

CenterPointDetector::CenterPointDetector(const CenterPointConfig &c) : impl_(std::make_unique<Impl>(c)) {}
CenterPointDetector::~CenterPointDetector() = default;
std::vector<Box3D> CenterPointDetector::infer(const std::vector<PointXYZI> &p) { return impl_->infer(p, nullptr); }
std::vector<Box3D> CenterPointDetector::infer(const std::vector<PointXYZI> &p, const std::vector<float> &ages) {
  return impl_->infer(p, &ages);
}
const DetectorStats &CenterPointDetector::lastStats() const noexcept { return impl_->stats_; }
const std::vector<std::string> &CenterPointDetector::classNames() const noexcept { return class_names; }

}  // namespace bev_lidar_detection
