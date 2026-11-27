#include "bev_lidar_detection/sweep_accumulator.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace bev_lidar_detection {
namespace {
using Matrix = std::array<std::array<double, 3>, 3>;

Matrix rotation(const std::array<double, 4> &q) {
  const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (!std::isfinite(n) || n < 1e-9) throw std::invalid_argument("Invalid sweep pose quaternion");
  const double x = q[0] / n, y = q[1] / n, z = q[2] / n, w = q[3] / n;
  return {{{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
           {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
           {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}}};
}
}  // namespace

SweepAccumulator::SweepAccumulator(std::size_t max_sweeps, double max_span_s)
    : max_sweeps_(max_sweeps), max_span_ns_(static_cast<std::int64_t>(max_span_s * 1e9)) {
  if (max_sweeps == 0 || !std::isfinite(max_span_s) || max_span_s < 0)
    throw std::invalid_argument("Invalid sweep accumulator limits");
}

void SweepAccumulator::add(std::vector<PointXYZI> points, std::int64_t stamp_ns, const SensorPose &pose) {
  rotation(pose.q);  // validate
  if (!sweeps_.empty() && stamp_ns <= sweeps_.back().stamp_ns) sweeps_.clear();
  sweeps_.push_back({std::move(points), stamp_ns, pose});
  while (sweeps_.size() > max_sweeps_) sweeps_.pop_front();
  while (sweeps_.size() > 1 && stamp_ns - sweeps_.front().stamp_ns > max_span_ns_) sweeps_.pop_front();
}

bool SweepAccumulator::refine_pose(std::int64_t stamp_ns, const SensorPose &pose) {
  rotation(pose.q);  // validate
  for (auto &s : sweeps_)
    if (std::llabs(s.stamp_ns - stamp_ns) <= 1000000) {
      s.pose = pose;
      return true;
    }
  return false;
}

void SweepAccumulator::build(std::vector<PointXYZI> &points, std::vector<float> &ages) const {
  points.clear();
  ages.clear();
  if (sweeps_.empty()) return;
  std::size_t total = 0;
  for (const auto &s : sweeps_) total += s.points.size();
  points.reserve(total);
  ages.reserve(total);
  const auto &newest = sweeps_.back();
  const Matrix rk = rotation(newest.pose.q);
  for (const auto &s : sweeps_) {
    const float age = static_cast<float>((newest.stamp_ns - s.stamp_ns) * 1e-9);
    if (&s == &newest) {
      points.insert(points.end(), s.points.begin(), s.points.end());
      ages.insert(ages.end(), s.points.size(), 0.0f);
      continue;
    }
    // p_newest = Rk^T (Rj p + tj - tk): sweep j's sensor frame -> world -> newest sensor frame.
    const Matrix rj = rotation(s.pose.q);
    Matrix r{};  // Rk^T Rj
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        for (int c = 0; c < 3; ++c) r[a][b] += rk[c][a] * rj[c][b];
    std::array<double, 3> t{};
    for (int a = 0; a < 3; ++a)
      for (int c = 0; c < 3; ++c) t[a] += rk[c][a] * (s.pose.t[c] - newest.pose.t[c]);
    for (const auto &p : s.points) {
      points.push_back({static_cast<float>(r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z + t[0]),
                        static_cast<float>(r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z + t[1]),
                        static_cast<float>(r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z + t[2]),
                        p.intensity});
      ages.push_back(age);
    }
  }
}

}  // namespace bev_lidar_detection
