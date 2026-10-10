#pragma once

#include <cstdint>

namespace bev_visualization {

// A source timestamp may restart only when the simulation clock also rewinds.
// Keep source ordering separate from wall-clock freshness and receipt age.
class ReplaySourceClock {
 public:
  enum class SourceOrder { Forward, Duplicate, Backwards };

  bool observe_clock(bool use_sim_time, std::int64_t ros_time_ns) {
    const bool rewind = use_sim_time && ros_time_ns < last_ros_time_ns_ &&
      last_ros_time_ns_ - ros_time_ns > 50000000LL;
    last_ros_time_ns_ = ros_time_ns;
    if (rewind) {
      last_processed_stamp_ns_ = 0;
      ++simulation_clock_rewinds_;
    }
    return rewind;
  }

  SourceOrder source_order(std::int64_t source_stamp_ns) const {
    if (source_stamp_ns == last_processed_stamp_ns_) return SourceOrder::Duplicate;
    if (source_stamp_ns < last_processed_stamp_ns_) return SourceOrder::Backwards;
    return SourceOrder::Forward;
  }

  void mark_processed(std::int64_t source_stamp_ns) {
    last_processed_stamp_ns_ = source_stamp_ns;
  }

  std::int64_t last_processed_stamp_ns() const { return last_processed_stamp_ns_; }
  std::uint64_t simulation_clock_rewinds() const { return simulation_clock_rewinds_; }

 private:
  std::int64_t last_ros_time_ns_{};
  std::int64_t last_processed_stamp_ns_{};
  std::uint64_t simulation_clock_rewinds_{};
};

}  // namespace bev_visualization
