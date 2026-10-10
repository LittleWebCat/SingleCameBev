#include "bev_visualization/replay_source_clock.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

using bev_visualization::ReplaySourceClock;
using Order = ReplaySourceClock::SourceOrder;

namespace {
constexpr std::int64_t second = 1000000000LL;
void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

void forward_clock_does_not_allow_backwards_source() {
  ReplaySourceClock clock;
  require(!clock.observe_clock(true, 100 * second), "Initial simulated clock must not rewind");
  clock.mark_processed(99 * second);
  require(!clock.observe_clock(true, 101 * second), "Forward clock incorrectly rewound");
  require(clock.source_order(98 * second) == Order::Backwards,
          "An out-of-order source image must remain rejected with a forward clock");
  require(clock.source_order(99 * second) == Order::Duplicate, "Repeated source image must remain duplicate");
  require(clock.simulation_clock_rewinds() == 0, "Source disorder counted as a simulation clock rewind");
}

void actual_simulation_loop_resets_source_gate_once() {
  ReplaySourceClock clock;
  clock.observe_clock(true, 100 * second);
  clock.mark_processed(99 * second);
  require(clock.observe_clock(true, 10 * second), "Actual simulated clock restart must reset ordering");
  require(clock.last_processed_stamp_ns() == 0, "Previous loop's source stamp survived rewind");
  require(clock.source_order(10 * second) == Order::Forward,
          "New loop's fresh source frame must be accepted after clock reset");
  clock.mark_processed(10 * second);
  require(!clock.observe_clock(true, 10 * second), "Unchanged clock reset state a second time");
  require(clock.source_order(10 * second) == Order::Duplicate, "New loop's duplicate image was accepted");
  require(clock.source_order(9 * second) == Order::Backwards, "New loop's ordering gate remained disabled");
  require(clock.source_order(11 * second) == Order::Forward, "Next new-loop image was rejected");
  require(clock.simulation_clock_rewinds() == 1, "One clock rewind must count exactly once");
}

void live_clock_rewind_keeps_source_gate() {
  ReplaySourceClock clock;
  clock.observe_clock(false, 100 * second);
  clock.mark_processed(99 * second);
  require(!clock.observe_clock(false, 10 * second), "Live clock rewind must not reset source ordering");
  require(clock.source_order(10 * second) == Order::Backwards,
          "Live nonmonotonic camera frame was incorrectly permitted");
  require(clock.simulation_clock_rewinds() == 0, "Live clock correction counted as a bag loop");
}

void only_jumps_over_fifty_milliseconds_reset() {
  ReplaySourceClock clock;
  clock.observe_clock(true, second);
  clock.mark_processed(900000000LL);
  require(!clock.observe_clock(true, 950000000LL), "Exactly 50 ms backwards must not reset");
  require(clock.source_order(800000000LL) == Order::Backwards, "Small clock jump reset source state");
  require(clock.observe_clock(true, 899999999LL), "Jump over 50 ms must reset");
  require(clock.source_order(800000000LL) == Order::Forward, "Clock rewind did not restore source progress");
}
}  // namespace

int main() {
  try {
    forward_clock_does_not_allow_backwards_source();
    actual_simulation_loop_resets_source_gate_once();
    live_clock_rewind_keeps_source_gate();
    only_jumps_over_fifty_milliseconds_reset();
    std::cout << "Replay source clock tests passed: simulated loops, source disorder, live clocks and rewind threshold\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
