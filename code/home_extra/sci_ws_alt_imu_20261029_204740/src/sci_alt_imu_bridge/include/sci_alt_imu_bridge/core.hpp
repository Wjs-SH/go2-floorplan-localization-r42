// sci_alt_imu_bridge core — ROS-free logic (unit tested with gtest).
//
// Go2 LowState carries no header stamp, only a uint32 `tick` (verified in recorded data: 1 count == 1 ms, 65.438 s of
// ticks over 65.435 s of receive time). The XT16 driver stamps point clouds with gettimeofday() at UDP receipt
// (Hesai use_timestamp_type: 1), i.e. the Jetson CLOCK_REALTIME / ROS system clock. This core therefore maps
//   tick (sensor clock, ms)  ->  Jetson system time
// with an affine model fitted to the LOWER ENVELOPE of (receive_time - tick) per time bin (minimum transport latency),
// so DDS jitter / bursts do not leak into the published stamps. The unknown constant transport latency is NOT guessed:
// the stamp means "earliest possible receive time of that tick"; the residual constant is part of the LiDAR-IMU
// time-offset calibration (FAST-LIO common.time_offset_lidar_to_imu).
#pragma once
#include <array>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace sci_alt_imu {

struct Sample {               // one LowState.imu_state as received
  uint32_t tick = 0;          // LowState.tick
  int64_t rx_sys_ns = 0;      // system clock (CLOCK_REALTIME / ROS system time) at callback entry
  int64_t rx_steady_ns = 0;   // steady clock at callback entry (clock-step detection, gap timing)
  std::array<float, 3> gyro{};   // rad/s, Go2 body IMU frame, untouched
  std::array<float, 3> acc{};    // m/s^2, Go2 body IMU frame, untouched
  std::array<float, 4> quat{};   // only used for duplicate classification
};

enum class Verdict {
  PUBLISH,                 // stamp valid, publish
  DROP_REPEAT,             // same tick, bit-identical payload: re-sent sample
  DROP_SAME_TICK_DIFFERENT,// same tick, different payload: ambiguous timing -> not published, counted
  DROP_INVALID_VALUE,      // NaN / inf / implausible magnitude
  DROP_NOT_MONOTONIC,      // would not be strictly after the last published stamp
  DROP_CLOCK_UNTRUSTED,    // clock model not (re)converged: publishing is blocked
};

enum class Health { INITIALIZING, READY, DEGRADED, STALE, CLOCK_UNTRUSTED };
const char *to_string(Health h);
const char *to_string(Verdict v);

struct Params {
  double bin_s = 0.5;                 // lower-envelope bin length
  int min_bins_ready = 4;             // bins needed before the first stamp is trusted (2 s)
  int max_bins = 120;                 // fit window (60 s)
  double max_drift_ppm = 500.0;       // |drift| above this -> model rejected
  double slew_max_ms_per_bin = 0.2;   // applied-offset change limit per bin after READY
  double neg_residual_ms = 3.0;       // receive earlier than model by more -> model inconsistent
  int neg_residual_reset_count = 20;  // consecutive inconsistent samples -> reset
  double clock_step_ms = 20.0;        // |Δ(system - steady)| above this -> system clock stepped
  double gap_ms = 50.0;               // receive gap -> STALE
  double recover_s = 1.0;             // continuous good input after STALE before READY again
  int64_t tick_jump_ms = 1000;        // forward tick jump -> gap event (model kept)
  int64_t tick_back_reset_ms = 5;     // backward tick beyond this -> sensor reset, model reset
  int frozen_payload_count = 25;      // identical payload with advancing tick (50 ms @500 Hz) -> STALE
  double max_gyro = 35.0;             // rad/s
  double min_acc = 1.0, max_acc = 80.0;   // m/s^2 norm (dynamic, not gravity-only)
  double degraded_hold_s = 5.0;       // anomalies within this window -> DEGRADED
};

struct Counters {
  uint64_t received = 0, published = 0, repeat = 0, same_tick_diff = 0, invalid = 0, not_monotonic = 0,
           blocked_clock = 0, gaps = 0, tick_jumps = 0, tick_resets = 0, clock_steps = 0, model_resets = 0,
           neg_residual = 0, frozen_events = 0;
};

class TickClock {
 public:
  explicit TickClock(const Params &p = Params()) : p_(p) { reset(); }
  void reset();
  // feed a sample (already unwrapped tick in ms) into the lower-envelope fit
  void observe(int64_t tick_ms, int64_t rx_sys_ns);
  bool ready() const { return ready_; }
  // stamp in system-time ns for an unwrapped tick (requires ready())
  int64_t stamp_ns(int64_t tick_ms) const;
  double drift_ppm() const { return drift_ * 1e6; }
  double offset_ms_at(int64_t tick_ms) const { return (double)(stamp_ns(tick_ms) - tick_ms * 1000000LL) * 1e-6; }
  int bins() const { return (int)bins_.size(); }

 private:
  struct Bin { int64_t key; int64_t tick_ms; double min_off_ns; };
  void refit();
  Params p_;
  std::deque<Bin> bins_;
  int64_t tick0_ = 0;
  int64_t base_sys_ns_ = 0;
  bool have_tick0_ = false;
  // fitted model: off(tick) = a_ + drift_ * (tick - tick0)[ns] ; applied model slews toward fitted one
  double a_fit_ = 0, drift_fit_ = 0, a_ = 0, drift_ = 0;
  bool ready_ = false;
};

class Bridge {
 public:
  explicit Bridge(const Params &p = Params()) : p_(p), clock_(p) {}
  // process one sample; on PUBLISH, out_stamp_ns holds the stamp to publish
  Verdict process(const Sample &s, int64_t *out_stamp_ns);
  // call periodically (e.g. 10 Hz) with steady time to detect input gaps while no message arrives
  void tick_watchdog(int64_t now_steady_ns);
  Health health() const { return health_; }
  const std::string &reason() const { return reason_; }
  const Counters &counters() const { return c_; }
  const TickClock &clock() const { return clock_; }
  double last_residual_ms() const { return last_residual_ms_; }   // receive - stamp of the last published sample
  int64_t unwrapped_tick() const { return tick_unwrapped_; }

 private:
  void set_health(Health h, const std::string &why);
  void note_anomaly(int64_t steady_ns, const std::string &why);
  void full_reset(const std::string &why);
  Params p_;
  TickClock clock_;
  Counters c_;
  Health health_ = Health::INITIALIZING;
  std::string reason_ = "waiting for samples";
  bool have_last_ = false;
  Sample last_{};
  uint32_t last_raw_tick_ = 0;
  int64_t tick_unwrapped_ = 0;
  int64_t last_pub_stamp_ns_ = 0;
  bool have_pub_ = false;
  bool gap_open_ = false;
  bool have_sys_steady_ = false;
  int64_t sys_minus_steady_ns_ = 0;
  int neg_residual_run_ = 0;
  int frozen_run_ = 0;
  int64_t last_rx_steady_ns_ = 0;
  int64_t stale_since_ok_ns_ = -1;     // steady time when input became continuous again after STALE
  int64_t last_anomaly_steady_ns_ = -1;
  double last_residual_ms_ = 0;
};

}  // namespace sci_alt_imu
