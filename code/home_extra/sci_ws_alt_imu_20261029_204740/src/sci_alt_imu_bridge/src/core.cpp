#include "sci_alt_imu_bridge/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace sci_alt_imu {

const char *to_string(Health h) {
  switch (h) {
    case Health::INITIALIZING: return "INITIALIZING";
    case Health::READY: return "READY";
    case Health::DEGRADED: return "DEGRADED";
    case Health::STALE: return "STALE";
    case Health::CLOCK_UNTRUSTED: return "CLOCK_UNTRUSTED";
  }
  return "?";
}

const char *to_string(Verdict v) {
  switch (v) {
    case Verdict::PUBLISH: return "PUBLISH";
    case Verdict::DROP_REPEAT: return "DROP_REPEAT";
    case Verdict::DROP_SAME_TICK_DIFFERENT: return "DROP_SAME_TICK_DIFFERENT";
    case Verdict::DROP_INVALID_VALUE: return "DROP_INVALID_VALUE";
    case Verdict::DROP_NOT_MONOTONIC: return "DROP_NOT_MONOTONIC";
    case Verdict::DROP_CLOCK_UNTRUSTED: return "DROP_CLOCK_UNTRUSTED";
  }
  return "?";
}

// ------------------------------------------------------------------ TickClock
void TickClock::reset() {
  bins_.clear();
  have_tick0_ = false;
  ready_ = false;
  a_fit_ = drift_fit_ = a_ = drift_ = 0;
}

// Internal time base: t = (tick - tick0) ms; offset value off = (rx_sys - base) - t  [ns], base = sys time of tick0's first
// sample, kept as integer to avoid double precision loss at 1.8e18 ns.
void TickClock::observe(int64_t tick_ms, int64_t rx_sys_ns) {
  if (!have_tick0_) {
    have_tick0_ = true;
    tick0_ = tick_ms;
    bins_.clear();
    bins_.push_back(Bin{0, tick_ms, 0.0});
    base_sys_ns_ = rx_sys_ns;
    return;
  }
  const int64_t dt_ms = tick_ms - tick0_;
  const double off = (double)(rx_sys_ns - base_sys_ns_) - (double)dt_ms * 1e6;
  const int64_t key = (int64_t)std::floor((double)dt_ms / (p_.bin_s * 1000.0));
  if (bins_.empty() || key != bins_.back().key) {
    const bool completed = !bins_.empty();
    bins_.push_back(Bin{key, tick_ms, off});
    while ((int)bins_.size() > p_.max_bins) bins_.pop_front();
    if (completed) refit();
  } else if (off < bins_.back().min_off_ns) {
    bins_.back().min_off_ns = off;
    bins_.back().tick_ms = tick_ms;
  }
}

void TickClock::refit() {
  // completed bins = all but the last (still filling)
  const int n = (int)bins_.size() - 1;
  if (n < p_.min_bins_ready) return;
  std::vector<int> use;
  for (int i = 0; i < n; ++i) use.push_back(i);
  double a = 0, b = 0;
  for (int pass = 0; pass < 2; ++pass) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i : use) {
      const double x = (double)(bins_[i].tick_ms - tick0_) * 1e6, y = bins_[i].min_off_ns;
      sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    const double m = (double)use.size();
    const double den = m * sxx - sx * sx;
    b = (den > 0) ? (m * sxy - sx * sy) / den : 0.0;
    a = (sy - b * sx) / m;
    if (pass == 0) {                      // reject bins whose minimum is > 2 ms above the line (whole bin late)
      std::vector<int> keep;
      for (int i : use) {
        const double x = (double)(bins_[i].tick_ms - tick0_) * 1e6;
        if (bins_[i].min_off_ns - (a + b * x) <= 2e6) keep.push_back(i);
      }
      if ((int)keep.size() >= p_.min_bins_ready) use.swap(keep);
    }
  }
  if (std::fabs(b) * 1e6 > p_.max_drift_ppm) {   // implausible clock ratio: do not trust
    ready_ = false;
    return;
  }
  a_fit_ = a;
  drift_fit_ = b;
  const double x_now = (double)(bins_.back().tick_ms - tick0_) * 1e6;
  if (!ready_) {
    a_ = a_fit_;
    drift_ = drift_fit_;
    ready_ = true;
    return;
  }
  // slew: move the applied offset at the current tick toward the fitted one by at most slew_max per bin
  const double applied_now = a_ + drift_ * x_now, target_now = a_fit_ + drift_fit_ * x_now;
  const double lim = p_.slew_max_ms_per_bin * 1e6;
  const double delta = std::max(-lim, std::min(lim, target_now - applied_now));
  drift_ = drift_fit_;
  a_ = applied_now + delta - drift_ * x_now;
}

int64_t TickClock::stamp_ns(int64_t tick_ms) const {
  const double x = (double)(tick_ms - tick0_) * 1e6;
  return base_sys_ns_ + (tick_ms - tick0_) * 1000000LL + (int64_t)std::llround(a_ + drift_ * x);
}

// ------------------------------------------------------------------ Bridge
static bool finite3(const std::array<float, 3> &v) {
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}
static double norm3(const std::array<float, 3> &v) {
  return std::sqrt((double)v[0] * v[0] + (double)v[1] * v[1] + (double)v[2] * v[2]);
}
static bool same_payload(const Sample &a, const Sample &b) {
  return std::memcmp(a.gyro.data(), b.gyro.data(), sizeof(a.gyro)) == 0 &&
         std::memcmp(a.acc.data(), b.acc.data(), sizeof(a.acc)) == 0 &&
         std::memcmp(a.quat.data(), b.quat.data(), sizeof(a.quat)) == 0;
}

void Bridge::set_health(Health h, const std::string &why) {
  health_ = h;
  reason_ = why;
}

void Bridge::note_anomaly(int64_t steady_ns, const std::string &why) {
  last_anomaly_steady_ns_ = steady_ns;
  if (health_ == Health::READY) set_health(Health::DEGRADED, why);
}

void Bridge::full_reset(const std::string &why) {
  clock_.reset();
  c_.model_resets++;
  neg_residual_run_ = 0;
  set_health(Health::CLOCK_UNTRUSTED, why);
}

void Bridge::tick_watchdog(int64_t now_steady_ns) {
  if (!have_last_) return;
  const double gap_ms = (now_steady_ns - last_rx_steady_ns_) * 1e-6;
  if (gap_ms > p_.gap_ms && health_ != Health::STALE) {
    c_.gaps++;
    gap_open_ = true;
    stale_since_ok_ns_ = -1;
    set_health(Health::STALE, "no LowState for " + std::to_string((int)gap_ms) + " ms");
  }
}

Verdict Bridge::process(const Sample &s, int64_t *out_stamp_ns) {
  c_.received++;
  // 1. system clock step (NTP step, manual set): system - steady jumps between consecutive messages
  const int64_t d = s.rx_sys_ns - s.rx_steady_ns;
  if (have_sys_steady_ && std::llabs(d - sys_minus_steady_ns_) > (int64_t)(p_.clock_step_ms * 1e6)) {
    c_.clock_steps++;
    full_reset("SYSTEM_CLOCK_STEP " + std::to_string((d - sys_minus_steady_ns_) / 1000000) + " ms");
  }
  sys_minus_steady_ns_ = d;
  have_sys_steady_ = true;

  // 2. receive gap
  if (have_last_) {
    const double gap_ms = (s.rx_steady_ns - last_rx_steady_ns_) * 1e-6;
    if (gap_ms > p_.gap_ms) {
      if (!gap_open_) c_.gaps++;
      gap_open_ = false;
      stale_since_ok_ns_ = s.rx_steady_ns;
      set_health(Health::STALE, "input gap " + std::to_string((int)gap_ms) + " ms");
    } else if (gap_open_) {
      gap_open_ = false;
      stale_since_ok_ns_ = s.rx_steady_ns;
    }
  }

  // 3. tick unwrap / duplicates / resets
  if (!have_last_) {
    tick_unwrapped_ = (int64_t)s.tick;
  } else {
    const int32_t diff = (int32_t)(s.tick - last_raw_tick_);   // wrap-safe for uint32 rollover
    if (diff == 0) {
      last_rx_steady_ns_ = s.rx_steady_ns;
      if (same_payload(s, last_)) { c_.repeat++; return Verdict::DROP_REPEAT; }
      c_.same_tick_diff++;
      note_anomaly(s.rx_steady_ns, "same tick, different payload");
      last_ = s;
      return Verdict::DROP_SAME_TICK_DIFFERENT;
    }
    if (diff < 0) {
      if (-diff > p_.tick_back_reset_ms) {
        c_.tick_resets++;
        full_reset("TICK_RESET (tick went back " + std::to_string(-diff) + " ms)");
        tick_unwrapped_ = (int64_t)s.tick + ((tick_unwrapped_ >> 32) + 1) * 4294967296LL;   // keep unwrapped ticks increasing
      } else {
        last_rx_steady_ns_ = s.rx_steady_ns;
        c_.not_monotonic++;
        return Verdict::DROP_NOT_MONOTONIC;
      }
    } else {
      if (diff > p_.tick_jump_ms) {
        c_.tick_jumps++;
        const double rx_gap_ms = (s.rx_steady_ns - last_rx_steady_ns_) * 1e-6;
        if (std::fabs(rx_gap_ms - diff) > 50.0) full_reset("TICK_JUMP " + std::to_string(diff) + " ms vs receive gap " +
                                                          std::to_string((int)rx_gap_ms) + " ms");
        else note_anomaly(s.rx_steady_ns, "tick jump with matching receive gap");
      }
      tick_unwrapped_ += diff;
    }
  }
  const bool frozen = have_last_ && same_payload(s, last_);
  last_raw_tick_ = s.tick;
  last_rx_steady_ns_ = s.rx_steady_ns;
  last_ = s;
  have_last_ = true;

  // 4. value validity (untouched values; only rejected, never altered)
  const double an = norm3(s.acc);
  if (!finite3(s.gyro) || !finite3(s.acc) || norm3(s.gyro) > p_.max_gyro || an < p_.min_acc || an > p_.max_acc) {
    c_.invalid++;
    note_anomaly(s.rx_steady_ns, "invalid IMU value");
    return Verdict::DROP_INVALID_VALUE;
  }
  // 5. frozen payload (sensor stuck while tick advances)
  if (frozen) {
    if (++frozen_run_ == p_.frozen_payload_count) {
      c_.frozen_events++;
      stale_since_ok_ns_ = -1;
      set_health(Health::STALE, "IMU payload frozen for " + std::to_string(frozen_run_) + " samples");
    }
  } else {
    if (frozen_run_ >= p_.frozen_payload_count) stale_since_ok_ns_ = s.rx_steady_ns;
    frozen_run_ = 0;
  }

  // 6. clock model
  clock_.observe(tick_unwrapped_, s.rx_sys_ns);
  if (!clock_.ready()) {
    c_.blocked_clock++;
    if (health_ != Health::STALE && health_ != Health::CLOCK_UNTRUSTED) set_health(Health::INITIALIZING, "clock model converging");
    return Verdict::DROP_CLOCK_UNTRUSTED;
  }
  const int64_t stamp = clock_.stamp_ns(tick_unwrapped_);
  const double residual_ms = (s.rx_sys_ns - stamp) * 1e-6;
  if (residual_ms < -p_.neg_residual_ms) {     // stamp later than the receive time: physically impossible
    c_.neg_residual++;
    if (++neg_residual_run_ >= p_.neg_residual_reset_count) full_reset("MODEL_INCONSISTENT (stamps ahead of receipt)");
    else note_anomaly(s.rx_steady_ns, "stamp ahead of receipt");
    c_.blocked_clock++;
    return Verdict::DROP_CLOCK_UNTRUSTED;
  }
  neg_residual_run_ = 0;
  if (have_pub_ && stamp <= last_pub_stamp_ns_) {
    c_.not_monotonic++;
    return Verdict::DROP_NOT_MONOTONIC;
  }

  // 7. health
  if (health_ == Health::STALE) {
    if (frozen_run_ < p_.frozen_payload_count && stale_since_ok_ns_ >= 0 &&
        (s.rx_steady_ns - stale_since_ok_ns_) * 1e-9 >= p_.recover_s)
      set_health(Health::READY, "recovered");
  } else if (health_ == Health::INITIALIZING || health_ == Health::CLOCK_UNTRUSTED) {
    set_health(Health::READY, "clock model converged");
  }
  if (health_ == Health::DEGRADED && last_anomaly_steady_ns_ >= 0 &&
      (s.rx_steady_ns - last_anomaly_steady_ns_) * 1e-9 > p_.degraded_hold_s)
    set_health(Health::READY, "no anomaly for " + std::to_string((int)p_.degraded_hold_s) + " s");
  if (health_ == Health::READY && last_anomaly_steady_ns_ >= 0 &&
      (s.rx_steady_ns - last_anomaly_steady_ns_) * 1e-9 <= p_.degraded_hold_s)
    health_ = Health::DEGRADED;

  last_residual_ms_ = residual_ms;
  last_pub_stamp_ns_ = stamp;
  have_pub_ = true;
  c_.published++;
  *out_stamp_ns = stamp;
  return Verdict::PUBLISH;
}

}  // namespace sci_alt_imu
