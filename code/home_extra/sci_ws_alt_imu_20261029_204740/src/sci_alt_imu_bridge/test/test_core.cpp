// Unit tests for the ROS-free bridge core (tick->system time, duplicates, resets, gaps, clock steps, values).
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "sci_alt_imu_bridge/core.hpp"

using namespace sci_alt_imu;

namespace {
const int64_t T0_SYS = 1793200000LL * 1000000000LL;   // arbitrary 2026 epoch (ns)
const int64_t T0_STEADY = 5000LL * 1000000000LL;

struct Sim {
  // true sample time of sensor tick k (ms) on the system clock: t = T0 + k*(1+drift) ms
  double drift = 0.0;            // ratio
  double base_latency_ms = 1.0;  // constant transport latency
  std::mt19937 rng{42};
  std::exponential_distribution<double> jitter{1.0 / 0.4};   // mean 0.4 ms extra delay
  uint32_t tick0 = 1000000;
  Sample make(int64_t k_ms, double extra_ms = 0, int64_t sys_step_ns = 0) {
    Sample s;
    s.tick = (uint32_t)(tick0 + k_ms);
    const double t_true_ns = (double)k_ms * 1e6 * (1.0 + drift);
    const double lat_ns = (base_latency_ms + jitter(rng) + extra_ms) * 1e6;
    s.rx_steady_ns = T0_STEADY + (int64_t)(t_true_ns + lat_ns);
    s.rx_sys_ns = T0_SYS + (int64_t)(t_true_ns + lat_ns) + sys_step_ns;
    const float v = (float)std::sin(k_ms * 0.01);      // varying payload
    s.gyro = {0.01f * v, 0.002f, -0.001f};
    s.acc = {0.88f + 0.01f * v, 0.07f, 9.49f};
    s.quat = {1.f, 0.f, 0.f, 0.001f * v};
    return s;
  }
  double true_ns(int64_t k_ms) const { return (double)k_ms * 1e6 * (1.0 + drift); }
};

// run k_ms = 0,2,4,... with 1 ms quantization pattern (1/2/3 ms steps like the recorded data)
std::vector<int64_t> tick_sequence(int n) {
  std::vector<int64_t> ks;
  int64_t k = 0;
  const int pattern[] = {2, 2, 2, 1, 3, 2, 2, 2, 2, 2};
  for (int i = 0; i < n; ++i) { ks.push_back(k); k += pattern[i % 10]; }
  return ks;
}
}  // namespace

TEST(Clock, StampsMonotonicAndTrackTrueTimeWithinLatency) {
  Sim sim;
  Bridge b;
  int64_t last = 0, stamp = 0;
  int published = 0;
  double max_err_ms = 0;
  for (int64_t k : tick_sequence(15000)) {   // ~30 s
    if (b.process(sim.make(k), &stamp) == Verdict::PUBLISH) {
      EXPECT_GT(stamp, last);
      last = stamp;
      ++published;
      // stamp - (true time + constant latency) must stay small: the constant latency is not observable (calibrated later)
      const double err_ms = ((double)(stamp - T0_SYS) - sim.true_ns(k)) * 1e-6 - sim.base_latency_ms;
      max_err_ms = std::max(max_err_ms, std::fabs(err_ms));
    }
  }
  EXPECT_GE(published, 13900);   // 15000 samples minus ~1000 held back during the 2 s clock convergence
  EXPECT_LT(max_err_ms, 0.6);
  EXPECT_EQ(b.health(), Health::READY);
}

TEST(Clock, NoPublishBeforeConvergence) {
  Sim sim;
  Bridge b;
  int64_t stamp = 0;
  for (int64_t k = 0; k < 1500; k += 2) {     // 1.5 s < 4 completed bins of 0.5 s
    EXPECT_NE(b.process(sim.make(k), &stamp), Verdict::PUBLISH);
  }
  EXPECT_EQ(b.health(), Health::INITIALIZING);
}

TEST(Clock, DriftEstimated) {
  Sim sim;
  sim.drift = 50e-6;   // 50 ppm
  Bridge b;
  int64_t stamp = 0;
  for (int64_t k : tick_sequence(30000)) b.process(sim.make(k), &stamp);
  EXPECT_NEAR(b.clock().drift_ppm(), 50.0, 5.0);
}

TEST(Duplicates, IdenticalPayloadRepeatDroppedDifferentPayloadAmbiguous) {
  Sim sim;
  Bridge b;
  int64_t stamp = 0;
  for (int64_t k = 0; k < 4000; k += 2) b.process(sim.make(k), &stamp);
  Sample s = sim.make(4000);
  EXPECT_EQ(b.process(s, &stamp), Verdict::PUBLISH);
  Sample rep = s;
  rep.rx_sys_ns += 500000; rep.rx_steady_ns += 500000;
  EXPECT_EQ(b.process(rep, &stamp), Verdict::DROP_REPEAT);
  Sample diff = rep;
  diff.gyro[0] += 0.5f;
  EXPECT_EQ(b.process(diff, &stamp), Verdict::DROP_SAME_TICK_DIFFERENT);
  EXPECT_EQ(b.counters().repeat, 1u);
  EXPECT_EQ(b.counters().same_tick_diff, 1u);
  EXPECT_EQ(b.health(), Health::DEGRADED);
}

TEST(Ticks, Uint32RolloverIsContinuous) {
  Sim sim;
  sim.tick0 = 0xFFFFFFFFu - 3000u;   // wraps after 3 s
  Bridge b;
  int64_t stamp = 0, last = 0;
  int pub = 0;
  for (int64_t k = 0; k < 8000; k += 2) {
    if (b.process(sim.make(k), &stamp) == Verdict::PUBLISH) { EXPECT_GT(stamp, last); last = stamp; ++pub; }
  }
  EXPECT_EQ(b.counters().tick_resets, 0u);
  EXPECT_EQ(b.counters().model_resets, 0u);
  EXPECT_GT(pub, 2500);
}

TEST(Ticks, BackwardTickResetsModelAndBlocksUntilReconverged) {
  Sim sim;
  Bridge b;
  int64_t stamp = 0;
  for (int64_t k = 0; k < 4000; k += 2) b.process(sim.make(k), &stamp);
  Sim sim2;              // Go2 rebooted: tick restarts near 0, time continues
  sim2.tick0 = 10;
  Sample s = sim2.make(0);
  s.rx_sys_ns = T0_SYS + 4002LL * 1000000; s.rx_steady_ns = T0_STEADY + 4002LL * 1000000;
  EXPECT_NE(b.process(s, &stamp), Verdict::PUBLISH);
  EXPECT_EQ(b.counters().tick_resets, 1u);
  EXPECT_EQ(b.health(), Health::CLOCK_UNTRUSTED);
}

TEST(SystemClock, StepDetectedAndPublishingBlocked) {
  Sim sim;
  Bridge b;
  int64_t stamp = 0, last = 0;
  for (int64_t k = 0; k < 4000; k += 2)
    if (b.process(sim.make(k), &stamp) == Verdict::PUBLISH) last = stamp;
  EXPECT_EQ(b.process(sim.make(4002, 0, 2000000000LL), &stamp), Verdict::DROP_CLOCK_UNTRUSTED);   // +2 s NTP step
  EXPECT_EQ(b.counters().clock_steps, 1u);
  EXPECT_EQ(b.health(), Health::CLOCK_UNTRUSTED);
  // after re-convergence, stamps are on the stepped clock and still strictly after the last published one
  int pub = 0;
  for (int64_t k = 4004; k < 9000; k += 2)
    if (b.process(sim.make(k, 0, 2000000000LL), &stamp) == Verdict::PUBLISH) { EXPECT_GT(stamp, last); last = stamp; ++pub; }
  EXPECT_GT(pub, 1000);
}

TEST(Gaps, ShortGapAndL1LikeGapGoStaleThenRecover) {
  Sim sim;
  Bridge b;
  int64_t stamp = 0;
  int64_t k = 0;
  for (; k < 4000; k += 2) b.process(sim.make(k), &stamp);
  k += 60;                                    // 60 ms receive gap
  b.process(sim.make(k), &stamp);
  EXPECT_EQ(b.health(), Health::STALE);
  for (k += 2; k < 5600; k += 2) b.process(sim.make(k), &stamp);
  EXPECT_NE(b.health(), Health::STALE);
  // watchdog detects a 7.67 s outage while nothing arrives
  const int64_t last_steady = sim.make(k - 2).rx_steady_ns;
  b.tick_watchdog(last_steady + 100LL * 1000000);
  EXPECT_EQ(b.health(), Health::STALE);
  EXPECT_EQ(b.counters().gaps, 2u);
  k += 7670;
  b.process(sim.make(k), &stamp);
  EXPECT_EQ(b.counters().gaps, 2u);             // same gap not double counted
  for (k += 2; k < 15000; k += 2) b.process(sim.make(k), &stamp);
  EXPECT_NE(b.health(), Health::STALE);
}

TEST(Gaps, DdsBurstKeepsTickBasedStamps) {
  Sim sim;
  Bridge b;
  int64_t stamp = 0;
  for (int64_t k = 0; k < 6000; k += 2) b.process(sim.make(k), &stamp);
  // 40 ms of samples delayed and delivered in a burst: stamps must stay at the true sample times
  double max_err = 0;
  for (int64_t k = 6000; k < 6040; k += 2) {
    const double extra = 40.0 - (k - 6000);
    if (b.process(sim.make(k, extra), &stamp) == Verdict::PUBLISH)
      max_err = std::max(max_err, std::fabs(((double)(stamp - T0_SYS) - sim.true_ns(k)) * 1e-6 - sim.base_latency_ms));
  }
  EXPECT_LT(max_err, 0.8);
}

TEST(Values, NanInfAndImplausibleRejectedNotAltered) {
  Sim sim;
  Bridge b;
  int64_t stamp = 0;
  for (int64_t k = 0; k < 4000; k += 2) b.process(sim.make(k), &stamp);
  Sample s = sim.make(4000);
  s.gyro[1] = NAN;
  EXPECT_EQ(b.process(s, &stamp), Verdict::DROP_INVALID_VALUE);
  s = sim.make(4002);
  s.acc[2] = INFINITY;
  EXPECT_EQ(b.process(s, &stamp), Verdict::DROP_INVALID_VALUE);
  s = sim.make(4004);
  s.gyro[0] = 100.f;
  EXPECT_EQ(b.process(s, &stamp), Verdict::DROP_INVALID_VALUE);
  EXPECT_EQ(b.counters().invalid, 3u);
  s = sim.make(4006);
  EXPECT_EQ(b.process(s, &stamp), Verdict::PUBLISH);
}

TEST(Values, FrozenPayloadGoesStale) {
  Sim sim;
  Bridge b;
  int64_t stamp = 0;
  for (int64_t k = 0; k < 4000; k += 2) b.process(sim.make(k), &stamp);
  Sample frozen = sim.make(4000);
  for (int64_t k = 4002; k < 4100; k += 2) {
    Sample s = sim.make(k);
    s.gyro = frozen.gyro; s.acc = frozen.acc; s.quat = frozen.quat;
    b.process(s, &stamp);
  }
  EXPECT_EQ(b.health(), Health::STALE);
  EXPECT_EQ(b.counters().frozen_events, 1u);
}
