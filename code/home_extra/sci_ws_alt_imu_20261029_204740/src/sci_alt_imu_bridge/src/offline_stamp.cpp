// offline_stamp — run the SAME bridge core on recorded LowState samples (verification/replay only).
// stdin CSV: tick,rx_sys_ns,gx,gy,gz,ax,ay,az   (rx_sys_ns = recorder receive time; also used as steady time: no clock steps offline)
// stdout CSV: verdict,tick,rx_sys_ns,stamp_ns,gx,gy,gz,ax,ay,az   (stamp_ns only for PUBLISH)
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include "sci_alt_imu_bridge/core.hpp"
using namespace sci_alt_imu;
int main() {
  Bridge b;
  std::string line;
  std::ios::sync_with_stdio(false);
  while (std::getline(std::cin, line)) {
    if (line.empty() || line[0] == 't') continue;
    std::istringstream is(line);
    Sample s; char c; unsigned long long tick; long long rx;
    is >> tick >> c >> rx >> c >> s.gyro[0] >> c >> s.gyro[1] >> c >> s.gyro[2] >> c >> s.acc[0] >> c >> s.acc[1] >> c >> s.acc[2];
    s.tick = (uint32_t)tick; s.rx_sys_ns = rx; s.rx_steady_ns = rx;
    int64_t stamp = 0;
    Verdict v = b.process(s, &stamp);
    std::printf("%s,%llu,%lld,%lld,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", to_string(v), tick, rx, (long long)(v == Verdict::PUBLISH ? stamp : 0),
                s.gyro[0], s.gyro[1], s.gyro[2], s.acc[0], s.acc[1], s.acc[2]);
  }
  const Counters &k = b.counters();
  std::fprintf(stderr, "received=%llu published=%llu repeat=%llu same_tick_diff=%llu gaps=%llu tick_jumps=%llu resets=%llu neg=%llu drift_ppm=%.3f health=%s\n",
               (unsigned long long)k.received, (unsigned long long)k.published, (unsigned long long)k.repeat, (unsigned long long)k.same_tick_diff,
               (unsigned long long)k.gaps, (unsigned long long)k.tick_jumps, (unsigned long long)k.model_resets, (unsigned long long)k.neg_residual,
               b.clock().drift_ppm(), to_string(b.health()));
  return 0;
}
