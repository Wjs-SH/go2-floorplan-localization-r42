// calib_monitor — read-only /lowstate summariser for the calibration guide web UI.
// Subscribes /lowstate (BEST_EFFORT, never publishes anything to the robot) and writes a small JSON status file 10x/s
// (atomic rename). The web server reads only that file. Reported:
//   rate / last receive age / max tick gap (1 s), body gyro window stats, CUMULATIVE yaw-rate integral (tick time base)
//   so the server can compute bias-corrected rotation over any interval, wireless-remote stick values (Unitree remote only),
//   posture (thigh/calf joint angles, |tau_est|, foot force).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>

#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_state.hpp"

struct Smp { double rx, t, gx, gy, gz, stick; };

class CalibMonitor : public rclcpp::Node {
 public:
  CalibMonitor() : Node("sci_calib_monitor") {
    out_ = declare_parameter<std::string>("status_file", "/tmp/calib_monitor_status.json");
    sub_ = create_subscription<unitree_go::msg::LowState>("/lowstate", rclcpp::QoS(rclcpp::KeepLast(50)).best_effort(),
        [this](unitree_go::msg::LowState::ConstSharedPtr m) { on(*m); });
    timer_ = create_wall_timer(std::chrono::milliseconds(100), [this]() { write(); });
  }

 private:
  static double now() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }
  void on(const unitree_go::msg::LowState &m) {
    const double rx = now();
    if (have_ && m.tick == last_tick_) return;                     // re-sent sample (same tick)
    double dt = 0;
    if (have_) {
      int32_t d = (int32_t)(m.tick - last_tick_);
      if (d < 0 || d > 5000) { cum_t_ = 0; cum_gz_ = 0; resets_++; d = 0; }   // tick reset / jump: restart integrals
      dt = d * 1e-3;
    }
    const double gz = m.imu_state.gyroscope[2];
    if (have_ && dt > 0 && dt < 0.5) { cum_gz_ += 0.5 * (gz + last_gz_) * dt; cum_t_ += dt; }
    if (have_ && dt >= 0.05) { max_gap_ms_ = std::max(max_gap_ms_, dt * 1e3); }
    last_tick_ = m.tick; last_gz_ = gz; have_ = true; n_++;
    float st[5];
    std::memcpy(st, m.wireless_remote.data() + 4, sizeof(st));   // lx, rx, ry, L2, ly (Unitree xRockerBtnDataStruct)
    double sm = 0;
    for (int i : {0, 1, 2, 4}) if (std::isfinite(st[i])) sm = std::max(sm, (double)std::fabs(st[i]));
    win_.push_back(Smp{rx, cum_t_, m.imu_state.gyroscope[0], m.imu_state.gyroscope[1], gz, sm});
    while (!win_.empty() && rx - win_.front().rx > 1.0) win_.pop_front();
    double th = 0, ca = 0, tau = 0;
    for (int leg = 0; leg < 4; ++leg) { th += m.motor_state[leg * 3 + 1].q; ca += m.motor_state[leg * 3 + 2].q; }
    for (int j = 0; j < 12; ++j) tau += std::fabs(m.motor_state[j].tau_est);
    thigh_ = th / 4; calf_ = ca / 4; tau_ = tau / 12;
    for (int i = 0; i < 4; ++i) ff_[i] = m.foot_force[i];
    for (int i = 0; i < 5; ++i) stick_[i] = st[i];
    head_ok_ = m.wireless_remote[0] == 0x55 && m.wireless_remote[1] == 0x51;
  }

  void write() {
    const double t = now();
    double gzm = 0, gnmax = 0, smax = 0;
    std::vector<double> gn;
    for (auto &s : win_) { gzm += s.gz; double g = std::sqrt(s.gx * s.gx + s.gy * s.gy + s.gz * s.gz); gn.push_back(g); gnmax = std::max(gnmax, g); smax = std::max(smax, s.stick); }
    const size_t k = win_.size();
    if (k) gzm /= k;
    std::sort(gn.begin(), gn.end());
    const double gp95 = k ? gn[std::min(k - 1, (size_t)(0.95 * (k - 1) + 0.5))] : 0.0;
    std::ostringstream o;
    o.setf(std::ios::fixed); o.precision(5);
    o << "{\"wall\":" << t << ",\"n\":" << n_ << ",\"rate_hz_1s\":" << k << ",\"last_rx_age_s\":" << (win_.empty() ? 1e9 : t - win_.back().rx)
      << ",\"tick_gap_max_ms_1s\":" << max_gap_ms_ << ",\"tick_resets\":" << resets_
      << ",\"gz_mean_1s\":" << gzm << ",\"gyro_norm_max_1s\":" << gnmax << ",\"gyro_norm_p95_1s\":" << gp95
      << ",\"cum_t_s\":" << cum_t_ << ",\"cum_gz_rad\":" << cum_gz_
      << ",\"stick_max_abs_1s\":" << smax << ",\"stick\":[" << stick_[0] << "," << stick_[1] << "," << stick_[2] << "," << stick_[3] << "," << stick_[4]
      << "],\"remote_header_ok\":" << (head_ok_ ? "true" : "false")
      << ",\"posture\":{\"thigh\":" << thigh_ << ",\"calf\":" << calf_ << ",\"tau_mean\":" << tau_ << ",\"foot_force\":[" << ff_[0] << "," << ff_[1] << ","
      << ff_[2] << "," << ff_[3] << "]}}";
    max_gap_ms_ = 0;
    const std::string tmp = out_ + ".tmp";
    { std::ofstream f(tmp, std::ios::trunc); f << o.str() << '\n'; }
    std::rename(tmp.c_str(), out_.c_str());
  }

  std::string out_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::deque<Smp> win_;
  bool have_ = false, head_ok_ = false;
  uint32_t last_tick_ = 0;
  double last_gz_ = 0, cum_gz_ = 0, cum_t_ = 0, max_gap_ms_ = 0, thigh_ = 0, calf_ = 0, tau_ = 0;
  double stick_[5] = {0, 0, 0, 0, 0};
  int ff_[4] = {0, 0, 0, 0};
  uint64_t n_ = 0, resets_ = 0;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CalibMonitor>());
  rclcpp::shutdown();
  return 0;
}
