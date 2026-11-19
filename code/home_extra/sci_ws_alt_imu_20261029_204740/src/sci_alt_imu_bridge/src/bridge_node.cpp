// sci_alt_imu_bridge — Go2 /lowstate (unitree_go/LowState.imu_state + tick) -> sensor_msgs/Imu on a dedicated topic.
// Never touches /utlidar/*, never publishes to Go2 command topics, never rotates/filters/de-biases the IMU values.
#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sci_alt_imu_bridge/core.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/string.hpp"
#include "unitree_go/msg/low_state.hpp"

using namespace sci_alt_imu;
using std::chrono::steady_clock;

static int64_t steady_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

static double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, (size_t)(q * (v.size() - 1) + 0.5))];
}

class BridgeNode : public rclcpp::Node {
 public:
  BridgeNode() : Node("sci_alt_imu_bridge"), sys_clock_(RCL_SYSTEM_TIME) {
    in_topic_ = declare_parameter<std::string>("input_topic", "/lowstate");
    out_topic_ = declare_parameter<std::string>("output_topic", "/sci_alt_imu/imu");
    health_topic_ = declare_parameter<std::string>("health_topic", "/sci_alt_imu/health");
    frame_id_ = declare_parameter<std::string>("frame_id", "go2_body_imu");
    const bool in_reliable = declare_parameter<bool>("input_reliable", false);
    const int in_depth = declare_parameter<int>("input_depth", 50);
    const int out_depth = declare_parameter<int>("output_depth", 200);
    health_jsonl_ = declare_parameter<std::string>("health_jsonl", "");
    timing_csv_ = declare_parameter<std::string>("timing_csv", "");
    timing_every_ = declare_parameter<int>("timing_every", 10);
    max_log_bytes_ = declare_parameter<int64_t>("max_log_bytes", 50LL * 1024 * 1024);
    Params p;
    p.gap_ms = declare_parameter<double>("gap_ms", p.gap_ms);
    p.clock_step_ms = declare_parameter<double>("clock_step_ms", p.clock_step_ms);
    p.recover_s = declare_parameter<double>("recover_s", p.recover_s);
    bridge_ = std::make_unique<Bridge>(p);

    rclcpp::QoS qin{rclcpp::KeepLast(in_depth)};
    if (in_reliable) qin.reliable(); else qin.best_effort();
    rclcpp::QoS qout{rclcpp::KeepLast(out_depth)};
    qout.reliable();
    pub_ = create_publisher<sensor_msgs::msg::Imu>(out_topic_, qout);
    pub_health_ = create_publisher<std_msgs::msg::String>(health_topic_, rclcpp::QoS(10).reliable());
    sub_ = create_subscription<unitree_go::msg::LowState>(
        in_topic_, qin, [this](unitree_go::msg::LowState::ConstSharedPtr m) { on_lowstate(*m); });
    watchdog_ = create_wall_timer(std::chrono::milliseconds(20), [this]() { bridge_->tick_watchdog(steady_ns()); });
    health_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() { publish_health(); });
    if (!timing_csv_.empty()) {
      timing_.open(timing_csv_, std::ios::out | std::ios::trunc);
      timing_ << "tick,tick_unwrapped,rx_sys_ns,rx_steady_ns,verdict,stamp_ns,residual_ms\n";
    }
    if (!health_jsonl_.empty()) health_.open(health_jsonl_, std::ios::out | std::ios::app);
    RCLCPP_INFO(get_logger(), "in=%s (%s, depth %d) out=%s frame=%s", in_topic_.c_str(), in_reliable ? "reliable" : "best_effort",
                in_depth, out_topic_.c_str(), frame_id_.c_str());
  }

 private:
  void on_lowstate(const unitree_go::msg::LowState &m) {
    Sample s;
    s.rx_sys_ns = sys_clock_.now().nanoseconds();
    s.rx_steady_ns = steady_ns();
    s.tick = m.tick;
    for (int i = 0; i < 3; ++i) {
      s.gyro[i] = m.imu_state.gyroscope[i];
      s.acc[i] = m.imu_state.accelerometer[i];
    }
    for (int i = 0; i < 4; ++i) s.quat[i] = m.imu_state.quaternion[i];
    int64_t stamp = 0;
    const Verdict v = bridge_->process(s, &stamp);
    if (v == Verdict::PUBLISH) {
      sensor_msgs::msg::Imu out;
      out.header.stamp.sec = (int32_t)(stamp / 1000000000LL);
      out.header.stamp.nanosec = (uint32_t)(stamp % 1000000000LL);
      out.header.frame_id = frame_id_;
      out.orientation_covariance[0] = -1.0;    // orientation not provided (FAST-LIO uses only gyro/acc/stamp)
      out.angular_velocity.x = s.gyro[0]; out.angular_velocity.y = s.gyro[1]; out.angular_velocity.z = s.gyro[2];
      out.linear_acceleration.x = s.acc[0]; out.linear_acceleration.y = s.acc[1]; out.linear_acceleration.z = s.acc[2];
      // covariances left 0 = unknown (no device spec / calibrated noise model yet)
      pub_->publish(out);
      win_residual_.push_back(bridge_->last_residual_ms());
      win_gnorm_.push_back(std::sqrt((double)s.acc[0] * s.acc[0] + (double)s.acc[1] * s.acc[1] + (double)s.acc[2] * s.acc[2]));
      // 2026-10-30 body-shake metrics for the E6 MANUAL_SETTLE check (1 s window, raw gyro incl. bias ~0.15 deg/s)
      const double w2 = (double)s.gyro[0] * s.gyro[0] + (double)s.gyro[1] * s.gyro[1] + (double)s.gyro[2] * s.gyro[2];
      win_gyro_sq_ += w2;
      win_gyro_peak_ = std::max(win_gyro_peak_, std::sqrt(w2));
      win_pub_++;
    }
    if (timing_.is_open() && (++nsamples_ % timing_every_ == 0 || v != Verdict::PUBLISH) && bytes_ok(timing_csv_)) {
      timing_ << m.tick << ',' << bridge_->unwrapped_tick() << ',' << s.rx_sys_ns << ',' << s.rx_steady_ns << ',' << to_string(v) << ','
              << (v == Verdict::PUBLISH ? stamp : 0) << ',' << (v == Verdict::PUBLISH ? bridge_->last_residual_ms() : 0.0) << '\n';
    }
  }

  bool bytes_ok(const std::string &path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    return !f.good() || (int64_t)f.tellg() < max_log_bytes_;
  }

  void publish_health() {
    const Counters &c = bridge_->counters();
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    const double cpu_s = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
    const int64_t now = steady_ns();
    const double cpu_pct = last_cpu_t_ns_ ? 100.0 * (cpu_s - last_cpu_s_) / ((now - last_cpu_t_ns_) * 1e-9) : 0.0;
    last_cpu_s_ = cpu_s;
    last_cpu_t_ns_ = now;
    double gn = 0;
    for (double g : win_gnorm_) gn += g;
    gn = win_gnorm_.empty() ? 0 : gn / win_gnorm_.size();
    double gvar = 0;
    for (double g : win_gnorm_) gvar += (g - gn) * (g - gn);
    const double acc_std = win_gnorm_.size() > 1 ? std::sqrt(gvar / (win_gnorm_.size() - 1)) : 0.0;
    const double rad2deg = 180.0 / M_PI;
    const double gyro_rms = win_pub_ > 0 ? std::sqrt(win_gyro_sq_ / win_pub_) * rad2deg : 0.0;
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(3);
    o << "{\"wall_ns\":" << sys_clock_.now().nanoseconds() << ",\"health\":\"" << to_string(bridge_->health()) << "\",\"reason\":\""
      << bridge_->reason() << "\",\"publish_hz\":" << win_pub_ << ",\"residual_ms_p50\":" << pct(win_residual_, 0.5)
      << ",\"residual_ms_p95\":" << pct(win_residual_, 0.95) << ",\"residual_ms_max\":" << pct(win_residual_, 1.0)
      << ",\"gravity_norm_mean\":" << gn << ",\"gyro_rms_dps\":" << gyro_rms << ",\"gyro_peak_dps\":" << win_gyro_peak_ * rad2deg
      << ",\"acc_norm_std\":" << acc_std << ",\"clock_ready\":" << (bridge_->clock().ready() ? "true" : "false")
      << ",\"drift_ppm\":" << bridge_->clock().drift_ppm() << ",\"received\":" << c.received << ",\"published\":" << c.published
      << ",\"repeat\":" << c.repeat << ",\"same_tick_diff\":" << c.same_tick_diff << ",\"invalid\":" << c.invalid
      << ",\"not_monotonic\":" << c.not_monotonic << ",\"blocked_clock\":" << c.blocked_clock << ",\"gaps\":" << c.gaps
      << ",\"tick_jumps\":" << c.tick_jumps << ",\"tick_resets\":" << c.tick_resets << ",\"clock_steps\":" << c.clock_steps
      << ",\"model_resets\":" << c.model_resets << ",\"neg_residual\":" << c.neg_residual << ",\"frozen_events\":" << c.frozen_events
      << ",\"cpu_pct\":" << cpu_pct << ",\"maxrss_kb\":" << ru.ru_maxrss << ",\"frame_in\":\"go2 LowState.imu_state (raw)\",\"frame_out\":\""
      << frame_id_ << "\"}";
    std_msgs::msg::String msg;
    msg.data = o.str();
    pub_health_->publish(msg);
    if (health_.is_open() && bytes_ok(health_jsonl_)) health_ << msg.data << '\n' << std::flush;
    win_residual_.clear();
    win_gnorm_.clear();
    win_gyro_sq_ = 0;
    win_gyro_peak_ = 0;
    win_pub_ = 0;
  }

  rclcpp::Clock sys_clock_;
  std::string in_topic_, out_topic_, health_topic_, frame_id_, health_jsonl_, timing_csv_;
  int timing_every_ = 10;
  int64_t max_log_bytes_ = 0;
  std::unique_ptr<Bridge> bridge_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_health_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr watchdog_, health_timer_;
  std::ofstream timing_, health_;
  uint64_t nsamples_ = 0;
  std::vector<double> win_residual_, win_gnorm_;
  double win_gyro_sq_ = 0, win_gyro_peak_ = 0;
  int win_pub_ = 0;
  double last_cpu_s_ = 0;
  int64_t last_cpu_t_ns_ = 0;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BridgeNode>());
  rclcpp::shutdown();
  return 0;
}
