// l1_adapter_node — republishes the body-IMU FAST-LIO outputs in the frame contract of the L1-IMU FAST-LIO, so the UNCHANGED
// LIVE2 consumers (lio_monitor2 -> e2_feed -> E2..E8, run with the UNCHANGED L1 yaml) can be fed by the body-IMU path.
//   in : /sci_alt_imu/Odometry, /sci_alt_imu/cloud_registered   (alt FAST-LIO, world_b = body-IMU frame at start)
//   out: /Odometry, /cloud_registered                            (world_1 = L1-IMU frame at start, pose of the L1-IMU)
// Same math as e2_compat/alt_to_l1_adapter.py (T_1b from BOTH FAST-LIO extrinsics, computed by the launcher with that module and
// passed as parameters R_1b / t_1b; nothing tuned here):
//   pose : R_w1 = R_1b R_wb R_1b^T ,  p_w1 = R_1b (p_wb - R_wb R_1b^T t_1b) + t_1b
//   cloud: p_w1 = R_1b p_wb + t_1b      (normal_x/y/z, if present, are rotated by R_1b; all other fields copied byte for byte)
// Header stamps and frame ids are copied unchanged (e2_feed pairs cloud and odometry on EQUAL stamps).
// Gate: nothing is published unless the IMU bridge health (/sci_alt_imu/health JSON) is READY and younger than 2.5 s, so a dead
// body-IMU input looks to lio2 exactly like a stalled FAST-LIO (no /Odometry) and E2 gets no new samples.
// Never publishes anything to the robot. Status JSON (counts, gate, latency, T_1b) written every second (atomic rename).
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/string.hpp"

using M3 = std::array<double, 9>;

static void mul(const M3 &a, const M3 &b, M3 &o) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
}
static M3 tr(const M3 &a) { return {a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]}; }
static void mv(const M3 &a, const double *v, double *o) {
  for (int i = 0; i < 3; ++i) o[i] = a[i * 3] * v[0] + a[i * 3 + 1] * v[1] + a[i * 3 + 2] * v[2];
}
static M3 q2R(double x, double y, double z, double w) {
  return {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
          2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
          2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)};
}
static void R2q(const M3 &R, double &x, double &y, double &z, double &w) {   // Shepperd
  const double t = R[0] + R[4] + R[8];
  if (t > 0) { double s = std::sqrt(t + 1.0) * 2; w = 0.25 * s; x = (R[7] - R[5]) / s; y = (R[2] - R[6]) / s; z = (R[3] - R[1]) / s; }
  else if (R[0] > R[4] && R[0] > R[8]) { double s = std::sqrt(1.0 + R[0] - R[4] - R[8]) * 2; w = (R[7] - R[5]) / s; x = 0.25 * s; y = (R[1] + R[3]) / s; z = (R[2] + R[6]) / s; }
  else if (R[4] > R[8]) { double s = std::sqrt(1.0 + R[4] - R[0] - R[8]) * 2; w = (R[2] - R[6]) / s; x = (R[1] + R[3]) / s; y = 0.25 * s; z = (R[5] + R[7]) / s; }
  else { double s = std::sqrt(1.0 + R[8] - R[0] - R[4]) * 2; w = (R[3] - R[1]) / s; x = (R[2] + R[6]) / s; y = (R[5] + R[7]) / s; z = 0.25 * s; }
  const double n = std::sqrt(x * x + y * y + z * z + w * w); x /= n; y /= n; z /= n; w /= n;
}

class L1Adapter : public rclcpp::Node {
 public:
  L1Adapter() : Node("sci_alt_l1_adapter") {
    auto Rv = declare_parameter<std::vector<double>>("R_1b", std::vector<double>{});
    auto tv = declare_parameter<std::vector<double>>("t_1b", std::vector<double>{});
    tag_ = declare_parameter<std::string>("transform_tag", "");
    status_ = declare_parameter<std::string>("status_file", "");
    require_health_ = declare_parameter<bool>("require_health", true);
    health_max_age_ = declare_parameter<double>("health_max_age_s", 2.5);
    const auto in_odom = declare_parameter<std::string>("in_odom", "/sci_alt_imu/Odometry");
    const auto in_cloud = declare_parameter<std::string>("in_cloud", "/sci_alt_imu/cloud_registered");
    const auto out_odom = declare_parameter<std::string>("out_odom", "/Odometry");
    const auto out_cloud = declare_parameter<std::string>("out_cloud", "/cloud_registered");
    const auto health = declare_parameter<std::string>("health_topic", "/sci_alt_imu/health");
    if (Rv.size() != 9 || tv.size() != 3) throw std::runtime_error("R_1b (9) and t_1b (3) parameters are required");
    for (int i = 0; i < 9; ++i) R_[i] = Rv[i];
    for (int i = 0; i < 3; ++i) t_[i] = tv[i];
    M3 RtR; mul(tr(R_), R_, RtR);
    double orth = 0, det = R_[0] * (R_[4] * R_[8] - R_[5] * R_[7]) - R_[1] * (R_[3] * R_[8] - R_[5] * R_[6]) + R_[2] * (R_[3] * R_[7] - R_[4] * R_[6]);
    for (int i = 0; i < 9; ++i) orth = std::max(orth, std::fabs(RtR[i] - (i % 4 == 0 ? 1.0 : 0.0)));
    if (orth > 1e-6 || std::fabs(det - 1.0) > 1e-6) throw std::runtime_error("R_1b is not a rotation");
    Rt_ = tr(R_); mv(Rt_, t_.data(), Rtt_.data());
    // publishers: same depth/reliability FAST-LIO uses (reliable, 20) so the L1 consumers (BEST_EFFORT) match as before
    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>(out_odom, rclcpp::QoS(20).reliable());
    pub_cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>(out_cloud, rclcpp::QoS(20).reliable());
    // subscriptions BEST_EFFORT (like lio_monitor2): the adapter can never back-pressure the alt FAST-LIO
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(in_odom, rclcpp::QoS(20).best_effort(),
        [this](nav_msgs::msg::Odometry::UniquePtr m) { on_odom(std::move(m)); });
    sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(in_cloud, rclcpp::QoS(5).best_effort(),
        [this](sensor_msgs::msg::PointCloud2::UniquePtr m) { on_cloud(std::move(m)); });
    sub_health_ = create_subscription<std_msgs::msg::String>(health, rclcpp::QoS(10).reliable(),
        [this](std_msgs::msg::String::ConstSharedPtr m) {
          health_ready_ = m->data.find("\"health\":\"READY\"") != std::string::npos; health_rx_ = now_s();
        });
    timer_ = create_wall_timer(std::chrono::seconds(1), [this]() { write_status(); });
    RCLCPP_INFO(get_logger(), "L1 adapter %s: %s -> %s, %s -> %s, require_health=%d", tag_.c_str(), in_odom.c_str(), out_odom.c_str(),
                in_cloud.c_str(), out_cloud.c_str(), (int)require_health_);
  }

 private:
  static double now_s() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }
  bool gate() {
    gate_ok_ = !require_health_ || (health_ready_ && now_s() - health_rx_ <= health_max_age_);
    return gate_ok_;
  }
  static double stamp(const builtin_interfaces::msg::Time &s) { return s.sec + s.nanosec * 1e-9; }

  void on_odom(nav_msgs::msg::Odometry::UniquePtr m) {
    n_odom_in_++;
    if (!gate()) { n_odom_gated_++; return; }
    auto &P = m->pose.pose;
    const M3 Rwb = q2R(P.orientation.x, P.orientation.y, P.orientation.z, P.orientation.w);
    M3 a, Rw1; mul(R_, Rwb, a); mul(a, Rt_, Rw1);
    double v[3], u[3], pb[3] = {P.position.x, P.position.y, P.position.z};
    mv(Rwb, Rtt_.data(), u);                                   // R_wb R_1b^T t_1b
    for (int i = 0; i < 3; ++i) v[i] = pb[i] - u[i];
    double p1[3]; mv(R_, v, p1);
    P.position.x = p1[0] + t_[0]; P.position.y = p1[1] + t_[1]; P.position.z = p1[2] + t_[2];
    R2q(Rw1, P.orientation.x, P.orientation.y, P.orientation.z, P.orientation.w);
    rot_cov(m->pose.covariance); rot_cov(m->twist.covariance);
    rot_vec(m->twist.twist.linear); rot_vec(m->twist.twist.angular);
    last_odom_stamp_ = stamp(m->header.stamp);
    pub_odom_->publish(std::move(m)); n_odom_out_++;
  }

  template <class V> void rot_vec(V &v) {
    double a[3] = {v.x, v.y, v.z}, b[3]; mv(R_, a, b); v.x = b[0]; v.y = b[1]; v.z = b[2];
  }
  void rot_cov(std::array<double, 36> &C) {                    // blockdiag(R, R) C blockdiag(R, R)^T
    std::array<double, 36> T{}, O{};
    double B[36] = {0};
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { B[i * 6 + j] = R_[i * 3 + j]; B[(i + 3) * 6 + j + 3] = R_[i * 3 + j]; }
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) { double s = 0; for (int k = 0; k < 6; ++k) s += B[i * 6 + k] * C[k * 6 + j]; T[i * 6 + j] = s; }
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) { double s = 0; for (int k = 0; k < 6; ++k) s += T[i * 6 + k] * B[j * 6 + k]; O[i * 6 + j] = s; }
    C = O;
  }

  int off(const sensor_msgs::msg::PointCloud2 &m, const char *name) {
    for (const auto &f : m.fields)
      if (f.name == name && f.datatype == sensor_msgs::msg::PointField::FLOAT32 && f.count == 1) return (int)f.offset;
    return -1;
  }

  void on_cloud(sensor_msgs::msg::PointCloud2::UniquePtr m) {
    n_cloud_in_++;
    if (!gate()) { n_cloud_gated_++; return; }
    const auto t0 = std::chrono::steady_clock::now();
    const int ox = off(*m, "x"), oy = off(*m, "y"), oz = off(*m, "z");
    const int nx = off(*m, "normal_x"), ny = off(*m, "normal_y"), nz = off(*m, "normal_z");
    if (ox < 0 || oy < 0 || oz < 0 || m->is_bigendian) { n_cloud_bad_++; return; }
    const size_t n = (size_t)m->width * m->height, ps = m->point_step;
    if (m->data.size() < n * ps) { n_cloud_bad_++; return; }
    uint8_t *d = m->data.data();
    const bool norms = nx >= 0 && ny >= 0 && nz >= 0;
    for (size_t k = 0; k < n; ++k) {
      uint8_t *p = d + k * ps;
      float x, y, z; std::memcpy(&x, p + ox, 4); std::memcpy(&y, p + oy, 4); std::memcpy(&z, p + oz, 4);
      const double a[3] = {x, y, z}; double b[3]; mv(R_, a, b);
      x = (float)(b[0] + t_[0]); y = (float)(b[1] + t_[1]); z = (float)(b[2] + t_[2]);
      std::memcpy(p + ox, &x, 4); std::memcpy(p + oy, &y, 4); std::memcpy(p + oz, &z, 4);
      if (norms) {
        float u, v, w; std::memcpy(&u, p + nx, 4); std::memcpy(&v, p + ny, 4); std::memcpy(&w, p + nz, 4);
        const double c[3] = {u, v, w}; double e[3]; mv(R_, c, e);
        u = (float)e[0]; v = (float)e[1]; w = (float)e[2];
        std::memcpy(p + nx, &u, 4); std::memcpy(p + ny, &v, 4); std::memcpy(p + nz, &w, 4);
      }
    }
    last_cloud_stamp_ = stamp(m->header.stamp);
    pub_cloud_->publish(std::move(m)); n_cloud_out_++;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    cloud_ms_max_ = std::max(cloud_ms_max_, ms); cloud_ms_sum_ += ms;
  }

  void write_status() {
    if (status_.empty()) return;
    gate();
    std::ostringstream o;
    o.precision(10);
    o << "{\"kind\":\"ALT_L1_ADAPTER\",\"wall\":" << now_s() << ",\"tag\":\"" << tag_ << "\",\"gate_ok\":" << (gate_ok_ ? "true" : "false")
      << ",\"health_ready\":" << (health_ready_ ? "true" : "false") << ",\"health_age_s\":" << (health_rx_ > 0 ? now_s() - health_rx_ : -1)
      << ",\"odom_in\":" << n_odom_in_ << ",\"odom_out\":" << n_odom_out_ << ",\"odom_gated\":" << n_odom_gated_
      << ",\"cloud_in\":" << n_cloud_in_ << ",\"cloud_out\":" << n_cloud_out_ << ",\"cloud_gated\":" << n_cloud_gated_
      << ",\"cloud_bad\":" << n_cloud_bad_ << ",\"cloud_ms_max\":" << cloud_ms_max_
      << ",\"cloud_ms_mean\":" << (n_cloud_out_ ? cloud_ms_sum_ / n_cloud_out_ : 0.0)
      << ",\"last_odom_stamp\":" << last_odom_stamp_ << ",\"last_cloud_stamp\":" << last_cloud_stamp_ << ",\"R_1b\":[";
    for (int i = 0; i < 9; ++i) o << (i ? "," : "") << R_[i];
    o << "],\"t_1b\":[" << t_[0] << "," << t_[1] << "," << t_[2] << "]}";
    const std::string tmp = status_ + ".tmp";
    { std::ofstream f(tmp); f << o.str() << "\n"; }
    std::rename(tmp.c_str(), status_.c_str());
  }

  M3 R_{}, Rt_{};
  std::array<double, 3> t_{}, Rtt_{};
  std::string tag_, status_;
  bool require_health_ = true, health_ready_ = false, gate_ok_ = false;
  double health_max_age_ = 2.5, health_rx_ = 0, last_odom_stamp_ = 0, last_cloud_stamp_ = 0, cloud_ms_max_ = 0, cloud_ms_sum_ = 0;
  long n_odom_in_ = 0, n_odom_out_ = 0, n_odom_gated_ = 0, n_cloud_in_ = 0, n_cloud_out_ = 0, n_cloud_gated_ = 0, n_cloud_bad_ = 0;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_health_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<L1Adapter>());
  rclcpp::shutdown();
  return 0;
}
