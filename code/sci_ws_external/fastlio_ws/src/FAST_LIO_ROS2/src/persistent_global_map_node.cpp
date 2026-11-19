#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <pcl/common/point_tests.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

namespace
{

struct VoxelKey
{
    std::int32_t x;
    std::int32_t y;
    std::int32_t z;

    bool operator==(const VoxelKey & other) const noexcept
    {
        return x == other.x &&
               y == other.y &&
               z == other.z;
    }
};

struct VoxelKeyHash
{
    std::size_t operator()(const VoxelKey & key) const noexcept
    {
        std::size_t seed = 0U;

        const auto combine = [&seed](std::int32_t value)
        {
            const std::size_t hash_value =
                std::hash<std::int32_t>{}(value);

            seed ^= hash_value +
                    0x9e3779b97f4a7c15ULL +
                    (seed << 6U) +
                    (seed >> 2U);
        };

        combine(key.x);
        combine(key.y);
        combine(key.z);

        return seed;
    }
};

struct VoxelValue
{
    double x_sum{0.0};
    double y_sum{0.0};
    double z_sum{0.0};
    double intensity_sum{0.0};
    std::uint32_t sample_count{0U};

    void add(
        const pcl::PointXYZI & point,
        std::uint32_t max_samples)
    {
        if (sample_count < max_samples)
        {
            x_sum += static_cast<double>(point.x);
            y_sum += static_cast<double>(point.y);
            z_sum += static_cast<double>(point.z);
            intensity_sum +=
                static_cast<double>(point.intensity);

            ++sample_count;
            return;
        }

        // Continue adapting the voxel representative without allowing
        // sums and counters to grow indefinitely during long operation.
        constexpr double old_weight = 0.95;
        constexpr double new_weight = 0.05;

        const double divisor =
            static_cast<double>(
                sample_count == 0U ? 1U : sample_count);

        const double mean_x = x_sum / divisor;
        const double mean_y = y_sum / divisor;
        const double mean_z = z_sum / divisor;
        const double mean_intensity =
            intensity_sum / divisor;

        x_sum =
            (old_weight * mean_x +
             new_weight * static_cast<double>(point.x)) *
            divisor;

        y_sum =
            (old_weight * mean_y +
             new_weight * static_cast<double>(point.y)) *
            divisor;

        z_sum =
            (old_weight * mean_z +
             new_weight * static_cast<double>(point.z)) *
            divisor;

        intensity_sum =
            (old_weight * mean_intensity +
             new_weight *
                 static_cast<double>(point.intensity)) *
            divisor;
    }

    pcl::PointXYZI mean() const
    {
        pcl::PointXYZI point;

        const double divisor =
            static_cast<double>(
                sample_count == 0U ? 1U : sample_count);

        point.x =
            static_cast<float>(x_sum / divisor);
        point.y =
            static_cast<float>(y_sum / divisor);
        point.z =
            static_cast<float>(z_sum / divisor);
        point.intensity =
            static_cast<float>(intensity_sum / divisor);

        return point;
    }
};

}  // namespace

class PersistentGlobalMapNode final : public rclcpp::Node
{
public:
    using PointCloud =
        pcl::PointCloud<pcl::PointXYZI>;

    PersistentGlobalMapNode()
    : Node("persistent_global_map")
    {
        input_topic_ =
            declare_parameter<std::string>(
                "input_topic",
                "/cloud_registered");

        output_topic_ =
            declare_parameter<std::string>(
                "output_topic",
                "/Laser_map_persistent");

        fixed_frame_ =
            declare_parameter<std::string>(
                "fixed_frame",
                "camera_init");

        save_path_ =
            declare_parameter<std::string>(
                "save_path",
                "/home/unitree/go2_ws/PCD/"
                "fastlio_persistent_map.pcd");

        voxel_size_ =
            declare_parameter<double>(
                "voxel_size",
                0.12);

        publish_hz_ =
            declare_parameter<double>(
                "publish_hz",
                0.1);

        update_stride_ =
            declare_parameter<int>(
                "update_stride",
                1);

        max_voxels_ =
            declare_parameter<int>(
                "max_voxels",
                2000000);

        max_samples_per_voxel_ =
            declare_parameter<int>(
                "max_samples_per_voxel",
                100);

        log_interval_sec_ =
            declare_parameter<double>(
                "log_interval_sec",
                10.0);

        validate_parameters();

        map_publisher_ =
            create_publisher<
                sensor_msgs::msg::PointCloud2>(
                output_topic_,
                rclcpp::QoS(1)
                    .reliable()
                    .transient_local());

        cloud_subscription_ =
            create_subscription<
                sensor_msgs::msg::PointCloud2>(
                input_topic_,
                rclcpp::SensorDataQoS(),
                std::bind(
                    &PersistentGlobalMapNode::
                        cloud_callback,
                    this,
                    std::placeholders::_1));

        const auto publish_period =
            std::chrono::duration<double>(
                1.0 / publish_hz_);

        publish_timer_ =
            create_wall_timer(
                std::chrono::duration_cast<
                    std::chrono::milliseconds>(
                    publish_period),
                std::bind(
                    &PersistentGlobalMapNode::
                        publish_map,
                    this));

        save_service_ =
            create_service<
                std_srvs::srv::Trigger>(
                "~/save",
                std::bind(
                    &PersistentGlobalMapNode::
                        save_callback,
                    this,
                    std::placeholders::_1,
                    std::placeholders::_2));

        clear_service_ =
            create_service<
                std_srvs::srv::Trigger>(
                "~/clear",
                std::bind(
                    &PersistentGlobalMapNode::
                        clear_callback,
                    this,
                    std::placeholders::_1,
                    std::placeholders::_2));

        last_log_time_ = now();

        RCLCPP_INFO(
            get_logger(),
            "Persistent map started: "
            "input=%s output=%s frame=%s "
            "voxel=%.3f m max_voxels=%d "
            "update_stride=%d publish_hz=%.3f",
            input_topic_.c_str(),
            output_topic_.c_str(),
            fixed_frame_.c_str(),
            voxel_size_,
            max_voxels_,
            update_stride_,
            publish_hz_);
    }

private:
    void validate_parameters()
    {
        if (voxel_size_ <= 0.0)
        {
            throw std::runtime_error(
                "voxel_size must be greater than zero");
        }

        if (publish_hz_ <= 0.0)
        {
            throw std::runtime_error(
                "publish_hz must be greater than zero");
        }

        if (update_stride_ < 1)
        {
            RCLCPP_WARN(
                get_logger(),
                "update_stride was less than 1. "
                "Using 1 instead.");

            update_stride_ = 1;
        }

        if (max_voxels_ < 1)
        {
            throw std::runtime_error(
                "max_voxels must be greater than zero");
        }

        if (max_samples_per_voxel_ < 1)
        {
            RCLCPP_WARN(
                get_logger(),
                "max_samples_per_voxel was less "
                "than 1. Using 1 instead.");

            max_samples_per_voxel_ = 1;
        }

        if (log_interval_sec_ <= 0.0)
        {
            log_interval_sec_ = 10.0;
        }
    }

    VoxelKey make_voxel_key(
        const pcl::PointXYZI & point) const
    {
        return VoxelKey{
            static_cast<std::int32_t>(
                std::floor(
                    static_cast<double>(point.x) /
                    voxel_size_)),
            static_cast<std::int32_t>(
                std::floor(
                    static_cast<double>(point.y) /
                    voxel_size_)),
            static_cast<std::int32_t>(
                std::floor(
                    static_cast<double>(point.z) /
                    voxel_size_))
        };
    }

    void cloud_callback(
        const sensor_msgs::msg::PointCloud2::
            SharedPtr message)
    {
        ++received_scan_count_;

        const std::uint64_t stride =
            static_cast<std::uint64_t>(
                update_stride_);

        if (((received_scan_count_ - 1U) %
             stride) != 0U)
        {
            return;
        }

        if (!message->header.frame_id.empty() &&
            message->header.frame_id != fixed_frame_)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                5000,
                "Input cloud frame is '%s', "
                "but fixed_frame is '%s'. "
                "The input cloud must already be "
                "registered in one fixed frame.",
                message->header.frame_id.c_str(),
                fixed_frame_.c_str());
        }

        PointCloud input_cloud;
        pcl::fromROSMsg(
            *message,
            input_cloud);

        std::size_t inserted_count = 0U;
        std::size_t updated_count = 0U;
        std::size_t rejected_count = 0U;

        {
            std::lock_guard<std::mutex>
                lock(map_mutex_);

            for (const auto & point :
                 input_cloud.points)
            {
                if (!pcl::isFinite(point))
                {
                    continue;
                }

                const VoxelKey key =
                    make_voxel_key(point);

                auto iterator =
                    voxels_.find(key);

                if (iterator != voxels_.end())
                {
                    iterator->second.add(
                        point,
                        static_cast<std::uint32_t>(
                            max_samples_per_voxel_));

                    ++updated_count;
                    continue;
                }

                if (
                    static_cast<int>(
                        voxels_.size()) >=
                    max_voxels_)
                {
                    // Preserve all existing voxels.
                    // Reject only newly observed voxels
                    // after reaching the configured limit.
                    ++rejected_count;
                    continue;
                }

                VoxelValue value;
                value.add(
                    point,
                    static_cast<std::uint32_t>(
                        max_samples_per_voxel_));

                voxels_.emplace(
                    key,
                    value);

                ++inserted_count;
            }

            last_message_stamp_ =
                message->header.stamp;
        }

        inserted_points_ +=
            static_cast<std::uint64_t>(
                inserted_count);

        updated_points_ +=
            static_cast<std::uint64_t>(
                updated_count);

        rejected_points_ +=
            static_cast<std::uint64_t>(
                rejected_count);

        print_status_if_needed();
    }

    void print_status_if_needed()
    {
        const rclcpp::Time current_time =
            now();

        if (
            (current_time - last_log_time_)
                .seconds() <
            log_interval_sec_)
        {
            return;
        }

        std::size_t voxel_count = 0U;

        {
            std::lock_guard<std::mutex>
                lock(map_mutex_);

            voxel_count =
                voxels_.size();
        }

        RCLCPP_INFO(
            get_logger(),
            "Persistent map status: "
            "voxels=%zu inserted=%llu "
            "updated=%llu rejected_at_limit=%llu "
            "received_scans=%llu",
            voxel_count,
            static_cast<unsigned long long>(
                inserted_points_),
            static_cast<unsigned long long>(
                updated_points_),
            static_cast<unsigned long long>(
                rejected_points_),
            static_cast<unsigned long long>(
                received_scan_count_));

        last_log_time_ =
            current_time;
    }

    PointCloud::Ptr create_snapshot() const
    {
        PointCloud::Ptr snapshot(
            new PointCloud());

        std::lock_guard<std::mutex>
            lock(map_mutex_);

        snapshot->points.reserve(
            voxels_.size());

        for (const auto & entry :
             voxels_)
        {
            snapshot->points.push_back(
                entry.second.mean());
        }

        snapshot->width =
            static_cast<std::uint32_t>(
                snapshot->points.size());

        snapshot->height = 1U;
        snapshot->is_dense = false;

        return snapshot;
    }

    void publish_map()
    {
        const PointCloud::Ptr snapshot =
            create_snapshot();

        if (snapshot->empty())
        {
            return;
        }

        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(
            *snapshot,
            output);

        {
            std::lock_guard<std::mutex>
                lock(map_mutex_);

            output.header.stamp =
                last_message_stamp_;
        }

        output.header.frame_id =
            fixed_frame_;

        map_publisher_->publish(
            output);
    }

    void save_callback(
        const std::shared_ptr<
            std_srvs::srv::Trigger::Request>,
        std::shared_ptr<
            std_srvs::srv::Trigger::Response>
            response)
    {
        const PointCloud::Ptr snapshot =
            create_snapshot();

        if (snapshot->empty())
        {
            response->success = false;
            response->message =
                "Persistent map is empty.";
            return;
        }

        const int save_result =
            pcl::io::savePCDFileBinary(
                save_path_,
                *snapshot);

        if (save_result != 0)
        {
            response->success = false;
            response->message =
                "Failed to save PCD: " +
                save_path_;

            RCLCPP_ERROR(
                get_logger(),
                "%s",
                response->message.c_str());

            return;
        }

        response->success = true;
        response->message =
            "Saved " +
            std::to_string(
                snapshot->size()) +
            " voxels to " +
            save_path_;

        RCLCPP_INFO(
            get_logger(),
            "%s",
            response->message.c_str());
    }

    void clear_callback(
        const std::shared_ptr<
            std_srvs::srv::Trigger::Request>,
        std::shared_ptr<
            std_srvs::srv::Trigger::Response>
            response)
    {
        {
            std::lock_guard<std::mutex>
                lock(map_mutex_);

            voxels_.clear();
            last_message_stamp_ =
                builtin_interfaces::msg::Time();
        }

        inserted_points_ = 0U;
        updated_points_ = 0U;
        rejected_points_ = 0U;
        received_scan_count_ = 0U;

        response->success = true;
        response->message =
            "Persistent map cleared.";

        RCLCPP_WARN(
            get_logger(),
            "Persistent map was cleared "
            "by service request.");
    }

    std::string input_topic_;
    std::string output_topic_;
    std::string fixed_frame_;
    std::string save_path_;

    double voxel_size_{0.12};
    double publish_hz_{0.1};
    double log_interval_sec_{10.0};

    int update_stride_{1};
    int max_voxels_{2000000};
    int max_samples_per_voxel_{100};

    mutable std::mutex map_mutex_;

    std::unordered_map<
        VoxelKey,
        VoxelValue,
        VoxelKeyHash>
        voxels_;

    builtin_interfaces::msg::Time
        last_message_stamp_;

    std::uint64_t
        received_scan_count_{0U};

    std::uint64_t
        inserted_points_{0U};

    std::uint64_t
        updated_points_{0U};

    std::uint64_t
        rejected_points_{0U};

    rclcpp::Time last_log_time_;

    rclcpp::Subscription<
        sensor_msgs::msg::PointCloud2>::
        SharedPtr cloud_subscription_;

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2>::
        SharedPtr map_publisher_;

    rclcpp::TimerBase::SharedPtr
        publish_timer_;

    rclcpp::Service<
        std_srvs::srv::Trigger>::
        SharedPtr save_service_;

    rclcpp::Service<
        std_srvs::srv::Trigger>::
        SharedPtr clear_service_;
};

int main(
    int argc,
    char ** argv)
{
    rclcpp::init(
        argc,
        argv);

    int exit_code = 0;

    try
    {
        const auto node =
            std::make_shared<
                PersistentGlobalMapNode>();

        rclcpp::spin(node);
    }
    catch (const std::exception & exception)
    {
        std::cerr
            << "Persistent global map node failed: "
            << exception.what()
            << std::endl;

        exit_code = 1;
    }

    rclcpp::shutdown();

    return exit_code;
}
