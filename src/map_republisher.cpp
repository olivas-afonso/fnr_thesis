#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/utils.h"

class MapRepublisher : public rclcpp::Node {
public:
    MapRepublisher() : Node("map_republisher") {
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/persistent_map", 10,
            [this](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
                if (!map_received_) {
                    original_map_ = *msg;
                    map_received_ = true;
                    timer_ = create_wall_timer(
                        std::chrono::milliseconds(100),
                        [this]() { republishMap(); });
                }
            });

        map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
            "/map", rclcpp::QoS(1).transient_local());
    }
    bool map_published_ = false;

private:
    void republishMap() {
        try {
            // Get the latest transform
            auto transform = tf_buffer_->lookupTransform(
                "map", "transformed_map",
                tf2::TimePointZero);

            // Create transformed map
            auto message = original_map_;
            
            // Apply scale to resolution
            message.info.resolution *= transform.transform.translation.z;
            
            // Apply translation to origin
            message.info.origin.position.x += transform.transform.translation.x;
            message.info.origin.position.y += transform.transform.translation.y;
            
            // Apply rotation to origin
            tf2::Quaternion q;
            tf2::fromMsg(transform.transform.rotation, q);
            double roll, pitch, yaw;
            tf2::getEulerYPR(q, yaw, pitch, roll);
            
            tf2::Quaternion origin_q;
            tf2::fromMsg(message.info.origin.orientation, origin_q);
            origin_q *= q;
            message.info.origin.orientation = tf2::toMsg(origin_q);

            message.header.stamp = now();
            message.header.frame_id = "map"; // Keep original frame
            
            if (!map_published_) {
                map_pub_->publish(message);
                map_published_ = true;
            }
            
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(get_logger(), "Transform error: %s", ex.what());
        }
    }

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    nav_msgs::msg::OccupancyGrid original_map_;
    bool map_received_ = false;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MapRepublisher>());
    rclcpp::shutdown();
    return 0;
}