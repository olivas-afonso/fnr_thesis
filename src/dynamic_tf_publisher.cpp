#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/odometry.hpp"

class DynamicTfPublisher : public rclcpp::Node {
public:
    DynamicTfPublisher() : Node("dynamic_tf_publisher") {
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        
        // Subscribe to odometry to get proper timestamps
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/zed/zed_node/odom", 10,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                last_odom_time_ = msg->header.stamp;
                current_odom_pose_ = msg->pose.pose;
                odom_received_ = true;
            });

        // Only publish when we get new odometry data
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),  // Higher rate for responsiveness
            std::bind(&DynamicTfPublisher::publish_transforms, this));
    }

private:
    void publish_transforms() {
        if (!odom_received_) return;

        // Use the EXACT timestamp from odometry
        auto stamp = last_odom_time_;

        geometry_msgs::msg::TransformStamped odom_to_camera;
        odom_to_camera.header.stamp = stamp;  // Critical: Use odometry time
        odom_to_camera.header.frame_id = "odom";
        odom_to_camera.child_frame_id = "zed_camera_link";
        odom_to_camera.transform.translation.x = current_odom_pose_.position.x;
        odom_to_camera.transform.translation.y = current_odom_pose_.position.y;
        odom_to_camera.transform.translation.z = current_odom_pose_.position.z;
        odom_to_camera.transform.rotation = current_odom_pose_.orientation;

        tf_broadcaster_->sendTransform(odom_to_camera);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Time last_odom_time_;
    geometry_msgs::msg::Pose current_odom_pose_;
    bool odom_received_ = false;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DynamicTfPublisher>());
    rclcpp::shutdown();
    return 0;
}