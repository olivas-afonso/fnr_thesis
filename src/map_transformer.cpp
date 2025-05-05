#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"  // Required for tf2::toMsg

class MapTransformer : public rclcpp::Node {
public:
    MapTransformer() : Node("map_transformer") {
        // Parameters
        this->declare_parameter<double>("scale", 1.0);
        this->declare_parameter<double>("rotation_deg", 0.0);
        this->declare_parameter<double>("translation_x", 0.0);
        this->declare_parameter<double>("translation_y", 0.0);

        // TF Broadcaster
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // Timer (10Hz for smoother updates)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&MapTransformer::broadcast_transform, this));
    }

private:
    void broadcast_transform() {
        auto t = geometry_msgs::msg::TransformStamped();
        
        // Get parameters
        double scale = this->get_parameter("scale").as_double();
        double rotation = this->get_parameter("rotation_deg").as_double() * M_PI / 180.0;
        double tx = this->get_parameter("translation_x").as_double();
        double ty = this->get_parameter("translation_y").as_double();

        // Set transform (map -> transformed_map)
        t.header.stamp = this->now();
        t.header.frame_id = "map";
        t.child_frame_id = "transformed_map";
        
        t.transform.translation.x = tx;
        t.transform.translation.y = ty;
        t.transform.translation.z = scale;  // Store scale in z for reference
        
        tf2::Quaternion q;
        q.setRPY(0, 0, rotation);
        t.transform.rotation = tf2::toMsg(q);  // Now properly included

        // Broadcast transform
        tf_broadcaster_->sendTransform(t);
        
        RCLCPP_DEBUG(this->get_logger(), 
            "Broadcasting transform: scale=%.2f, rotation=%.2f°, x=%.2f, y=%.2f",
            scale, rotation * 180.0 / M_PI, tx, ty);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MapTransformer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}