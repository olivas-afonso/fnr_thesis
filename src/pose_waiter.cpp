#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "std_srvs/srv/empty.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.h>

using namespace std::chrono_literals;

class PoseWaiter : public rclcpp::Node {
public:
    PoseWaiter() : Node("pose_waiter") {
        // Service to trigger system readiness
        ready_service_ = this->create_service<std_srvs::srv::Empty>(
            "/system_ready",
            [this](const std::shared_ptr<std_srvs::srv::Empty::Request>,
                  std::shared_ptr<std_srvs::srv::Empty::Response>) {
                RCLCPP_INFO(this->get_logger(), "System ready to receive initial pose");
            });

        // Subscribe to initial pose from RViz
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", 10,
            [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
                if (!initial_pose_received_) {
                    initial_pose_received_ = true;
                    
                    // Reset AMCL first
                    auto global_loc_client = this->create_client<std_srvs::srv::Empty>("/global_localization");
                    if (!global_loc_client->wait_for_service(1s)) {
                        RCLCPP_WARN(this->get_logger(), "Service /global_localization not available");
                        return;
                    }
                    global_loc_client->async_send_request(std::make_shared<std_srvs::srv::Empty::Request>());

                    // Forward pose to AMCL after small delay
                    auto timer = this->create_wall_timer(
                        500ms,  // Wait for AMCL reset to complete
                        [this, msg]() {
                            auto pub = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
                                "/initialpose", 10);
                            pub->publish(*msg);
                            RCLCPP_INFO(this->get_logger(), 
                                "Initial pose set to: (%.2f, %.2f, %.2f)",
                                msg->pose.pose.position.x,
                                msg->pose.pose.position.y,
                                tf2::getYaw(msg->pose.pose.orientation));
                            this->timer_.reset();  // Cancel timer after one execution
                        }
                    );
                    timer_ = timer;
                }
            });
    }

private:
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr ready_service_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    bool initial_pose_received_ = false;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PoseWaiter>());
    rclcpp::shutdown();
    return 0;
}