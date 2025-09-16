#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include <fstream>
#include <iomanip>
#include <string>

class AMCLPathRecorder : public rclcpp::Node
{
public:
    AMCLPathRecorder() : Node("amcl_path_recorder")
    {
        // Declare single parameter for output file path
        this->declare_parameter("output_file", "amcl_path.csv");
        this->get_parameter("output_file", output_file_);
        
        // Create subscription
        amcl_subscription_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/amcl_pose", 10,
            std::bind(&AMCLPathRecorder::amclPoseCallback, this, std::placeholders::_1));
        
        // Initialize output file
        initializeOutputFile();
        
        RCLCPP_INFO(this->get_logger(), "AMCL Path Recorder started");
        RCLCPP_INFO(this->get_logger(), "Saving to: %s", output_file_.c_str());
    }

private:
    void initializeOutputFile()
    {
        std::ofstream file(output_file_);
        if (file.is_open()) {
            file << "timestamp,x,y,z,qx,qy,qz,qw\n";
            file.close();
            RCLCPP_INFO(this->get_logger(), "Initialized output file: %s", output_file_.c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize output file: %s", output_file_.c_str());
        }
    }

    void amclPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
    {
        // Check if pose has changed significantly
        bool pose_changed = false;
        if (last_saved_pose_.header.stamp.sec == 0) { // First pose
            pose_changed = true;
        } else {
            double dx = msg->pose.pose.position.x - last_saved_pose_.pose.pose.position.x;
            double dy = msg->pose.pose.position.y - last_saved_pose_.pose.pose.position.y;
            double dz = msg->pose.pose.position.z - last_saved_pose_.pose.pose.position.z;
            double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            // Save if moved more than 1cm (same threshold as global localization)
            if (distance > 0.01) {
                pose_changed = true;
            }
        }
        
        if (pose_changed) {
            std::ofstream file(output_file_, std::ios_base::app);
            
            if (file.is_open()) {
                auto current_time = this->now();
                auto nanoseconds = current_time.nanoseconds();
                auto seconds = nanoseconds / 1000000000;
                auto fractional = nanoseconds % 1000000000;
                
                file << seconds << "." << std::setw(9) << std::setfill('0') << fractional << ","
                     << msg->pose.pose.position.x << ","
                     << msg->pose.pose.position.y << ","
                     << msg->pose.pose.position.z << ","
                     << msg->pose.pose.orientation.x << ","
                     << msg->pose.pose.orientation.y << ","
                     << msg->pose.pose.orientation.z << ","
                     << msg->pose.pose.orientation.w << "\n";
                
                file.close();
                last_saved_pose_ = *msg;
                
                RCLCPP_DEBUG(this->get_logger(), "Saved AMCL pose to %s", output_file_.c_str());
            } else {
                RCLCPP_WARN(this->get_logger(), "Failed to open file %s for writing", output_file_.c_str());
            }
        }
    }

    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_subscription_;
    geometry_msgs::msg::PoseWithCovarianceStamped last_saved_pose_;
    std::string output_file_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<AMCLPathRecorder>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}