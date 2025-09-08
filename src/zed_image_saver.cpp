#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <filesystem>

class ZedImageSaver : public rclcpp::Node
{
public:
    ZedImageSaver() : Node("zed_image_saver"), exit_flag_(false), image_counter_(1)
    {
        // Subscribe to ZED2i image topic - let's try the left image topic instead
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/zed/zed_node/left/image_rect_color", 10,
            std::bind(&ZedImageSaver::image_callback, this, std::placeholders::_1));
        
        // Create images directory
        std::string dir_path = "/home/" + std::string(getenv("USER")) + "/zed_saved_images/bigger_test_3";
        std::filesystem::create_directories(dir_path);
        
        // Start keyboard listener thread
        keyboard_thread_ = std::thread(&ZedImageSaver::keyboard_listener, this);
        
        RCLCPP_INFO(this->get_logger(), "ZED2i Image Saver Node Started");
        RCLCPP_INFO(this->get_logger(), "Press SPACE to save current image");
        RCLCPP_INFO(this->get_logger(), "Press ESC or 'q' to exit");
    }

    ~ZedImageSaver()
    {
        exit_flag_ = true;
        if (keyboard_thread_.joinable()) {
            keyboard_thread_.join();
        }
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Store the latest image
        latest_image_ = msg;
    }

    int kbhit()
    {
        struct termios oldt, newt;
        int ch;
        int oldf;

        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

        ch = getchar();

        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        fcntl(STDIN_FILENO, F_SETFL, oldf);

        if (ch != EOF) {
            ungetc(ch, stdin);
            return 1;
        }

        return 0;
    }

    void keyboard_listener()
    {
        while (!exit_flag_ && rclcpp::ok()) {
            if (kbhit()) {
                int key = getchar();
                
                if (key == 32) { // SPACE key
                    save_image();
                } else if (key == 27) { // ESC key
                    RCLCPP_INFO(this->get_logger(), "Exiting...");
                    rclcpp::shutdown();
                    break;
                } else if (key == 'q' || key == 'Q') { // Alternative exit
                    RCLCPP_INFO(this->get_logger(), "Exiting...");
                    rclcpp::shutdown();
                    break;
                }
            }
            
            // Small delay to prevent CPU overload
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    void save_image()
    {
        if (!latest_image_) {
            RCLCPP_WARN(this->get_logger(), "No image received yet!");
            return;
        }

        try {
            // Convert ROS image to OpenCV format
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(latest_image_, "bgr8");
            
            // Generate filename with sequential numbering
            std::string filename = "/home/" + std::string(getenv("USER")) + 
                                  "/zed_saved_images/bigger_test_3/lights_" + 
                                  std::to_string(image_counter_++) + ".jpg";
            
            // Save image
            if (cv::imwrite(filename, cv_ptr->image)) {
                RCLCPP_INFO(this->get_logger(), "Image saved as: %s", filename.c_str());
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to save image: %s", filename.c_str());
            }
            
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV Bridge error: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error: %s", e.what());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    sensor_msgs::msg::Image::SharedPtr latest_image_;
    std::thread keyboard_thread_;
    std::atomic<bool> exit_flag_;
    std::atomic<int> image_counter_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ZedImageSaver>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}