#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <point_cloud_interfaces/msg/compressed_point_cloud2.hpp>

// Draco
#include "draco/compression/decode.h"
#include "draco/point_cloud/point_cloud.h"

class DracoPointCloudProcessor : public rclcpp::Node {
public:
    DracoPointCloudProcessor() : Node("draco_pointcloud_processor") {

        subscription_ = this->create_subscription<point_cloud_interfaces::msg::CompressedPointCloud2>(
            "/zed/zed_node/point_cloud/cloud_registered/draco",
            10,
            std::bind(&DracoPointCloudProcessor::compressedCallback, this, std::placeholders::_1)
        );

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "decoded",
            10
        );

        RCLCPP_INFO(this->get_logger(), "Draco decompressor node initialized.");
    }

private:
    void compressedCallback(const point_cloud_interfaces::msg::CompressedPointCloud2::SharedPtr msg) {

        draco::DecoderBuffer buffer;
        buffer.Init(reinterpret_cast<const char*>(msg->compressed_data.data()), msg->compressed_data.size());

        draco::Decoder decoder;
        auto result = decoder.DecodePointCloudFromBuffer(&buffer);

        if (!result.ok()) {
            RCLCPP_ERROR(this->get_logger(), "Draco decompression failed: %s", result.status().error_msg());
            return;
        }

        std::unique_ptr<draco::PointCloud> draco_cloud = std::move(result).value();

        RCLCPP_INFO(this->get_logger(), "Decoded cloud with %d points and %d attributes.", draco_cloud->num_points(), draco_cloud->num_attributes());

        for (int i = 0; i < draco_cloud->num_attributes(); ++i) {
            const auto* attr = draco_cloud->attribute(i);
            RCLCPP_INFO(this->get_logger(), "Attribute %d: type = %d, num_components = %d", i, attr->attribute_type(), attr->num_components());
        }

        // Position attributes
        const draco::PointAttribute* x_attr = draco_cloud->attribute(0);
        const draco::PointAttribute* y_attr = draco_cloud->attribute(1);
        const draco::PointAttribute* z_attr = draco_cloud->attribute(2);

        if (!x_attr || !y_attr || !z_attr) {
            RCLCPP_ERROR(this->get_logger(), "One or more position attributes missing (X, Y, Z).");
            return;
        }

        // Try to get color attribute
        const draco::PointAttribute* color_attr = nullptr;
        if (draco_cloud->num_attributes() > 3) {
            color_attr = draco_cloud->attribute(3);
        }

        bool has_color = (color_attr && color_attr->num_components() >= 3);

        // Prepare PointCloud2 message
        auto cloud_msg = sensor_msgs::msg::PointCloud2();
        cloud_msg.header = msg->header;
        cloud_msg.height = 1;
        cloud_msg.width = draco_cloud->num_points();
        cloud_msg.is_bigendian = false;
        cloud_msg.is_dense = true;
        cloud_msg.point_step = has_color ? 16 : 12;  // 3 floats (12 bytes) + 1 rgba (4 bytes)
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
        cloud_msg.data.resize(cloud_msg.row_step * cloud_msg.height);

        // Define fields
        sensor_msgs::msg::PointField field_x, field_y, field_z;
        field_x.name = "x"; field_x.offset = 0; field_x.datatype = sensor_msgs::msg::PointField::FLOAT32; field_x.count = 1;
        field_y.name = "y"; field_y.offset = 4; field_y.datatype = sensor_msgs::msg::PointField::FLOAT32; field_y.count = 1;
        field_z.name = "z"; field_z.offset = 8; field_z.datatype = sensor_msgs::msg::PointField::FLOAT32; field_z.count = 1;

        cloud_msg.fields = { field_x, field_y, field_z };

        if (has_color) {
            sensor_msgs::msg::PointField field_color;
            field_color.name = "rgb";
            field_color.offset = 12;
            field_color.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field_color.count = 1;
            cloud_msg.fields.push_back(field_color);
        }

        for (draco::PointIndex i(0); i < draco_cloud->num_points(); ++i) {
            float x, y, z;
            x_attr->ConvertValue<float>(x_attr->mapped_index(i), &x);
            y_attr->ConvertValue<float>(y_attr->mapped_index(i), &y);
            z_attr->ConvertValue<float>(z_attr->mapped_index(i), &z);

            size_t offset = i.value() * cloud_msg.point_step;
            memcpy(&cloud_msg.data[offset + 0], &x, sizeof(float));
            memcpy(&cloud_msg.data[offset + 4], &y, sizeof(float));
            memcpy(&cloud_msg.data[offset + 8], &z, sizeof(float));

            if (has_color) {
                uint8_t color[4] = {255, 255, 255, 255}; // default white
                color_attr->GetValue(color_attr->mapped_index(i), &color[0]);
            
                // Pack BGRA into float
                uint32_t bgra = color[2] | (color[1] << 8) | (color[0] << 16) | (color[3] << 24);
                float color_float;
                memcpy(&color_float, &bgra, sizeof(float));
                memcpy(&cloud_msg.data[offset + 12], &color_float, sizeof(float));
            
                if (i.value() < 5) {
                    RCLCPP_INFO(this->get_logger(), "Decoded point %d: x=%.3f y=%.3f z=%.3f rgba=(%u, %u, %u, %u)",
                                i.value(), x, y, z, color[0], color[1], color[2], color[3]);
                }
            }
        }

        publisher_->publish(cloud_msg);
        RCLCPP_INFO(this->get_logger(), "Published PointCloud2 with %d points.", cloud_msg.width);
    }

    rclcpp::Subscription<point_cloud_interfaces::msg::CompressedPointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DracoPointCloudProcessor>());
    rclcpp::shutdown();
    return 0;
}
