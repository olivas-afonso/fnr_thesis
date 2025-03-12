#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include "std_msgs/msg/float32_multi_array.hpp"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

class RelLocalizationProcessor : public rclcpp::Node
{
public:
    RelLocalizationProcessor() : Node("rel_localization_processor")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ground_plane", rclcpp::SensorDataQoS(),
            std::bind(&RelLocalizationProcessor::pointCloudCallback, this, std::placeholders::_1));

        white_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/white_only", 10);
        localization_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("relative_localization", 10);
        marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("visualization_markers", 10);
        marker_publisher2_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("visualization_markers2", 10);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr white_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        /*
        for (const auto& point : cloud->points)
        {
            int brightness = (point.r + point.g + point.b) / 3;
            if (brightness > 150)  
            {
                white_cloud->push_back(pcl::PointXYZ(point.x, point.y, point.z));
            }
        }

        if (white_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "No white points detected!");
            return;
        }
            */

        for (const auto &point : cloud->points)
        {
            float h, s, v;
            RGBtoHSV(point.r, point.g, point.b, h, s, v);
    
            if (v > 0.6f && s < 0.2f) // High brightness & low saturation
            {
                white_cloud->push_back(pcl::PointXYZ(point.x, point.y, point.z));
            }
        }
    
        if (white_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "No white points detected!");
        }

        // Get the three closest clusters
        std::vector<pcl::PointIndices> selected_clusters = clusterWhitePoints(white_cloud);

        if (selected_clusters.size() < 3)
        {
            RCLCPP_WARN(this->get_logger(), "Not enough clusters detected!");
            return;
        }

        publishClusterMarkers(white_cloud, selected_clusters);

        sensor_msgs::msg::PointCloud2 white_msg;
        pcl::toROSMsg(*white_cloud, white_msg);
        white_msg.header = msg->header;
        white_publisher_->publish(white_msg);

        // ------------------- UPDATED: Selecting Leftmost and Rightmost Clusters -------------------

        pcl::PointIndices rightmost_cluster, leftmost_cluster;
        float max_angle = -std::numeric_limits<float>::infinity();
        float min_angle = std::numeric_limits<float>::infinity();
        
        Eigen::Vector3f camera_position(0.0, 0.0, 0.0);  // Assuming camera at (0,0,0) in its own frame

        for (const auto& cluster : selected_clusters)
        {
            Eigen::Vector3f centroid(0.0, 0.0, 0.0);
            for (int idx : cluster.indices)
            {
                centroid[0] += white_cloud->points[idx].x;
                centroid[1] += white_cloud->points[idx].y;
                centroid[2] += white_cloud->points[idx].z;
            }
            centroid /= cluster.indices.size();  // Compute centroid

            // Compute relative angle to the camera
            float angle = atan2(centroid[1] - camera_position[1], centroid[0] - camera_position[0]);

            if (angle > max_angle) 
            {
                max_angle = angle;
                leftmost_cluster = cluster;
            }

            if (angle < min_angle) 
            {
                min_angle = angle;
                rightmost_cluster = cluster;
            }
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr right_cluster(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr left_cluster(new pcl::PointCloud<pcl::PointXYZ>());
        
        if (!rightmost_cluster.indices.empty())
            pcl::copyPointCloud(*white_cloud, leftmost_cluster, *right_cluster);
        
        if (!leftmost_cluster.indices.empty())
            pcl::copyPointCloud(*white_cloud, rightmost_cluster, *left_cluster);

        if (!right_cluster->empty())
            publishCurveMarkers(right_cluster, "right_curve", 1, 0.0, 1.0); // Green for right

        if (!left_cluster->empty())
            publishCurveMarkers(left_cluster, "left_curve", 2, 1.0, 0.0); // Red for left

        float distance, orientation;
        computeFrenetFrame(right_cluster, distance, orientation);

        std_msgs::msg::Float32MultiArray localization_msg;
        localization_msg.data = {distance, orientation};
        localization_publisher_->publish(localization_msg);
    }

    void RGBtoHSV(int r, int g, int b, float &h, float &s, float &v)
    {
        float rf = r / 255.0f, gf = g / 255.0f, bf = b / 255.0f;
        float maxC = std::max({rf, gf, bf});
        float minC = std::min({rf, gf, bf});
        v = maxC;

        float delta = maxC - minC;
        if (delta < 1e-5)
        {
            h = 0.0f;
            s = 0.0f;
            return;
        }

        s = (maxC > 0.0f) ? (delta / maxC) : 0.0f;

        if (maxC == rf)
            h = 60.0f * (fmod(((gf - bf) / delta), 6));
        else if (maxC == gf)
            h = 60.0f * (((bf - rf) / delta) + 2);
        else
            h = 60.0f * (((rf - gf) / delta) + 4);

        if (h < 0)
            h += 360.0f;
    }


    std::vector<pcl::PointIndices> clusterWhitePoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);

        std::vector<pcl::PointIndices> clusters;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.1);
        ec.setMinClusterSize(30);
        ec.setMaxClusterSize(1500);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(clusters);

        if (clusters.size() < 3)
        {
            RCLCPP_WARN(this->get_logger(), "Not enough clusters detected!");
            return {};
        }

        struct ClusterData
        {
            pcl::PointIndices indices;
            float min_distance;
        };
        
        std::vector<ClusterData> cluster_data;

        for (const auto& cluster : clusters)
        {
            float min_distance = std::numeric_limits<float>::max();
            for (int idx : cluster.indices)
            {
                float x = cloud->points[idx].x;
                float z = cloud->points[idx].z;
                float distance = std::sqrt(x * x + z * z);
                min_distance = std::min(min_distance, distance);
            }
            cluster_data.push_back({cluster, min_distance});
        }

        // Sort clusters by distance and keep the closest three
        std::sort(cluster_data.begin(), cluster_data.end(), [](const ClusterData& a, const ClusterData& b) {
            return a.min_distance < b.min_distance;
        });

        return {cluster_data[0].indices, cluster_data[1].indices, cluster_data[2].indices};
    }

    

    float computeClusterMeanX(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const pcl::PointIndices& indices)
    {
        if (indices.indices.empty()) return 0.0;

        float sum_x = 0.0;
        for (int idx : indices.indices) {
            sum_x += cloud->points[idx].x;
        }
        float mean_x = sum_x / indices.indices.size();
        return mean_x;
    }

    void computeFrenetFrame(pcl::PointCloud<pcl::PointXYZ>::Ptr right,
                            float &distance, float &orientation)
    {
        float right_x = computeClusterMeanX(right, pcl::PointIndices());
        distance = right_x;
        orientation = 0.0; 
    }
    void publishCurveMarkers(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster, 
        const std::string& ns, int marker_id,
        float r, float g)
    {
        if (cluster->size() < 3)
        {
        RCLCPP_WARN(this->get_logger(), "Not enough points to fit a curve.");
        return;
        }

        std::vector<float> x_vals, y_vals;
        for (const auto &point : cluster->points)
        {
        x_vals.push_back(point.x);
        y_vals.push_back(point.y);
        }

        // Fit a quadratic curve: y = ax^2 + bx + c
        Eigen::MatrixXf A(x_vals.size(), 3);
        Eigen::VectorXf Y(y_vals.size());

        for (size_t i = 0; i < x_vals.size(); i++)
        {
        A(i, 0) = x_vals[i] * x_vals[i]; // x^2
        A(i, 1) = x_vals[i];             // x
        A(i, 2) = 1;                     // constant
        Y(i) = y_vals[i];
        }

        Eigen::VectorXf coeffs = A.colPivHouseholderQr().solve(Y);
        float a = coeffs(0), b = coeffs(1), c = coeffs(2);

        // Create a line strip marker
        visualization_msgs::msg::Marker curve_marker;
        curve_marker.header.frame_id = "map";
        curve_marker.header.stamp = this->get_clock()->now();
        curve_marker.ns = ns; // Different namespaces for left and right clusters
        curve_marker.id = marker_id;
        curve_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        curve_marker.scale.x = 0.05; // Line width
        curve_marker.color.r = r;
        curve_marker.color.g = g;
        curve_marker.color.b = 0.0;
        curve_marker.color.a = 1.0;

        for (float x = *std::min_element(x_vals.begin(), x_vals.end());
        x <= *std::max_element(x_vals.begin(), x_vals.end()); x += 0.05)
        {
        geometry_msgs::msg::Point p;
        p.x = x;
        p.y = a * x * x + b * x + c; // Quadratic equation
        p.z = 0.0;
        curve_marker.points.push_back(p);
        }

        visualization_msgs::msg::MarkerArray markers;
        markers.markers.push_back(curve_marker);
        marker_publisher_->publish(markers);
    }

    void publishClusterMarkers(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::vector<pcl::PointIndices>& clusters)
    {
        visualization_msgs::msg::MarkerArray cluster_markers;
        std::vector<std::tuple<float, float, float>> colors = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

        for (size_t i = 0; i < clusters.size(); ++i)
        {
            visualization_msgs::msg::Marker cluster_marker;
            cluster_marker.header.frame_id = "map";
            cluster_marker.header.stamp = this->get_clock()->now();
            cluster_marker.ns = "clusters";
            cluster_marker.id = i;
            cluster_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
            cluster_marker.scale.x = 0.05;
            cluster_marker.scale.y = 0.05;
            cluster_marker.scale.z = 0.05;

            auto [r, g, b] = colors[i % colors.size()];
            cluster_marker.color.r = r;
            cluster_marker.color.g = g;
            cluster_marker.color.b = b;
            cluster_marker.color.a = 1.0;

            for (int idx : clusters[i].indices)
            {
                geometry_msgs::msg::Point p;
                p.x = cloud->points[idx].x;
                p.y = cloud->points[idx].y;
                p.z = cloud->points[idx].z;
                cluster_marker.points.push_back(p);
            }
            cluster_markers.markers.push_back(cluster_marker);
        }
        marker_publisher2_->publish(cluster_markers);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr white_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr localization_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher2_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RelLocalizationProcessor>());
    rclcpp::shutdown();
    return 0;
}