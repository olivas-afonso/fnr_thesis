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
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <visualization_msgs/msg/marker.hpp>



#include <pcl/common/common.h>
#include <pcl/common/pca.h>



#include <vector>
#include <cmath>
#include <numeric>



class KalmanCircleLocalization : public rclcpp::Node
{
public:
    KalmanCircleLocalization() : Node("kalman_circ_node")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ground_plane", rclcpp::SensorDataQoS(),
            std::bind(&KalmanCircleLocalization::pointCloudCallback, this, std::placeholders::_1));

        pose_subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/zed/zed_node/pose", 10,
            std::bind(&KalmanCircleLocalization::poseCallback, this, std::placeholders::_1));

        white_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/white_only", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("camera_axes", 10);
    
        curve_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("line_fit_original", 10);
        cluster_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("clusters", 10);

        this->declare_parameter("fit_side", true);
        this->get_parameter("fit_side", fit_side_);

        // Initialize state [x_c, y_c, r, theta]
        state = Eigen::VectorXf::Zero(6);
        state << 0, 0, 3.0, 0, 0, 3.0; // Assume initial curve with 5m radius

        // Initialize covariance matrix
        P = Eigen::MatrixXf::Identity(6,6) * 0.1;

        // Process and measurement noise
        Q = Eigen::MatrixXf::Identity(6,6) * 0.01;
        R = Eigen::MatrixXf::Identity(6,6) * 0.5;
        I = Eigen::MatrixXf::Identity(6,6);

        // State transition model (assume slow-moving lane changes)
        F = Eigen::MatrixXf::Identity(6,6);



    }

private:
    Eigen::VectorXf state; // [x_c, y_c, r, theta]
    Eigen::MatrixXf P, Q, R, I, F;
    float left_start_angle = 0.0, left_end_angle = 0.0;
    float right_start_angle = 0.0, right_end_angle = 0.0;


    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        current_pose_ = *msg;
    }

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr white_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        for (const auto &point : cloud->points)
        {
            float h, s, v;
            RGBtoHSV(point.r, point.g, point.b, h, s, v);
            if (v > 0.65f && s < 0.2f)
            {
                white_cloud->push_back(pcl::PointXYZ(point.x, point.y, point.z));
            }
        }
        publishCameraAxes(current_pose_);
        if (white_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "No white points detected!");
            return;
        }

        std::vector<pcl::PointIndices> selected_clusters = clusterWhitePoints(white_cloud, current_pose_.pose);
        /*
        if (selected_clusters.size() < 2)
        {
            RCLCPP_WARN(this->get_logger(), "Not enough clusters detected!");
            return;
        }
        */

        publishClusterMarkers(white_cloud, selected_clusters);

        sensor_msgs::msg::PointCloud2 white_msg;
        pcl::toROSMsg(*white_cloud, white_msg);
        white_msg.header = msg->header;
        white_publisher_->publish(white_msg);

        pcl::PointIndices rightmost_cluster, leftmost_cluster;
        Eigen::Vector2f left_observed, right_observed;
        bool left_detected = false, right_detected = false;
        float max_angle = -std::numeric_limits<float>::infinity();
        float min_angle = std::numeric_limits<float>::infinity();
            
        Eigen::Vector3f camera_position(0.0, 0.0, 0.0);  
        camera_position[0]=current_pose_.pose.position.x;
        camera_position[1]=current_pose_.pose.position.y;
        camera_position[2]=current_pose_.pose.position.z;
        
        if (selected_clusters.size() == 1)
        {
            Eigen::Vector3f centroid(0.0, 0.0, 0.0);
            pcl::PointIndices cluster = selected_clusters[0]; 
            for (int idx : cluster.indices)
            {
                centroid[0] += white_cloud->points[idx].x;
                centroid[1] += white_cloud->points[idx].y;
                centroid[2] += white_cloud->points[idx].z;
            }
            centroid /= cluster.indices.size();  // Compute centroid
    
            float angle = atan2(centroid[1] - camera_position[1], centroid[0] - camera_position[0]);
            
            if (angle >= 0)  // Cluster is on the left side
            {
                leftmost_cluster = cluster;
                left_observed = centroid.head<2>();
                left_detected = true;
                RCLCPP_INFO(this->get_logger(), "OI");
            }
            else  // Cluster is on the right side
            {
                rightmost_cluster = cluster;
                right_observed = centroid.head<2>();
                right_detected = true;
            }
        }
        else  // Standard case: multiple clusters
        {
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
                    left_observed = centroid.head<2>();
                    left_detected = true;
                }
            
                if (angle < min_angle) 
                {
                    min_angle = angle;
                    rightmost_cluster = cluster;
                    right_observed = centroid.head<2>();
                    right_detected = true;
                }
            }
        }

        // Fit circles to left and right lane markings
        Eigen::Vector3f left_circle, right_circle;
        //fitCircle(white_cloud, leftmost_cluster, left_circle);
        //RCLCPP_INFO(this->get_logger(), "Circle 1: x=%f, y=%f, r=%f", left_circle[0], left_circle[1], left_circle[2]);

        //fitCircle(white_cloud, rightmost_cluster, right_circle);
        //RCLCPP_INFO(this->get_logger(), "Circle 1: x=%f, y=%f, r=%f", right_circle[0], right_circle[1], right_circle[2]);
        
        Eigen::VectorXf Z(6); // Ensure correct size

        if (left_detected && right_detected)
        {
            fitCircle(white_cloud, leftmost_cluster, left_circle);
            fitCircle(white_cloud, rightmost_cluster, right_circle);

            Z << left_circle[0], left_circle[1], left_circle[2], 
                right_circle[0], right_circle[1], right_circle[2];


        }
        else if (left_detected)
        {
            fitCircle(white_cloud, leftmost_cluster, left_circle);
            Z << left_circle[0], left_circle[1], left_circle[2], 
                state[3], state[4], state[5]; // Keep last known right lane estimate


        }
        else if (right_detected)
        {
            fitCircle(white_cloud, rightmost_cluster, right_circle);
            Z << state[0], state[1], state[2],  
                right_circle[0], right_circle[1], right_circle[2];

        }
        else
        {
            return;
        }

        // Kalman Prediction Step
        state = F * state;
        P = F * P * F.transpose() + Q;

        // Kalman Update Step
        Eigen::MatrixXf S = P + R;
        Eigen::MatrixXf K = P * S.inverse();

        state = state + K * (Z - state);
        P = (I - K) * P;

        Eigen::Vector3f left_state = Z.head<3>(); 
        Eigen::Vector3f right_state = Z.tail<3>();
        //(float left_start_angle, left_end_angle, right_start_angle, right_end_angle;
        if(right_detected) computeArcAngles(white_cloud, rightmost_cluster, right_state, right_start_angle, right_end_angle);
        if(left_detected) computeArcAngles(white_cloud, leftmost_cluster, left_state, left_start_angle, left_end_angle);
        //computeArcAngles(white_cloud, leftmost_cluster, left_circle, left_start_angle, left_end_angle);
        //computeArcAngles(white_cloud, rightmost_cluster, right_circle, right_start_angle, right_end_angle);

        visualizeCircles(left_circle, right_circle, state, left_start_angle, left_end_angle, right_start_angle, right_end_angle, left_detected, right_detected);
    }


    bool fitCircle(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const pcl::PointIndices &indices, Eigen::Vector3f &circle)
    {
        if (indices.indices.size() < 3)
            return false;

        std::vector<Eigen::Vector2f> points;
        for (int idx : indices.indices)
        {
            points.emplace_back(cloud->points[idx].x, cloud->points[idx].y);
        }

        int N = points.size();
        if (N < 3)
            return false;

        float sum_x = 0, sum_y = 0, sum_x2 = 0, sum_y2 = 0;
        float sum_xy = 0, sum_x3 = 0, sum_y3 = 0;
        float sum_x2y = 0, sum_xy2 = 0;

        for (const auto &p : points)
        {
            float x = p[0], y = p[1];
            float x2 = x * x, y2 = y * y;

            sum_x += x;
            sum_y += y;
            sum_x2 += x2;
            sum_y2 += y2;
            sum_xy += x * y;
            sum_x3 += x2 * x;
            sum_y3 += y2 * y;
            sum_x2y += x2 * y;
            sum_xy2 += x * y2;
        }

        Eigen::Matrix3f A;
        Eigen::Vector3f B;

        A << sum_x2, sum_xy, sum_x,
            sum_xy, sum_y2, sum_y,
            sum_x, sum_y, N;

        B << (sum_x3 + sum_xy2) / 2,
            (sum_y3 + sum_x2y) / 2,
            (sum_x2 + sum_y2) / 2;

        Eigen::Vector3f solution = A.colPivHouseholderQr().solve(B);
        
        float x_c = solution[0];
        float y_c = solution[1];

        // Compute radius using average distance to all points
        float r = 0;
        for (const auto &p : points)
        {
            r += std::sqrt((p[0] - x_c) * (p[0] - x_c) + (p[1] - y_c) * (p[1] - y_c));
        }
        r /= N;  // Take the average

        circle = Eigen::Vector3f(x_c, y_c, r);
        return true;
    }





    void visualizeCircles(const Eigen::Vector3f &left_circle, const Eigen::Vector3f &right_circle, 
        const Eigen::VectorXf &filtered_state, float left_start_angle, float left_end_angle, 
        float right_start_angle, float right_end_angle, bool left_detected, bool right_detected)
    {
        visualization_msgs::msg::MarkerArray marker_array;

        // Visualize only the relevant parts of the circle
        if(left_detected) marker_array.markers.push_back(createArcMarker(left_circle, 0, 0.0, 0.0, 1.0, left_start_angle, left_end_angle));  // Blue
        if(right_detected) marker_array.markers.push_back(createArcMarker(right_circle, 1, 1.0, 0.0, 0.0, right_start_angle, right_end_angle)); // Red

        // Kalman-filtered arcs
        Eigen::Vector3f filtered_circle_left(filtered_state[0], filtered_state[1], filtered_state[2]);
        Eigen::Vector3f filtered_circle_right(filtered_state[3], filtered_state[4], filtered_state[5]);

        marker_array.markers.push_back(createArcMarker(filtered_circle_left, 2, 0.0, 1.0, 0.0, left_start_angle, left_end_angle)); // Green
        marker_array.markers.push_back(createArcMarker(filtered_circle_right, 3, 1.0, 1.0, 0.0, right_start_angle, right_end_angle)); // Yellow

        curve_publisher_->publish(marker_array);
    }


    visualization_msgs::msg::Marker createArcMarker(const Eigen::Vector3f &circle, int id, float r, float g, float b)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->now();
        marker.ns = "circle_fit";
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;

        for (float theta = 0; theta <= M_PI; theta += 0.1)
        {
            geometry_msgs::msg::Point p;
            p.x = circle[0] + circle[2] * cos(theta);
            p.y = circle[1] + circle[2] * sin(theta);
            marker.points.push_back(p);
        }

        marker.scale.x = 0.05;
        marker.color.r = r;
        marker.color.g = g;
        marker.color.b = b;
        marker.color.a = 1.0;

        return marker;
    }




    std::vector<pcl::PointIndices> clusterWhitePoints(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
        const geometry_msgs::msg::Pose &current_pose_) // Camera pose
    {
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);
    
        std::vector<pcl::PointIndices> clusters;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.08);
        ec.setMinClusterSize(10);
        ec.setMaxClusterSize(5000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(clusters);
    
        struct ClusterData
        {
            pcl::PointIndices indices;
            float min_distance;
            float depth; // Width in the camera direction (Z-axis)
        };
    
        std::vector<ClusterData> cluster_data;
    
        // Extract camera position
        Eigen::Vector3f camera_position(
            current_pose_.position.x,
            current_pose_.position.y,
            current_pose_.position.z);
    
        // Extract camera orientation
        tf2::Quaternion q(
            current_pose_.orientation.x,
            current_pose_.orientation.y,
            current_pose_.orientation.z,
            current_pose_.orientation.w);
        tf2::Matrix3x3 rotation_matrix(q);
    
        // Convert to Eigen rotation matrix
        Eigen::Matrix3f R;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R(i, j) = rotation_matrix[i][j];
    
        // Extract forward axis (Z direction in camera frame)
        Eigen::Vector3f forward_dir = R.col(2); // Z-axis
    
        for (const auto &cluster : clusters)
        {
            float min_proj = std::numeric_limits<float>::max();
            float max_proj = std::numeric_limits<float>::lowest();
            float sum_x = 0.0, sum_z = 0.0;
            int num_points = cluster.indices.size();
    
            for (int idx : cluster.indices)
            {
                Eigen::Vector3f point(
                    cloud->points[idx].x,
                    cloud->points[idx].y,
                    cloud->points[idx].z);
    
                // Transform to camera-aligned frame
                Eigen::Vector3f relative_point = R.transpose() * (point - camera_position);
    
                // Project onto the camera direction (Z-axis in camera frame)
                float projection = relative_point.dot(Eigen::Vector3f(0, 0, 1));
    
                min_proj = std::min(min_proj, projection);
                max_proj = std::max(max_proj, projection);
    
                sum_x += relative_point.x();
                sum_z += relative_point.z();
            }
    
            float centroid_x = sum_x / num_points;
            float centroid_z = sum_z / num_points;
            float distance = std::sqrt(centroid_x * centroid_x + centroid_z * centroid_z);
    
            float cluster_depth = max_proj - min_proj; // Width along camera direction
    
            cluster_data.push_back({cluster, distance, cluster_depth});
        }
    
        // Filter clusters based on depth (remove clusters with small Z-width)
        std::vector<pcl::PointIndices> filtered_clusters;
        for (const auto &cd : cluster_data)
        {
            if (cd.depth >= 0.3) // Adjust this threshold as needed
            {
                filtered_clusters.push_back(cd.indices);
            }
        }
    
        return filtered_clusters;
    }
    
    
    



    void computeArcAngles(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, 
        const pcl::PointIndices &cluster, 
        const Eigen::Vector3f &circle, 
        float &start_angle, float &end_angle)
    {
        if (cluster.indices.empty())
        {
        return;
        }

        float min_angle = std::numeric_limits<float>::infinity();
        float max_angle = -std::numeric_limits<float>::infinity();

        for (int idx : cluster.indices)
        {
        float x = cloud->points[idx].x;
        float y = cloud->points[idx].y;

        // Compute angle relative to the circle center
        float theta = atan2(y - circle[1], x - circle[0]);

        min_angle = std::min(min_angle, theta);
        max_angle = std::max(max_angle, theta);
        }

        start_angle = min_angle;
        end_angle = max_angle;
    }

    visualization_msgs::msg::Marker createArcMarker(const Eigen::Vector3f &circle, 
        int id, float r, float g, float b, 
        float start_angle, float end_angle)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->now();
        marker.ns = "circle_fit";
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;

        for (float theta = start_angle; theta <= end_angle; theta += 0.05) // Smaller step for smoother arc
        {
        geometry_msgs::msg::Point p;
        p.x = circle[0] + circle[2] * cos(theta);
        p.y = circle[1] + circle[2] * sin(theta);
        marker.points.push_back(p);
        }

        marker.scale.x = 0.05; // Line thickness
        marker.color.r = r;
        marker.color.g = g;
        marker.color.b = b;
        marker.color.a = 1.0;

        return marker;
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
        cluster_marker_pub_->publish(cluster_markers);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_subscription_;
    geometry_msgs::msg::PoseStamped current_pose_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr white_publisher_;
    //rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr distance_orientation_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr curve_publisher_;
    //rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher2_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr cluster_marker_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;


    rclcpp::Parameter fit_side_param_;
    bool fit_side_;

    void publishCameraAxes(const geometry_msgs::msg::PoseStamped &msg)
    {
        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker marker;

        // Extract camera position
        Eigen::Vector3f camera_position(
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z);

        // Extract rotation matrix from quaternion
        tf2::Quaternion q(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w);
        tf2::Matrix3x3 rotation_matrix(q);

        // Extract direction vectors
        Eigen::Vector3f x_axis(rotation_matrix[0][0], rotation_matrix[1][0], rotation_matrix[2][0]); // Right (Red)
        Eigen::Vector3f y_axis(rotation_matrix[0][1], rotation_matrix[1][1], rotation_matrix[2][1]); // Up (Green)
        Eigen::Vector3f z_axis(rotation_matrix[0][2], rotation_matrix[1][2], rotation_matrix[2][2]); // Forward (Blue)

        auto createArrowMarker = [&](int id, const Eigen::Vector3f &dir, const std_msgs::msg::ColorRGBA &color)
        {
            visualization_msgs::msg::Marker arrow;
            arrow.header.frame_id = "map"; // Change if needed
            arrow.header.stamp = this->now();
            arrow.ns = "camera_axes";
            arrow.id = id;
            arrow.type = visualization_msgs::msg::Marker::ARROW;
            arrow.action = visualization_msgs::msg::Marker::ADD;
            arrow.scale.x = 0.05;  // Shaft diameter
            arrow.scale.y = 0.1;   // Head diameter
            arrow.scale.z = 0.1;   // Head length

            geometry_msgs::msg::Point start, end;
            start.x = camera_position.x();
            start.y = camera_position.y();
            start.z = camera_position.z();

            end.x = start.x + dir.x() * 0.5; // Scale for visibility
            end.y = start.y + dir.y() * 0.5;
            end.z = start.z + dir.z() * 0.5;

            arrow.points.push_back(start);
            arrow.points.push_back(end);

            arrow.color = color;
            arrow.lifetime = rclcpp::Duration::from_seconds(0.5); // Short lifespan to update
            arrow.frame_locked = true;

            return arrow;
        };

        // Define colors
        std_msgs::msg::ColorRGBA red, green, blue;
        red.r = 1.0; red.a = 1.0;
        green.g = 1.0; green.a = 1.0;
        blue.b = 1.0; blue.a = 1.0;

        marker_array.markers.push_back(createArrowMarker(0, x_axis, red));   // X-axis (Right)
        marker_array.markers.push_back(createArrowMarker(1, y_axis, green)); // Y-axis (Up)
        marker_array.markers.push_back(createArrowMarker(2, z_axis, blue));  // Z-axis (Forward)

        marker_pub_->publish(marker_array);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<KalmanCircleLocalization>());
    rclcpp::shutdown();
    return 0;
}
