#include "kalman_rel_localization/visualization.hpp"
#include <visualization_msgs/msg/marker.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

namespace lane_detection_visualization {

    void visualizeKalmanState(
        const Eigen::VectorXf& state,
        const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& publisher,
        bool left_detected,
        bool right_detected,
        const pcl::PointIndices& left_cluster_indices,
        const pcl::PointIndices& right_cluster_indices,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& white_cloud,
        const geometry_msgs::msg::Pose &current_pose_,
        const rclcpp::Time& current_time)
    {
        visualization_msgs::msg::MarkerArray marker_array;
        
        // 1. Get current camera pose and orientation
        tf2::Quaternion q(
            current_pose_.orientation.x,
            current_pose_.orientation.y,
            current_pose_.orientation.z,
            current_pose_.orientation.w);
        tf2::Matrix3x3 rot(q);
        Eigen::Vector3f cam_pos(
            current_pose_.position.x,
            current_pose_.position.y,
            current_pose_.position.z);
    
        // Convert tf2::Matrix3x3 to Eigen::Matrix3f
        Eigen::Matrix3f rot_eigen;
        for(int i=0; i<3; i++) {
            for(int j=0; j<3; j++) {
                rot_eigen(i,j) = rot[i][j];
            }
        }
    
        // 2. Extract camera-relative coefficients from state
        Eigen::Vector3f left_coeffs(state[0], state[1], state[2]);  // a_L, b_L, c_L
        Eigen::Vector3f right_coeffs(state[3], state[4], state[5]); // a_R, b_R, c_R
    
        // 3. Determine visualization range in camera X coordinates
        auto getCameraXRange = [&](const pcl::PointIndices& indices) {
            if (indices.indices.empty()) {
                return std::make_pair(-5.0f, 5.0f); // Default 10m forward
            }
    
            float min_x = std::numeric_limits<float>::max();
            float max_x = std::numeric_limits<float>::lowest();
            
            for (int idx : indices.indices) {
                const auto& point = white_cloud->points[idx];
                Eigen::Vector3f pt_map(point.x, point.y, point.z);
                Eigen::Vector3f pt_cam = rot_eigen.transpose() * (pt_map - cam_pos);
                
                min_x = std::min(min_x, pt_cam.x());
                max_x = std::max(max_x, pt_cam.x());
            }
            return std::make_pair(min_x - 0.0f, max_x + 0.0f); // Add small padding
        };
    
        auto [left_min_x, left_max_x] = getCameraXRange(left_cluster_indices);
        auto [right_min_x, right_max_x] = getCameraXRange(right_cluster_indices);
    
        // 4. Create left lane marker (green for detected, blue for estimated)
        visualization_msgs::msg::Marker left_marker;
        left_marker.header.frame_id = "map";
        left_marker.header.stamp = current_time;
        left_marker.ns = "kalman_lanes";
        left_marker.id = 0;
        left_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        left_marker.action = visualization_msgs::msg::Marker::ADD;
        left_marker.scale.x = 0.05; // Line width
        left_marker.color.a = 1.0;
        left_marker.color.g = left_detected ? 1.0 : 0.5;  // Green if detected
        left_marker.color.b = left_detected ? 0.0 : 1.0;  // Blue if estimated
        left_marker.pose.orientation.w = 1.0;
    
        // 5. Create right lane marker (red for detected, pink for estimated)
        visualization_msgs::msg::Marker right_marker = left_marker;
        right_marker.id = 1;
        right_marker.color.r = 1.0;
        right_marker.color.g = right_detected ? 0.0 : 0.5;
        right_marker.color.b = right_detected ? 0.0 : 0.5;
    
        // 6. Generate points for left lane
        const float step_size = 0.2f; // meters between points
        for (float x_cam = left_min_x; x_cam <= left_max_x; x_cam += step_size) {
            // Calculate y in camera frame
            float y_cam = left_coeffs[0]*x_cam*x_cam + left_coeffs[1]*x_cam + left_coeffs[2];
            
            // Transform to map frame
            Eigen::Vector3f p_cam(x_cam, y_cam, 0);
            Eigen::Vector3f p_map = rot_eigen * p_cam + cam_pos;
            
            geometry_msgs::msg::Point p;
            p.x = p_map.x();
            p.y = p_map.y();
            p.z = 0;
            left_marker.points.push_back(p);
        }
    
        // 7. Generate points for right lane
        for (float x_cam = right_min_x; x_cam <= right_max_x; x_cam += step_size) {
            // Calculate y in camera frame
            float y_cam = right_coeffs[0]*x_cam*x_cam + right_coeffs[1]*x_cam + right_coeffs[2];
            
            // Transform to map frame
            Eigen::Vector3f p_cam(x_cam, y_cam, 0);
            Eigen::Vector3f p_map = rot_eigen * p_cam + cam_pos;
            
            geometry_msgs::msg::Point p;
            p.x = p_map.x();
            p.y = p_map.y();
            p.z = 0;
            right_marker.points.push_back(p);
        }
    
        // 8. Add markers to array and publish
        marker_array.markers.push_back(left_marker);
        marker_array.markers.push_back(right_marker);
        publisher->publish(marker_array);
    }

    void publishClusterMarkers(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
        const std::vector<pcl::PointIndices>& clusters,
        const geometry_msgs::msg::Pose& camera_pose,
        const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& publisher,
        const rclcpp::Time& current_time)
    {
        visualization_msgs::msg::MarkerArray cluster_markers;

        // Camera position and orientation in map frame
        Eigen::Vector3f camera_position(
        camera_pose.position.x,
        camera_pose.position.y,
        camera_pose.position.z);

        // Get camera orientation
        tf2::Quaternion q(
        camera_pose.orientation.x,
        camera_pose.orientation.y,
        camera_pose.orientation.z,
        camera_pose.orientation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        const float cos_yaw = cos(yaw);
        const float sin_yaw = sin(yaw);

        for (size_t i = 0; i < clusters.size(); ++i)
        {
            // Voting system for left/right determination
            int left_votes = 0;
            int right_votes = 0;
            float total_y_cam = 0.0f; // Accumulated y position in camera frame

            // First pass: analyze all points
            for (int idx : clusters[i].indices) {
                const auto& point = cloud->points[idx];
                Eigen::Vector3f relative_pos(point.x - camera_position[0],
                                    point.y - camera_position[1],
                                    point.z - camera_position[2]);

                // Transform to camera frame
                float y_cam = -relative_pos[0] * sin_yaw + relative_pos[1] * cos_yaw;
                total_y_cam += y_cam;

                // Vote based on point's position
                if (y_cam > 0) {
                    left_votes++;
                    
                } else {
                    right_votes++;
                }
            }

            // Determine cluster side based on voting
            bool is_left;
            if (left_votes == 0 && right_votes == 0) {
                is_left = true; // default if empty (shouldn't happen)

            } else {
                // Use both voting and mean y position for robustness
                float mean_y_cam = total_y_cam / clusters[i].indices.size();
                is_left = (left_votes > right_votes) || ((left_votes == right_votes) && (mean_y_cam > 0));
            }

            // Create marker
            visualization_msgs::msg::Marker cluster_marker;
            cluster_marker.header.frame_id = "map";
            cluster_marker.header.stamp = current_time;
            cluster_marker.ns = "clusters";
            cluster_marker.id = i;
            cluster_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
            cluster_marker.scale.x = 0.05;
            cluster_marker.scale.y = 0.05;
            cluster_marker.scale.z = 0.05;

            // Set color based on left/right
            if (is_left) {
                // Left cluster - green
                cluster_marker.color.r = 0.0;
                cluster_marker.color.g = 1.0;
                cluster_marker.color.b = 0.0;
            } else {
                // Right cluster - red
                cluster_marker.color.r = 1.0;
                cluster_marker.color.g = 0.0;
                cluster_marker.color.b = 0.0;
            }
            cluster_marker.color.a = 1.0;

            // Add points
            for (int idx : clusters[i].indices) {
                geometry_msgs::msg::Point p;
                p.x = cloud->points[idx].x;
                p.y = cloud->points[idx].y;
                p.z = cloud->points[idx].z;
                cluster_marker.points.push_back(p);
            }

            cluster_markers.markers.push_back(cluster_marker);
        }

        publisher->publish(cluster_markers);
    }

    void testClusterMarkers(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
        const pcl::PointIndices& left_indices, 
        const pcl::PointIndices& right_indices,
        const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& publisher,
        const rclcpp::Time& current_time)
    {
    visualization_msgs::msg::MarkerArray cluster_markers;

    // Define colors for left (red) and right (blue) clusters
    std::vector<std::tuple<float, float, float>> colors = {{1.0, 0.0, 0.0},  // Left cluster (Red)
                                            {0.0, 0.0, 1.0}}; // Right cluster (Blue)

    // Helper lambda to create markers
    auto createMarker = [&](const pcl::PointIndices& indices, int id, const std::tuple<float, float, float>& color) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = current_time;
    marker.ns = "clusters";
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;

    auto [r, g, b] = color;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = 1.0;

    for (int idx : indices.indices)
    {
    geometry_msgs::msg::Point p;
    p.x = cloud->points[idx].x;
    p.y = cloud->points[idx].y;
    p.z = cloud->points[idx].z;
    marker.points.push_back(p);
    }

    return marker;
    };

    // Add left and right cluster markers
    cluster_markers.markers.push_back(createMarker(left_indices, 0, colors[0]));  // Left
    cluster_markers.markers.push_back(createMarker(right_indices, 1, colors[1])); // Right

    // Publish the markers
    publisher->publish(cluster_markers);
    }

    void publishCameraAxes(
        const geometry_msgs::msg::PoseStamped &msg,
        const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& publisher,
        const rclcpp::Time& current_time)
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
            arrow.header.stamp = current_time;
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

        publisher->publish(marker_array);
    }

} // namespace lane_detection_visualization