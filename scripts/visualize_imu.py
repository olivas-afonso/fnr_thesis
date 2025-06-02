#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
from tf2_ros import TransformBroadcaster
import math

class ImuVisualizer(Node):
    def __init__(self):
        super().__init__('imu_visualizer')
        
        # Subscribe to IMU data
        self.sub_imu = self.create_subscription(
            Imu,
            '/sensors/imu/raw',
            self.imu_callback,
            10)
        
        # Publishers for visualization
        self.pose_pub = self.create_publisher(PoseStamped, '/imu_pose', 10)
        self.twist_pub = self.create_publisher(TwistStamped, '/imu_twist', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.get_logger().info("IMU Visualizer Ready - Launch RViz2")

    def imu_callback(self, msg):
        # Create pose from orientation
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose.orientation = msg.orientation
        self.pose_pub.publish(pose_msg)
        
        # Create twist from angular velocity
        twist_msg = TwistStamped()
        twist_msg.header = msg.header
        twist_msg.twist.angular = msg.angular_velocity
        self.twist_pub.publish(twist_msg)
        
        # Create TF frame for visualization
        tf_msg = TransformStamped()
        tf_msg.header = msg.header
        tf_msg.child_frame_id = "imu_viz_frame"
        tf_msg.transform.rotation = msg.orientation
        self.tf_broadcaster.sendTransform(tf_msg)
        
        # Log orientation in degrees (roll/pitch/yaw)
        q = msg.orientation
        roll = math.atan2(2*(q.w*q.x + q.y*q.z), 1 - 2*(q.x*q.x + q.y*q.y))
        pitch = math.asin(2*(q.w*q.y - q.z*q.x))
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.get_logger().info(
            f"IMU Orientation - Roll: {math.degrees(roll):.1f}°, Pitch: {math.degrees(pitch):.1f}°, Yaw: {math.degrees(yaw):.1f}°",
            throttle_duration_sec=1)

def main():
    rclpy.init()
    node = ImuVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down IMU visualizer...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()