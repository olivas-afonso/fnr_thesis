import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

ekf_config_path = PathJoinSubstitution([
    FindPackageShare('point_cloud_processor'),
    'config',
    'ekf_config.yaml'
])

def generate_launch_description():
    # Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    record_bag = LaunchConfiguration('record_bag', default='true')
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )
    
    declare_record_bag = DeclareLaunchArgument(
        'record_bag',
        default_value='true',
        description='Record data to bag file if true'
    )

    # EKF Node (minimal configuration)
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_localization_node',
        output='screen',
        parameters=[
            ekf_config_path,
            {'use_sim_time': use_sim_time}  # Add this line
        ]
    )
    # Data recording (conditionally launched)
    record_process = ExecuteProcess(
        cmd=['ros2', 'bag', 'record',
             '/zed/zed_node/odom',
             '/vesc/odom',
             '/odometry/filtered',  # EKF output
             '/tf',
             '/tf_static',
             '-o', 'odom_calibration'],
        output='screen',
        condition=IfCondition(record_bag)
    )

    # Optional: Automated movement node
    mover_node = Node(
        package='odom_calibration',
        executable='automated_mover',
        output='screen'
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_record_bag,
        ekf_node,
        record_process,
        mover_node  # Comment out if using manual control
    ])