import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import TimerAction
from launch_ros.actions import Node
from launch.conditions import IfCondition  # Added this import

def generate_launch_description():
    # Common launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    launch_camera = LaunchConfiguration('launch_camera')
    launch_ekf = LaunchConfiguration('launch_ekf')
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )
    
    declare_launch_camera = DeclareLaunchArgument(
        'launch_camera',
        default_value='true',
        description='Whether to launch the ZED camera'
    )
    
    declare_launch_ekf = DeclareLaunchArgument(
        'launch_ekf',
        default_value='true',
        description='Whether to launch the EKF node'
    )
    
    # EKF configuration path
    ekf_config_path = PathJoinSubstitution([
        FindPackageShare('point_cloud_processor'),
        'config',
        'ekf_config.yaml'
    ])
    
    # ZED Camera launch (conditional)
    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('zed_wrapper'),
                'launch',
                'zed_camera.launch.py'
            ])
        ]),
        launch_arguments={
            'camera_model': 'zed2i',
            'publish_urdf': 'false',
            'publish_map_tf': 'false',
            'publish_tf': 'false'
        }.items(),
        condition=IfCondition(launch_camera)  # Fixed this line
    )
    
    # Point Cloud Processor launch
    point_cloud_processor_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('point_cloud_processor'),
                'launch',
                'robot_integration_test.launch.py'
            ])
        ]),
        launch_arguments={
            'use_zed_localization': 'false',
            'use_sim_time': use_sim_time
        }.items()
    )
    
    # F1Tenth Stack launch
    f1tenth_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('f1tenth_stack'),
                'launch',
                'bringup_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )
    
    # EKF Node (conditional)
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_localization_node',
        output='screen',
        parameters=[
            ekf_config_path,
            {'use_sim_time': use_sim_time}
        ],
        condition=IfCondition(launch_ekf)  # Fixed this line
    )
    
    return LaunchDescription([
        declare_use_sim_time,
        declare_launch_camera,
        declare_launch_ekf,
        
        # Conditional nodes
        zed_launch,
        ekf_node,
        
        # Always launched nodes
        point_cloud_processor_launch,
        f1tenth_launch
    ])