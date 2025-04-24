import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Package and directory setup
    pkg_name = 'point_cloud_processor'
    pkg_dir = get_package_share_directory(pkg_name)

    # Launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock if true'
    )
    
    # TF Tree Publishers
    tf_publishers = [
        # map -> odom
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen'
        ),
        # odom -> base_footprint
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_footprint'],
            output='screen'
        ),
        # base_footprint -> base_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'base_footprint', 'base_link'],
            output='screen'
        )
    ]
    
    # Map server node
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': PathJoinSubstitution([
                FindPackageShare(pkg_name),
                'maps',
                'map.yaml'
            ]),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'negate': True
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )
    
    # AMCL node with updated parameters
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare(pkg_name),
                'config',
                'amcl_config.yaml'
            ]),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'odom_frame_id': 'odom',
                'base_frame_id': 'base_link',
                'global_frame_id': 'map',
                'transform_tolerance': 0.2
            }
        ],
        remappings=[
            ('/scan', '/cluster_scan'),
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )
    
    # Lifecycle manager
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'autostart': True,
            'node_names': ['map_server', 'amcl']
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )
    
    # Global Localization node
    global_localization = Node(
        package=pkg_name,
        executable='global_localization',
        name='global_localization',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'fit_side': True,
            'time_jump_threshold': 5.0,
            'min_scan_range': 0.1,
            'max_scan_range': 10.0
        }]
    )
    
    return LaunchDescription([
        use_sim_time,
        *tf_publishers,
        map_server,
        amcl,
        lifecycle_manager,
        global_localization
    ])