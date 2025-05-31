import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import TimerAction, ExecuteProcess
from launch.substitutions import Command


ekf_config_path = PathJoinSubstitution([
    FindPackageShare('point_cloud_processor'),
    'config',
    'ekf_config.yaml'
])

def generate_launch_description():
    # Package and directory setup
    pkg_name = 'point_cloud_processor'
    pkg_dir = get_package_share_directory(pkg_name)
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock if true'
    )
    
    # TF Tree Publishers
    tf_publishers = [
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        )
    ]

    map_transformer_node = Node(
        package=pkg_name,
        executable='map_transformer',
        name='map_transformer',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[{
            'yaml_filename': PathJoinSubstitution([
                FindPackageShare(pkg_name),
                'maps',
                'map.yaml'
            ]),
            'use_sim_time': use_sim_time,
            'topic_name': 'map',
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )

    map_republisher = Node(
        package='point_cloud_processor',
        executable='map_republisher',
        name='map_republisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[
            PathJoinSubstitution([FindPackageShare(pkg_name), 'config', 'amcl_config.yaml']),
            {
                'use_sim_time': use_sim_time,
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
    
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server', 'amcl']
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )
    
    global_localization = Node(
        package=pkg_name,
        executable='global_localization',
        name='global_localization',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'fit_side': True,
            'time_jump_threshold': 5.0,
            'min_scan_range': 0.1,
            'max_scan_range': 10.0
        }]
    )

    tf_publisher_node = Node(
        package='point_cloud_processor',
        executable='dynamic_tf_publisher',
        name='dynamic_tf_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_localization_node',
        output='screen',
        parameters=[
            ekf_config_path,
            {'use_sim_time': use_sim_time}
        ]
    )

    map_creator_node = Node(
        package='point_cloud_processor',
        executable='map_creator_ls',
        name='map_creator_ls',
        output='screen',
        parameters=[{
            'map_resolution': 0.05,
            'map_width': 40.0,
            'map_height': 40.0,
            'map_frame': 'map',
            'output_dir': '/home/olivas/maps'
        }]
    )

    # Delay remaining nodes AND bag playback
    delayed_nodes = TimerAction(
        period=3.0,
        actions=[
            ekf_node,
            global_localization,
            # map_creator_node,
        ]
    )

    return LaunchDescription([
        declare_use_sim_time,
        *tf_publishers,
        map_server,
        amcl,
        lifecycle_manager,
        
        delayed_nodes
    ])
