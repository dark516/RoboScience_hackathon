from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    
    # Declare launch arguments
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyACM0',
        description='Serial port device'
    )
    
    failsafe_timeout_arg = DeclareLaunchArgument(
        'failsafe_timeout', 
        default_value='1.0',
        description='Failsafe timeout in seconds'
    )
    
    return LaunchDescription([
        serial_port_arg,
        failsafe_timeout_arg,
        
        # Node: vel_to_sticks
        Node(
            package='vel_to_sticks',
            executable='vel_to_sticks',
            name='vel_to_sticks',
            parameters=[{
                'serial_port': LaunchConfiguration('serial_port'),
                'failsafe_timeout': LaunchConfiguration('failsafe_timeout')
            }]
        ),
        
        # Launch: vision_pose
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('vision_pose'), 
                    'launch',
                    'vision_pose.launch.py'
                ])
            ])
        ),
    ])
