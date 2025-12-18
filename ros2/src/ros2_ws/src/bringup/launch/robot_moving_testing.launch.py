from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Объявляем параметры которые можно переопределить при запуске
    max_angular_vel_arg = DeclareLaunchArgument(
        'max_angular_vel',
        default_value='0.2',
        description='Максимальная угловая скорость'
    )
    
    max_linear_vel_arg = DeclareLaunchArgument(
        'max_linear_vel', 
        default_value='0.2',
        description='Максимальная линейная скорость'
    )
    
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyUSB0',
        description='Последовательный порт для связи с мотор-контроллером'
    )
    
    failsafe_timeout_arg = DeclareLaunchArgument(
        'failsafe_timeout',
        default_value='1.0',
        description='Таймаут аварийного отключения в секундах'
    )

    return LaunchDescription([
        # Аргументы
        max_angular_vel_arg,
        max_linear_vel_arg,
        serial_port_arg,
        failsafe_timeout_arg,
        
        # Нода джойстика
        Node(
            package='joy',
            executable='joy_node',
            output='screen',
            name='joy_node',
            parameters=[{
                'dev': '/dev/input/js0',  # можно изменить при необходимости
                'deadzone': 0.1,
                'autorepeat_rate': 20.0
            }]
        ),
        
        # Контроллер джойстика
        Node(
            package='joy_controller',
            executable='joy_controller',
            output='screen',
            name='joy_controller',
            parameters=[{
                'max_angular_vel': LaunchConfiguration('max_angular_vel'),
                'max_linear_vel': LaunchConfiguration('max_linear_vel'),
            }]
        ),
        
        # Конвертер скорости в команды для моторов
        Node(
            package='vel_to_sticks',
            executable='vel_to_sticks',
            output='screen',
            name='vel_to_sticks',
            parameters=[{
                'serial_port': LaunchConfiguration('serial_port'),
                'failsafe_timeout': LaunchConfiguration('failsafe_timeout'),
            }]
        )
    ])