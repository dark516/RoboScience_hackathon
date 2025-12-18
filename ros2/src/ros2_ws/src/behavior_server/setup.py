from setuptools import setup
from glob import glob
import os

package_name = 'behavior_server'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='PID-based path follower that consumes nav_msgs/Path and publishes cmd_vel.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # формат: <имя_исполняемого_скрипта> = <пакет>.<модуль>:<функция>
            'path_follower_pid_node = behavior_server.path_follower_pid_node:main'
#            'behavior_server_node = behavior_server.behavior_server_node:main'
        ],
    },
)
