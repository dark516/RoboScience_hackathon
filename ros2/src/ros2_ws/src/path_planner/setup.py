from setuptools import find_packages, setup

package_name = 'path_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dark516',
    maintainer_email='sashakulagin2007@gmail.com',
    description='A* path planner with obstacle avoidance',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'path_planner_node = path_planner.path_planner_node:main'
        ],
    },
)
