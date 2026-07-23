from setuptools import find_packages, setup

package_name = 'cpx_comm_test'

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
    maintainer='umd-user',
    maintainer_email='showmen.dey78@gmail.com',
    description='Test communication between Python 3.7 and ROS 2 Jazzy',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'localization_publisher = '
            'cpx_comm_test.localization_publisher:main',
            'perception_publisher = '
            'cpx_comm_test.perception_publisher:main',
            'traffic_light_publisher = '
            'cpx_comm_test.traffic_light_publisher:main',
            'data_subscriber = cpx_comm_test.data_subscriber:main',
        ],
    },
)
