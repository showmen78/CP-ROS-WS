from setuptools import find_packages, setup

package_name = 'cpx_planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    
    package_data={
        "cpx_planning": [
            "config/*.yaml",
            "MPC/*.yaml",
            "behavior_planner/*.yaml",
            "Global_Planner/*.sh",
            "Global_Planner/*.yaml",
            "Global_Planner/maps/*.xodr",
        ],
    },
    include_package_data=True,
    
    install_requires=['setuptools', 'numpy', 'scipy', 'osqp', 'PyYAML'],
    zip_safe=True,
    maintainer='umd-user',
    maintainer_email='showmen.dey78@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "planner_node = cpx_planning.planner_node:main",
        ],
    },
)
