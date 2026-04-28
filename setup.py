import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'final_challenge2026'

setup(
    name=package_name,
    version='0.0.0',
    # Automatically finds the 'final_challenge2026' python module folder
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.*')),
        (os.path.join('share', package_name, 'media'), glob('media/*')),
        (os.path.join('share', package_name, 'racetrack_images'), glob('racetrack_images/*/*.png')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MIT RSS',
    maintainer_email='racecar@mit.edu',
    description='Final challenge package for the RSS racecar',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_detector = final_challenge2026.part_a.lane_detector_node:main',
            'lane_follower = final_challenge2026.part_b.lane_follower_node:main',
        ],
    },
