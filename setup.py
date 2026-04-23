import glob
import os

from setuptools import setup


package_name = 'final_challenge2026'


setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    scripts=[
        'scripts/homography_transformer.py',
        'scripts/lane_detector_node.py',
        'scripts/lane_follower_node.py'
    ],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', '*.launch.xml'))),
        ('share/' + package_name + '/media', glob.glob(os.path.join('media', '*'))),
        ('share/' + package_name + '/racetrack_images', glob.glob(os.path.join('racetrack_images', '*', '*.png'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MIT RSS',
    maintainer_email='racecar@mit.edu',
    description='Final challenge package for the RSS racecar',
    license='MIT',
)
