from setuptools import setup

setup(
    name='bev_mapping', version='0.1.0', packages=['bev_mapping'],
    data_files=[('share/ament_index/resource_index/packages', ['resource/bev_mapping']),
                ('share/bev_mapping', ['package.xml'])],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='BEV project maintainers', maintainer_email='maintainer@example.com',
    description='Ouster LiDAR–IMU local mapping adapter.', license='Apache-2.0',
    entry_points={'console_scripts': ['mapping_adapter = bev_mapping.mapping_adapter:main']})
