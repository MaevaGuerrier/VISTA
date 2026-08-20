#!/bin/bash

source /opt/ros/humble/setup.bash
target_dir=/workspace/.packages_vista_ros2
export PYTHONPATH=${target_dir}:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH
export ROS_DOMAIN_ID=126
exec "$@"