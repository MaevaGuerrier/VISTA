# FROM img_name:tag
FROM dustynv/l4t-pytorch:r36.2.0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -qq -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libssl-dev \
    libusb-1.0-0-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \    
    curl \	
    libusb-1.0-0 \
    udev \
    apt-transport-https \
    ca-certificates \
    curl \
    swig \
    wget \
    tmux \
    software-properties-common \
    apt-utils \
    python3-grpcio \
    python3-scipy \
    psmisc \
    vim-gtk \ 
    libcppunit-dev \
    python3-pip \
    nano

# Add ROS 2 repo key and source
RUN mkdir -p /usr/share/keyrings && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        | gpg --dearmor > /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
    > /etc/apt/sources.list.d/ros2.list

# Install ROS 2 Humble base packages
RUN apt update && apt install -y \
    ros-humble-ros-base \
    ros-humble-ament-cmake \
    ros-humble-rmw-zenoh-cpp \
    ros-humble-cv-bridge \
    ros-humble-nav-msgs \
    python3-colcon-common-extensions \
    python3-argcomplete \
    && rm -rf /var/lib/apt/lists/*



RUN pip install --no-cache-dir --index-url https://pypi.org/simple \
    matplotlib==3.5.1 \
    pyyaml \
    einops \
    vit-pytorch \
    rospkg \
    defusedxml \
    protobuf==3.20 \
    huggingface_hub==0.23.5 \
    efficientnet_pytorch \
    warmup_scheduler \
    diffusers==0.11.1 \
    scipy==1.10.1 \
    prettytable \
    onnx \
    pycuda \
    bagpy \
    coloredlogs \
    flatbuffers \
    wandb \
    transformers \
    sympy

RUN wget https://nvidia.box.com/shared/static/i7n40ki3pl2x57vyn4u7e9asyiqlnl7n.whl \ 
      -O onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl \
    && pip install --no-cache-dir --index-url https://pypi.org/simple  numpy==1.26.3 onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl 
    

ARG CACHE_BUST=1
COPY ./setup.sh /tmp/setup.sh

RUN chmod +x /tmp/setup.sh
RUN echo "source /tmp/setup.sh" >> /root/.bashrc
ENTRYPOINT ["/tmp/setup.sh"]
CMD ["bash","-l"]