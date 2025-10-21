FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
    http_proxy=http://127.0.0.1:8889 \
    https_proxy=http://127.0.0.1:8889 \
    HTTP_PROXY=http://127.0.0.1:8889 \
    HTTPS_PROXY=http://127.0.0.1:8889 \
    no_proxy=localhost,127.0.0.1,::1 \
    NO_PROXY=localhost,127.0.0.1,::1

SHELL ["/bin/bash", "-lc"]

# Base system packages (Steps 1,5)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip \
    git git-lfs wget curl ca-certificates gedit \
    build-essential libeigen3-dev unzip tmux \
    libgl1 libglvnd0 libglx0 libglu1-mesa libglew-dev python3-tk \
    libegl1 libxext6 libx11-6 libxrender1 libxrandr2 libxi6 libxmu6 libxxf86vm1

# Align python binaries
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip

# Install ROS Noetic (rospy dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    lsb-release gnupg && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && \
    apt-get install -y --no-install-recommends ros-noetic-desktop-full

ENV ROS_DISTRO=noetic \
    ROS_ROOT=/opt/ros/${ROS_DISTRO} \
    ROS_PYTHON_VERSION=3 \
    PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:${PYTHONPATH}

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# PyTorch & friends (Step 4)
RUN pip install \
    numpy==1.20.0 \
    pillow==9.0.0

RUN pip install \
    networkx==2.8.4 tensorboard

RUN pip install \
    torch==2.0.0 \
    torchvision==0.15.0 \
    torchaudio==2.0.0 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install pytorch3d

# NVIDIA ICD overrides (from official docker)
RUN rm -f /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json || true
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json

WORKDIR /workspace

# rlPx4Controller (Step 5)
RUN git clone https://github.com/emNavi/rlPx4Controller.git
RUN pip install pybind11 && \
    pip install -e /workspace/rlPx4Controller

# AirGym (Step 6)
COPY . /workspace/AirGym
RUN pip install usd-core rospkg matplotlib opencv-python tensorboardX && \
    pip install -e /workspace/AirGym

# Isaac Gym (Step 7)
ARG ISAACGYM_DOWNLOAD_URL=https://developer.nvidia.com/isaac-gym-preview-4
RUN mkdir -p /opt/isaacgym && \
    wget --progress=bar:force -O /opt/isaacgym_preview4.tar.gz "${ISAACGYM_DOWNLOAD_URL}" && \
    tar -xzf /opt/isaacgym_preview4.tar.gz -C /opt && \
    rm /opt/isaacgym_preview4.tar.gz && \
    ISAAC_DIR=$(find /opt -maxdepth 1 -mindepth 1 -type d -name "isaacgym*" -o -name "IsaacGym*" | head -n1) && \
    pip install -e "${ISAAC_DIR}/python" && \
    ln -s "${ISAAC_DIR}" /opt/isaacgym

ENV ISAACGYM_ROOT_DIR=/opt/isaacgym \
    PYTHONPATH=/opt/isaacgym/python:${PYTHONPATH}

WORKDIR /workspace/AirGym

# TODO： 减小本地化size
# RUN apt -y autoremove && apt clean autoclean && \
#     rm -rf /var/lib/apt/lists/*
# TODO： 如果走了代理、但是想镜像本地化到其它机器，记得清空代理（或者容器内unset）
# ENV http_proxy=
# ENV https_proxy=
# ENV no_proxy=

CMD ["/bin/bash"]
