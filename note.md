# AirGym Simulation Guide

本说明涵盖两部分内容：一是如何使用 Docker 镜像启动 Isaac Gym + AirGym 仿真环境，二是如何在容器或裸机上运行 Planning 任务。

## 1. Docker 工作流
```bash
docker build --network=host --progress=plain \
  -f docker/simulation.dockerfile \
  -t airgym-image:v0 .

xhost +local:root

docker run --name airgym-sim -itd --gpus all --network host --ipc=host --privileged \
  -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
  -v $HOME/.Xauthority:/root/.Xauthority:rw \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $HOME/docker/airgym/cache/pip:/root/.cache/pip:rw \
  -v $HOME/docker/airgym/cache/ov:/root/.cache/ov:rw \
  -v $HOME/docker/airgym/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v $HOME/docker/airgym/cache/computecache:/root/.nv/ComputeCache:rw \
  -v $HOME/docker/airgym/logs:/root/.nvidia-omniverse/logs:rw \
  -v $HOME/docker/airgym/data:/root/.local/share/ov/data:rw \
  airgym-image:v0 /bin/bash

docker exec -it airgym-sim /bin/bash  # 默认进入airgym的workspace

python airgym/scripts/example.py --task hovering --ctl_mode rate --num_envs 4
```
镜像在构建阶段已完成 `rlPx4Controller`、AirGym 以及 Isaac Gym 的安装。容器默认工作目录为 `/workspace/AirGym`，开箱即用。

## 2. Planning: 深度视觉 + Rate Control

- 倉庫里没有 VAE 训练脚本，`trained/vae_model.pth` 仅供推理使用；`cnn` 方案可以端到端与控制策略一起学。  
- Planning 任务默认在每个环境的 `X152b` 机体上挂载深度相机，且 `scripts/config/ppo_planning.yaml` 已设置 `env_config.use_image=True` 与 `network.cnn`，因此直接启用即得到深度图 + 数值状态的联合观测。

### 训练端到端 CNN + Rate Control
```bash
python scripts/runner.py --task planning_local --ctl_mode rate --num_envs 256 --headless
# 服务器： python scripts/runner.py --task planning_server --ctl_mode rate --num_envs 2048 --headless
```
- 若算力不足，可把 `--num_envs` 调小（同时可在 `scripts/config/ppo_planning.yaml` 里调 `minibatch_size` 与 `horizon_length`）。  
- 训练产物默认写入 `runs/`。

### 使用预训练策略测试
```bash
python scripts/runner.py --play --task planning_local --ctl_mode rate --num_envs 4 \
  --checkpoint /workspace/AirGym/runs/ppo_planning_21-07-59-48/nn/ppo_planning.pth
```
- 可加 `--headless` 在无图形界面环境中运行。  
- 该 checkpoint 包含深度 CNN 编码器与 rate 控制策略，可直接复现 README 所述“纯深度输入穿越林区”行为。
