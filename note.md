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

python airgym/scripts/example.py --task hovering --ctl_mode pos --num_envs 4
```
镜像在构建阶段已完成 `rlPx4Controller`、AirGym 以及 Isaac Gym 的安装。容器默认工作目录为 `/workspace/AirGym`，开箱即用。

## 2. 容器 / 裸机内的常用命令

```bash
# 训练
python scripts/runner.py --task planning --ctl_mode rate --headless

# 评估预训练模型
python scripts/runner.py --play --task planning --ctl_mode rate \
  --num_envs 4 --headless --checkpoint trained/planning_cnn_rate.pth
```

容器默认工作目录为 `/workspace/AirGym`，镜像内已包含仓库源码。

---

## 3. Planning 任务概览

### 3.1 代码结构

- 任务逻辑：`airgym/envs/task/planning.py`
- 环境配置：`airgym/envs/task/planning_config.py`
- 训练配置：`scripts/config/ppo_planning.yaml`
- 启动入口：`scripts/runner.py` → `lib/torch_runner.py`

### 3.2 传感器与观测

| 内容 | 位置 | 说明 |
| --- | --- | --- |
| 深度相机启用 | `planning_config.py:49-63` | 默认 `enable_onboard_cameras=True`，分辨率 `120×212` |
| 相机挂载 | `airgym/envs/base/customized.py:170-214` | 每个 env 创建深度相机并缓存为 `full_camera_array` |
| 渲染频率 | 同上 | `cam_dt=0.04s`，相当于每 4 个物理步刷新一次 |
| RL 观测 | `planning.py:138-214` | 返回字典：深度图 + 16 维低维状态 |

### 3.3 奖励与 ESDF

- 最近障碍距离由深度图展平取最小值得到：`planning.py:162-164`  
- 奖励组合：`planning.py:223-307`（包括前进、姿态、平滑度、ESDF 等项）  
- ESDF 仅用于奖励塑形，未作为额外观测输入；测试模式仍沿用相同逻辑。

### 3.4 配置要点

| 参数 | 位置 | 默认值 |
| --- | --- | --- |
| 并行环境数 | `ppo_planning.yaml:65` | `4096`，可根据显存调整 |
| 输入编码 | `ppo_planning.yaml:31-39` | 默认 CNN；可改用预训练 VAE（注释中提供模板） |
| Episode 长度 | `planning_config.py:11-23` | 16 秒，10 ms 步长 |
| 控制模式 | CLI `--ctl_mode` + 配置文件 | 默认 `rate`，影响动作维度与 PID 控制器 |

---

## 4. 常见问题

- **深度相机是否在所有 env 中启用？** 是。`enable_onboard_cameras=True` 且 `env_config.use_image=True`。  
- **ESDF 是否为特权信息？** 否。它直接来源于实时深度图，仅在奖励内部使用。  
- **如何切换到预训练 VAE 编码器？** 在 `ppo_planning.yaml` 中注释 `cnn`，启用 `vae` 配置并指向 `trained/vae_model.pth`。

---

借助上述步骤即可在裸机或 Docker 环境中快速复现 AirGym Planning 任务。***
