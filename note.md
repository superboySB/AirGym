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
本机调试
```bash
python scripts/runner.py --task planning_local --ctl_mode rate --num_envs 512 --headless
# 服务器： python scripts/runner.py --task planning_server --ctl_mode rate --num_envs 2048 --headless
```
- 若算力不足，可把 `--num_envs` 调小（同时可在 `scripts/config/ppo_planning.yaml` 里调 `minibatch_size` 与 `horizon_length`）。  
- 训练产物默认写入 `runs/`。

### 使用预训练策略测试
```bash
python scripts/runner.py --play --task planning_local --ctl_mode rate --num_envs 4 \
  --checkpoint /workspace/AirGym/trained/ppo_planning_vae_30000.pth
```
- 可加 `--headless` 在无图形界面环境中运行。  
- 该 checkpoint 包含深度 CNN 编码器与 rate 控制策略，可直接复现 README 所述“纯深度输入穿越林区”行为。

### 复现/更新深度 VAE
- 新增脚本 `scripts/train_depth_vae.py`，会在线采样 `planning` 环境的深度图并训练 `lib/network/VAE.py` 内定义的编码器结构。
- 默认会在 `runs/vae_planning_*/` 下创建实验目录，其中 `nn/vae_model.pth` 保存最新权重，`summaries/` 存放 TensorBoard 记录。若要自定义目录，可使用 `--train-dir` 或 `--experiment-name`，也可以用 `--output` 指定完整路径。

示例命令（建议参数，约 2 万帧/30 epoch）：
```bash
python scripts/train_depth_vae.py \
  --num-envs 9 \
  --collection-steps 2500 \
  --loop-until-max \
  --max-samples 20000 \
  --epochs 30 \
  --batch-size 512 \
  --latent-dims 64 \
  --headless \
  --sim-device cuda:0 \
  --rl-device cuda:0 \
  --policy-config scripts/config/ppo_planning_local.yaml \
  --policy-checkpoint trained/planning_cnn_rate.pth \
  --policy-random-prob 0.2 \
  --kl-weight 1.0 \
  --kl-warmup-epochs 10 \
  --tb-logdir runs/vae_planning \
  --visualize-interval 5 \
  --visualize-count 6 \
  --max-grad-norm 5.0 \
  --lr 5e-4
```
- `--loop-until-max` 会在到达 `max-samples` 之前持续 rollout；可通过 `--policy-random-prob` 设定一定比例的随机动作以增加样本多样性（设为 0 表示只用策略动作，默认 0.0）。
- 运行时脚本会提示当前是否加载策略；采样得到的深度帧会被规范化到 `(1,120,212)`。采集不足时命令行会提示需要补采样的步数。
- `--kl-warmup-epochs` 用于线性升温 KL 权重，避免潜变量坍缩；如需关闭混合精度可加 `--no-amp`。
- `visualize-interval/count` 会在 TensorBoard 中输出原始与重建深度图，可直观检查模型效果；其它曲线仍提供重建、KL、总 loss。
- 若要替换原模型，只需把 `scripts/config/ppo_planning_*.yaml` 中 `network.vae.model_file` 指向生成的 `nn/vae_model.pth`（或自定义输出路径）。
