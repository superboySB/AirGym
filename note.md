# AirGym Simulation Guide

本说明涵盖两部分内容：一是如何使用 Docker 镜像启动 Isaac Gym + AirGym 仿真环境，二是如何在容器或裸机上运行 Planning 任务。

## 1. Docker 工作流
```bash
# 在本地 AirGym 项目根目录执行（不再把项目代码 COPY 进镜像）
docker build --network=host --progress=plain \
  -f docker/simulation.dockerfile \
  -t airgym-image:v0 .

xhost +local:root

LOCAL_AIRGYM_DIR=$(pwd)

docker run --name airgym-sim -itd --gpus all --network host --ipc=host --privileged \
  -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
  -v $HOME/.Xauthority:/root/.Xauthority:rw \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ${LOCAL_AIRGYM_DIR}:/workspace/AirGym:rw \
  -v $HOME/docker/airgym/cache/pip:/root/.cache/pip:rw \
  -v $HOME/docker/airgym/cache/ov:/root/.cache/ov:rw \
  -v $HOME/docker/airgym/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v $HOME/docker/airgym/cache/computecache:/root/.nv/ComputeCache:rw \
  -v $HOME/docker/airgym/logs:/root/.nvidia-omniverse/logs:rw \
  -v $HOME/docker/airgym/data:/root/.local/share/ov/data:rw \
  airgym-image:v0 /bin/bash

docker exec -it airgym-sim /bin/bash
cd /workspace/AirGym
pip install -e /workspace/AirGym

python scripts/example.py --task hovering --ctl_mode rate --num_envs 4
```
镜像在构建阶段已完成 `rlPx4Controller` 与 Isaac Gym 安装，不包含 AirGym 项目代码；项目代码通过 `docker run -v ${LOCAL_AIRGYM_DIR}:/workspace/AirGym` 挂载，便于持续开发与管理。首次进入容器后执行一次 `pip install -e /workspace/AirGym` 即可。

## 2. Planning: 深度视觉 + Rate Control

- Planning 任务默认在每个环境的 `X152b` 机体上挂载深度相机，观测包含 `image + observation`。  
- 下面三种方法都分为训练和测试（`--play`）；测试统一用 `--num_envs 1`。  
- `--play` 阶段已加入事件提示：每回合都会打印 `event: done=... goal=... collision=... timeout=... height=... oob=... heading=... unknown=...`，并输出带阈值与实测数值的 `event_reason` 句子（例如目标距离、当前位置 x/y/z、heading_reward 等）；若发生碰撞会额外打印 `event: collision detected`（训练阶段不打印）。

### 方法A：端到端 CNN + Rate Control
训练（从头学视觉+控制）
```bash
python scripts/runner.py --task planning_server --ctl_mode rate --num_envs 256 --headless
# 服务器大规模训练可用：--num_envs 2048
```
测试（单环境）
```bash
python -u scripts/runner.py --play --task planning_server --ctl_mode rate --num_envs 1 \
  --checkpoint /workspace/AirGym/trained/planning_cnn_rate.pth
```

### 方法B：预训练 VAE + Rate Control
训练（冻结预训练 VAE，仅训练 RL 策略）
```bash
python scripts/runner.py --task planning_local --ctl_mode rate --num_envs 256 --headless
# 服务器可提高并行环境数，例如 --num_envs 512
```
测试（单环境）
```bash
python -u scripts/runner.py --play --task planning_local --ctl_mode rate --num_envs 1 \
  --checkpoint /workspace/AirGym/trained/20251022/ppo_planning_vae_30000.pth
```

### 方法C：预训练 DeFM ViT-S/14 + Rate Control
- 已引入 DeFM ViT-S/14 关键源码并接入现有 Isaac Gym PPO：`lib/network/defm_vit.py`、`lib/network/defm_preprocess.py`、`lib/network/defm_image_encoder.py`。
- 新增任务/配置：`planning_vit_local`、`planning_vit_server`，`scripts/config/ppo_planning_vit_local.yaml`、`scripts/config/ppo_planning_vit_server.yaml`。
- 下载权重（一次即可）：
```bash
wget -O /workspace/AirGym/trained/defm_vit_s14.pth \
  https://huggingface.co/leggedrobotics/defm/resolve/main/defm_vit_s14.pth
```
训练
```bash
python scripts/runner.py --task planning_vit_local --ctl_mode rate --num_envs 256 --headless
# 服务器：python scripts/runner.py --task planning_vit_server --ctl_mode rate --num_envs 2048 --headless
```
测试（单环境）
```bash
python -u scripts/runner.py --play --task planning_vit_local --ctl_mode rate --num_envs 1 \
  --checkpoint /workspace/AirGym/runs/ppo_planning_vit_03-06-58-55/nn/ppo_planning_vit.pth
```
- Loading the Model：`network.defm.model_folder + model_file` 指向 `defm_vit_s14.pth`。  
- Preprocessing：AirGym 深度 `(N,1,120,212)` 会恢复为米制深度（`depth_scale_m=4.5`），再做 DeFM 三通道 metric-aware 归一化。  
- Inference：调用 `get_intermediate_layers(..., reshape=True, return_class_token=True)` 提取中间层 token，再与数值状态拼接进入 PPO。
- 已提供本地兼容 `defm` 包（源码在 `/workspace/AirGym/defm`），不需要额外 `pip install defm`：
```python
import torch
from defm import preprocess_depth_image, defm_vit_s14

normalized_depth = preprocess_depth_image(metric_depth, target_size=518, patch_size=14)
model = defm_vit_s14(pretrained=True, pretrained_path="/workspace/AirGym/trained/defm_vit_s14.pth").eval().to("cuda")

with torch.no_grad():
    output = model.get_intermediate_layers(
        normalized_depth.to("cuda"), n=1, reshape=True, return_class_token=True
    )

spatial_tokens = output[0][0]
class_token = output[0][1]
```

## experimental

### 复现/更新深度 VAE
- 新增脚本 `scripts/train_depth_vae.py`，会在线采样 `planning` 环境的深度图并训练 `lib/network/VAE.py` 内定义的编码器结构。
- 默认会在 `runs/vae_planning_*/` 下创建实验目录，其中 `nn/vae_model.pth` 保存最新权重，`summaries/` 存放 TensorBoard 记录。若要自定义目录，可使用 `--train-dir` 或 `--experiment-name`，也可以用 `--output` 指定完整路径。

示例命令（建议参数，约 2 万帧/30 epoch）：
```bash
python scripts/train_depth_vae.py \
  --num-envs 9 \
  --collection-steps 2500 \
  --random-prefill 20000 \
  --loop-until-max \
  --max-samples 60000 \
  --epochs 80 \
  --batch-size 768 \
  --latent-dims 64 \
  --headless \
  --sim-device cuda:0 \
  --rl-device cuda:0 \
  --policy-config scripts/config/ppo_planning_local.yaml \
  --policy-checkpoint trained/planning_cnn_rate.pth \
  --policy-random-prob 0.4 \
  --kl-weight 5.0 \
  --kl-warmup-epochs 5 \
  --visualize-interval 5 \
  --visualize-count 6 \
  --max-grad-norm 5.0 \
  --lr 3e-4
```
- `--random-prefill` 会先生成一批完全随机的深度帧，配合 `--loop-until-max` 可确保收集到设定数量；`--policy-random-prob` 再在策略阶段引入随机动作提升多样性。
- 运行时脚本会提示当前是否加载策略；采样得到的深度帧会被规范化到 `(1,120,212)`（并自动去除 NaN/Inf）。采集不足时命令行会提示需要补采样的步数，同时 `dataset/*` 指标会被写入 TensorBoard 以核对深度分布。
- `--kl-warmup-epochs` 用于线性升温 KL 权重，避免潜变量坍缩；如需关闭混合精度可加 `--no-amp`。训练日志还会输出 `latent_mean_abs`、`latent_var` 便于监控潜变量是否退化。
- `visualize-interval/count` 会在 TensorBoard 的 Images 标签页生成彩色的原始与重建深度图（单通道复制到 3 通道），方便检查重建质量。
- 若要替换原模型，只需把 `scripts/config/ppo_planning_*.yaml` 中 `network.vae.model_file` 指向生成的 `nn/vae_model.pth`（或自定义输出路径）。

## 3. 服务器脚本（sb-RL-172）
容器与镜像命名规则（以 `sb-RL-172`、`20260121` 为例）：

- 容器名：`airgym-sim-20260121`
- 镜像名：`airgym-image:train-server-sb-rl-172-20260121`
- 远端项目目录：`/data/nvme_data/dzp_is_sb/AirGym-20260121`

脚本说明：
1. `tools/clean_server.sh sb-RL-172 20260121`：清理远端对应日期的容器与镜像（建议先执行）。
2. `tools/deploy_on_server.sh sb-RL-172 20260121 device=all | device=6,7`：若远端已存在目标镜像（如 `airgym-image:train-server-sb-rl-172-20260121`），脚本会复用镜像，仅重新打包并上传最新源代码后直接启动容器；若远端镜像不存在，则按原流程在本地容器 `airgym-sim` 上提交镜像并上传。项目会解压到 `/data/nvme_data/dzp_is_sb/AirGym-20260121`，容器名 `airgym-sim-20260121`。默认项目路径 `/home/dzp/projects/AirGym`，可用 `LOCAL_PROJECT_PATH` 覆盖。可选第三个参数用于传给 `docker run --gpus`（默认 `all`）。
3. `tools/download_from_server.sh sb-RL-172 20260121`：把 `/data/nvme_data/dzp_is_sb/AirGym-20260121/runs` 同步到本地 `runs-sb-RL-172-20260121`。

推荐流程（本地执行）：
```bash
cd /workspace/AirGym

# 1) 清理远端历史同名容器/镜像
tools/clean_server.sh sb-RL-172 20260121

# 2) 部署（默认 all 卡，也可以 device=6,7）
tools/deploy_on_server.sh sb-RL-172 20260121 device=all

# 3) 进入远端容器（在本机通过 ssh 连接）
ssh sb-RL-172
docker exec -it airgym-sim-20260121 /bin/bash

# 4) 训练完成后下载 runs
exit
tools/download_from_server.sh sb-RL-172 20260121
```

可选环境变量（部署时覆盖默认路径）：
```bash
LOCAL_PROJECT_PATH=/home/dzp/projects/AirGym \
LOCAL_TMP_DIR=/home/dzp/Public \
REMOTE_BASE_DIR=/data/nvme_data/dzp_is_sb \
REMOTE_AIRGYM_CACHE=/data/nvme_data/dzp_is_sb/docker/airgym \
tools/deploy_on_server.sh sb-RL-172 20260121 device=6,7
```
