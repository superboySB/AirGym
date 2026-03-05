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

python scripts/runner.py --task hovering --ctl_mode rate --num_envs 4
```
镜像在构建阶段已完成 `rlPx4Controller` 与 Isaac Gym 安装，不包含 AirGym 项目代码；项目代码通过 `docker run -v ${LOCAL_AIRGYM_DIR}:/workspace/AirGym` 挂载，便于持续开发与管理。首次进入容器后执行一次 `pip install -e /workspace/AirGym` 即可。

## 2. Planning: 深度视觉 + Rate Control

- Planning 任务默认在每个环境的 `X152b` 机体上挂载深度相机，观测包含 `image + observation`。  
- 下面四种方法都分为训练和测试（`--play`）；测试统一用 `--num_envs 1 --seed 0`。  
- `--play` 阶段已加入事件提示：每回合都会打印 `event: done=... goal=... collision=... timeout=... height=... oob=... heading=... unknown=...`，并输出带阈值与实测数值的 `event_reason` 句子（例如目标距离、当前位置 x/y/z、heading_reward 等）；若发生碰撞会额外打印 `event: collision detected`（训练阶段不打印）。

### 方法A：端到端 CNN + Rate Control
训练（从头学视觉+控制）
```bash
python scripts/runner.py --task planning_cnn --ctl_mode rate --num_envs 512 --headless
# 服务器大规模训练可用：--num_envs 4096
```
测试（单环境）
```bash
python -u scripts/runner.py --play --task planning_cnn --ctl_mode rate --num_envs 1 --seed 0 \
  --checkpoint /workspace/AirGym/trained/20260304/ppo_planning_cnn_03-15-34-29/ppo_planning_cnn.pth
```

### 方法B：预训练 VAE + Rate Control
训练（冻结预训练 VAE，仅训练 RL 策略）
```bash
python scripts/runner.py --task planning_vae --ctl_mode rate --num_envs 512 --headless
# 服务器可提高并行环境数，例如 --num_envs 4096
```
测试（单环境）
```bash
python -u scripts/runner.py --play --task planning_vae --ctl_mode rate --num_envs 1 --seed 0 \
  --checkpoint /workspace/AirGym/trained/20260304/ppo_planning_vae_03-15-34-59/ppo_planning_vae.pth
```

### 方法C：预训练 DeFM ViT-S/14 + Rate Control
- 已引入 DeFM ViT-S/14 关键源码并接入现有 Isaac Gym PPO：`lib/network/defm_vit.py`、`lib/network/defm_preprocess.py`、`lib/network/defm_image_encoder.py`。
- 新增任务/配置：`planning_defm`，`scripts/config/ppo_planning_defm.yaml`。
- 下载权重（一次即可）：
```bash
wget -O /workspace/AirGym/trained/defm_vit_s14.pth \
  https://huggingface.co/leggedrobotics/defm/resolve/main/defm_vit_s14.pth
```
训练
```bash
python scripts/runner.py --task planning_defm --ctl_mode rate --num_envs 512 --headless
# 服务器可提高并行环境数，例如 （TODO: CUDA_VISIBLE_DEVICES的设置不起作用！）
# python scripts/runner.py --task planning_defm --ctl_mode rate --num_envs 4096 --headless
```
测试（单环境）
```bash
python -u scripts/runner.py --play --task planning_defm --ctl_mode rate --num_envs 1 --seed 0 \
  --checkpoint /workspace/AirGym/trained/20260304/ppo_planning_defm_03-15-30-48/ppo_planning_defm.pth
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

### 方法D：预训练 DeFM ResNet-18 + Rate Control
- 新增任务/配置：`planning_defm_resnet18`，`scripts/config/ppo_planning_defm_resnet18.yaml`。
- 下载权重（一次即可）：
```bash
wget -O /workspace/AirGym/trained/defm_resnet18.pth \
  https://huggingface.co/leggedrobotics/defm/resolve/main/defm_resnet18.pth
```
训练
```bash
python scripts/runner.py --task planning_defm_resnet18 --ctl_mode rate --num_envs 512 --headless
# 服务器可提高并行环境数，例如 --num_envs 4096
```
测试（单环境）
```bash
python -u scripts/runner.py --play --task planning_defm_resnet18 --ctl_mode rate --num_envs 1 --seed 0 \
  --checkpoint /workspace/AirGym/trained/20260304/ppo_planning_defm_resnet18_04-10-18-15/ppo_planning_defm_resnet18.pth
```
- Loading the Model：`network.defm.model_folder + model_file` 指向 `defm_resnet18.pth`。  
- Preprocessing：与方法 C 相同，先将 AirGym 深度恢复米制并做 DeFM metric-aware 三通道归一化。  
- Inference：按 DeFM ConvNet 用法提取 `global_backbone` 特征，再与数值状态拼接进入 PPO。

## ABCD如何对比以及判定阈值说明

- 公平对比建议固定：同一 `--ctl_mode rate`、同一 `--num_envs 1`、同一 `--seed 0`、同一地图与代码版本、同一统计回合数。  
- 当前 `planning` 默认 `reset_on_collision=True`，碰撞会像超时/越界一样触发回合结束，并在 `--play` 日志给出 `event` 与 `event_reason`。  

`event` 字段与阈值（A/B/C/D一致）：
- `goal=1`：目标距离 `< 0.5 m`。  
- `timeout=1`：`progress_buf >= max_episode_length - 1`。  
- `collision=1`：至少一个机体刚体链接接触力范数 `> 0.1`。  
- `height=1`：高度不在 `[1.2, 1.8] m`（即 `FLY_HEIGHT=1.5` 的 `±0.3`）。  
- `oob=1`：平面越界，要求 `x∈[-8.5, 8.5]` 且 `y∈[-4.0, 4.0]`。  
- `heading=1`：`heading_reward < 0.25`。  
- `unknown=1`：触发了未归类终止条件（理想应接近 0）。

## TensorBoard 指标详解（Planning / compact 模式）

以下说明基于当前 planning 系列配置（`tb_log_mode: compact`、`use_diagnostics: True`）下，训练时实际写入的标量条目。

- 横轴说明：`/frame` 表示按环境步数（更适合看训练趋势）；`info/*`、`losses/*` 没有后缀，但也是按 `frame` 记录。
- 统计窗口说明：`done/*` 是“单次打印统计周期（约一个 epoch）内”的计数与占比，不是全程累计。

### A. PPO 基础损失与训练状态

- `losses/a_loss`：策略损失（actor loss）。  
  一般不要求单调下降，关注是否稳定、是否突然爆炸。

- `losses/c_loss`：价值函数损失（critic loss）。  
  趋势通常应下降并趋于平稳，持续偏大常表示值函数拟合不足。

- `losses/entropy`：策略熵。  
  熵高表示探索强，熵低表示策略更确定；后期通常会下降。

- `losses/bounds_loss`：动作均值边界正则损失。  
  抑制策略输出超出 soft bound（默认 1.1）太多；过大说明动作分布过“冲”。

- `info/kl`：新旧策略 KL。  
  与自适应学习率强相关；过大代表更新过猛，过小代表更新偏保守。

- `info/last_lr`：当前学习率（调度后）。  
  配合 `info/kl` 看学习率调度是否正常。

- `info/lr_mul`：学习率倍率。  
  自适应调度内部系数；接近 1 表示变化不大。

- `info/e_clip`：当前 PPO clip 阈值（含 lr_mul 影响）。  
  反映当前策略更新允许的截断范围。

- `info/epochs`：训练 epoch 计数。

### B. 任务结果与回报

- `episode/reward/frame`：原始回合回报均值。  
  最核心目标指标之一，通常希望整体上升。

- `episode/shaped_reward/frame`：shape 后回报均值。  
  当前配置 `reward_shaper.scale_value=0.1`，且 timeout 场景会加 value bootstrap 项，所以它与原始回报不完全等比例。

- `episode/length/frame`：回合步数均值。  
  在导航任务里，和 `done/rate_*` 联合看更有意义（长回合可能是成功，也可能是超时）。

### C. 与任务约束直接相关（重点看）

- `done/episodes/frame`：该统计周期内结束回合总数（分母）。

- `done/count_goal/frame`：达到目标回合数。
- `done/count_timeout/frame`：超时结束回合数。
- `done/count_oob/frame`：平面越界回合数。
- `done/count_height/frame`：高度越界回合数。
- `done/count_heading/frame`：航向约束失败回合数。
- `done/count_collision/frame`：碰撞结束回合数。
- `done/count_unknown/frame`：未归类结束回合数（应尽量接近 0）。

- `done/rate_goal/frame`：`count_goal / episodes`，应上升。
- `done/rate_timeout/frame`：应按任务目标判断，通常中后期应下降或与目标率此消彼长。
- `done/rate_oob/frame`：应下降（越界减少）。
- `done/rate_height/frame`：应下降（高度违规减少）。
- `done/rate_heading/frame`：应下降（航向违规减少）。
- `done/rate_collision/frame`：应下降（碰撞减少）。
- `done/rate_unknown/frame`：应接近 0。

### D. 奖励分解（你关心“是否在优化约束”就看这组）

- `episode_info/reward/frame`：环境内部总 reward 均值（未经过 reward_shaper）。

- `episode_info/forward_reward/frame`：前进/接近目标奖励分量。  
  一般希望整体抬升。

- `episode_info/reach_goal_reward/frame`：到达目标奖励分量（触发时有大正奖励）。  
  与 `done/rate_goal/frame` 正相关。

- `episode_info/terminal_penalty/frame`：终止惩罚分量（oob/height/heading/collision）。  
  这是负值项，理想趋势是“绝对值变小、向 0 靠近”。

### E. 诊断项（训练稳定性）

- `diagnostics/exp_var`：值函数解释方差（越接近 1 越好）。  
  <0 说明 value 预测比常数基线还差。

- `diagnostics/rms_value/mean`：value 归一化器运行均值。
- `diagnostics/rms_value/var`：value 归一化器运行方差。  
  用于观察值函数尺度漂移是否异常。

- `diagnostics/clip_frac/0`
- `diagnostics/clip_frac/1`
- `diagnostics/clip_frac/2`
- `diagnostics/clip_frac/3`
- `diagnostics/clip_frac/4`  
  对应每个 mini-epoch（当前 `mini_epochs=5`）的 PPO clip 比例。  
  长期过高说明更新太激进，长期过低说明 clip 几乎不起作用。

### F. 性能项（吞吐，不是策略质量）

- `performance/total_fps`：采样+推理+更新总体 FPS。
- `performance/policy_fps`：采样+推理阶段 FPS（不含更新）。
- `performance/rl_update_time`：单次更新耗时。

这些用于排查训练速度，不用于判断策略是否学好。

### 建议的最小关注面板（10 条）

1. `episode/reward/frame`
2. `episode/length/frame`
3. `done/rate_goal/frame`
4. `done/rate_oob/frame`
5. `done/rate_height/frame`
6. `done/rate_heading/frame`
7. `episode_info/terminal_penalty/frame`
8. `losses/a_loss`
9. `losses/c_loss`
10. `info/kl`

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
  --policy-config scripts/config/ppo_planning_vae.yaml \
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

## 3. 服务器脚本（sb-RL-172）（注意这个环境没办法指定用什么卡，需要切割devices在docker run）
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
