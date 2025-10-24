import argparse
import os
from typing import List, Optional

try:
    from isaacgym import gymutil, gymapi  # noqa: F401
except ImportError:
    from airgym.utils.gym_utils import gymutil, gymapi  # noqa: F401

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import yaml
from datetime import datetime

from lib.utils import vecenv
from lib.network.VAE import VAE

from lib.agent.players import A2CPlayer


def parse_args():
    parser = argparse.ArgumentParser(description="Train depth VAE from AirGym planning observations.")
    parser.add_argument("--num-envs", type=int, default=256, help="Number of parallel envs used for data collection.")
    parser.add_argument("--collection-steps", type=int, default=200, help="Steps of policy-driven rollout used for data collection (after any random prefill).")
    parser.add_argument("--max-samples", type=int, default=50000, help="Upper bound on number of depth frames used for training.")
    parser.add_argument("--random-prefill", type=int, default=0, help="Number of depth frames to collect with pure random actions before using policy (counts toward max-samples).")
    parser.add_argument("--latent-dims", type=int, default=64, help="Latent dimensionality of the VAE.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train the VAE.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini batch size for VAE training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--kl-weight", type=float, default=1.0, help="Multiplier for KL-divergence term.")
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=5,
        help="Linearly ramp KL weight from 0 to the target value over this many epochs (0 disables warmup).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--sim-device", type=str, default="cuda:0", help="Isaac Gym simulation device string.")
    parser.add_argument("--rl-device", type=str, default="cuda:0", help="Device used for training the VAE.")
    parser.add_argument("--headless", action="store_true", help="Run environments without rendering.")
    parser.add_argument("--policy-config", type=str, default=None, help="Path to PPO yaml for action policy (optional).")
    parser.add_argument("--policy-checkpoint", type=str, default=None, help="Checkpoint path for action policy (optional).")
    parser.add_argument(
        "--policy-deterministic",
        action="store_true",
        help="Use deterministic actions from policy during data collection.",
    )
    parser.add_argument(
        "--policy-random-prob",
        type=float,
        default=0.0,
        help="Probability of replacing policy actions with random actions (0.0 keeps pure policy).",
    )
    parser.add_argument(
        "--tb-logdir",
        type=str,
        default=None,
        help="Optional TensorBoard log directory for VAE training metrics. Defaults to <train_dir>/<exp>/summaries.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="runs",
        help="Root directory to store VAE experiment artifacts.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name. Defaults to vae_planning_<timestamp>.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit path to save the trained VAE weights. Defaults to <train_dir>/<exp>/nn/vae_model.pth.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=5.0,
        help="Gradient clipping norm for VAE training.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed-precision (automatic casting).",
    )
    parser.add_argument(
        "--loop-until-max",
        action="store_true",
        help="Keep collecting frames until max-samples is reached (ignoring collection-steps limit).",
    )
    parser.add_argument(
        "--visualize-interval",
        type=int,
        default=5,
        help="Epoch interval for logging original vs reconstructed depth images (0 disables).",
    )
    parser.add_argument(
        "--visualize-count",
        type=int,
        default=6,
        help="Number of depth frames to visualize when logging reconstructions.",
    )
    return parser.parse_args()


def create_vec_env(num_envs: int, sim_device: str, headless: bool):
    env_config = {
        "num_envs": num_envs,
        "headless": headless,
        "ctl_mode": "rate",
        "use_gpu": True,
        "use_gpu_pipeline": True,
        "physics_engine": gymapi.SIM_PHYSX,
        "sim_device": sim_device,
        "device": sim_device,
        "subscenes": 0,
        "num_threads": 0,
        "seed": 0,
        "use_image": True,
    }
    return vecenv.create_vec_env("planning", num_envs, **env_config)


def collect_depth_samples(
    env,
    num_steps: int,
    max_samples: int,
    num_envs: int,
    rollout_device: torch.device,
    policy_player: Optional[A2CPlayer],
    deterministic_policy: bool,
    policy_random_prob: float,
    random_prefill: int,
    loop_until_max: bool,
) -> torch.Tensor:
    env_info = env.get_env_info()
    action_space = env_info["action_space"]
    action_low = torch.from_numpy(action_space.low).to(rollout_device)
    action_high = torch.from_numpy(action_space.high).to(rollout_device)
    action_dim = action_low.shape[0]

    collected: List[torch.Tensor] = []

    obs = env.reset()
    if isinstance(obs, dict):
        first_images = obs["image"].detach().to("cpu")
    else:
        raise RuntimeError("Unexpected observation format: expected dict with 'image' key.")
    collected.append(first_images)

    total_needed = max_samples
    steps = 0
    def need_more_samples():
        return sum(t.size(0) for t in collected) < total_needed

    remaining_prefill = max(0, random_prefill)
    while need_more_samples() and remaining_prefill > 0:
        rand = torch.rand((num_envs, action_dim), device=rollout_device)
        actions = action_low.unsqueeze(0) + (action_high - action_low).unsqueeze(0) * rand
        obs, _, _, _ = env.step(actions)
        images = obs["image"].detach().to("cpu")
        collected.append(images)
        remaining_prefill -= num_envs

    while need_more_samples() and (steps < num_steps or loop_until_max):
        steps += 1
        if policy_player is not None:
            with torch.no_grad():
                policy_obs = {
                    "observation": obs["observation"].to(policy_player.device),
                    "image": obs["image"].to(policy_player.device),
                }
                actions = policy_player.get_action(policy_obs, is_deterministic=deterministic_policy)
                actions = actions.to(rollout_device)
                if policy_random_prob > 0.0:
                    rand = torch.rand((num_envs, action_dim), device=rollout_device)
                    random_actions = action_low.unsqueeze(0) + (action_high - action_low).unsqueeze(0) * rand
                    mask = (torch.rand((num_envs, 1), device=rollout_device) < policy_random_prob).float()
                    actions = random_actions * mask + actions * (1.0 - mask)
        else:
            rand = torch.rand((num_envs, action_dim), device=rollout_device)
            actions = action_low.unsqueeze(0) + (action_high - action_low).unsqueeze(0) * rand

        obs, _, _, _ = env.step(actions)
        images = obs["image"].detach().to("cpu")
        collected.append(images)

    frames = torch.cat(collected, dim=0)
    if frames.size(0) > max_samples:
        frames = frames[:max_samples]
    # Convert from (C, W, H) to (C, H, W) expected by the VAE.
    if frames.dim() == 4 and frames.shape[-2:] == (212, 120):
        frames = frames.permute(0, 1, 3, 2).contiguous()
    return frames


def prepare_images_for_logging(tensor: torch.Tensor, make_rgb: bool = True) -> torch.Tensor:
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.clone()
    # stretch each image to full [0,1] range for visualization only
    b, c, h, w = tensor.shape
    tensor_flat = tensor.view(b, -1)
    mins = tensor_flat.min(dim=1, keepdim=True).values
    maxs = tensor_flat.max(dim=1, keepdim=True).values
    denom = (maxs - mins).clamp_min(1e-6)
    tensor = ((tensor_flat - mins) / denom).view_as(tensor)
    if make_rgb and tensor.size(1) == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    return tensor.clamp(0.0, 1.0).detach().cpu()


def train_vae(
    dataset: TensorDataset,
    latent_dims: int,
    epochs: int,
    batch_size: int,
    lr: float,
    kl_weight: float,
    kl_warmup_epochs: int,
    device: torch.device,
    output_path: str,
    writer: Optional[SummaryWriter],
    max_grad_norm: float,
    amp_enabled: bool,
    visualize_interval: int,
    visualize_count: int,
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = VAE(input_dim=1, latent_dim=latent_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = (device.type == "cuda") and amp_enabled
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    global_step = 0
    sample_batch = None
    if visualize_interval > 0 and len(dataset) > 0:
        count = min(visualize_count, len(dataset))
        sample_batch = torch.stack([dataset[i][0] for i in range(count)])
    model.train()
    warned_nonfinite = False
    for epoch in range(epochs):
        epoch_recon = 0.0
        epoch_kld = 0.0
        if kl_warmup_epochs > 0:
            warmup_scale = min(1.0, (epoch + 1) / kl_warmup_epochs)
            effective_kl_weight = kl_weight * warmup_scale
        else:
            effective_kl_weight = kl_weight

        for batch in dataloader:
            inputs = batch[0].to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                recon, mean, logvar, _ = model(inputs)
                recon_loss = F.mse_loss(recon, inputs, reduction="mean")
                kld = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                loss = recon_loss + effective_kl_weight * kld
            if not torch.isfinite(loss):
                if not warned_nonfinite:
                    print("[WARN] Encountered non-finite loss. Affected batches will be skipped.")
                    warned_nonfinite = True
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_recon += recon_loss.item()
            epoch_kld += kld.item()
            if writer is not None:
                writer.add_scalar("train/recon_loss", recon_loss.item(), global_step)
                writer.add_scalar("train/kld", kld.item(), global_step)
                writer.add_scalar("train/total_loss", loss.item(), global_step)
            global_step += 1

        num_batches = len(dataloader)
        if num_batches > 0:
            print(
                f"Epoch {epoch+1}/{epochs} - recon: {epoch_recon/num_batches:.6f}, "
                f"kld: {epoch_kld/num_batches:.6f}, total: {(epoch_recon + epoch_kld)/num_batches:.6f}"
            )
            if writer is not None:
                writer.add_scalar("epoch/recon_loss", epoch_recon / num_batches, epoch)
                writer.add_scalar("epoch/kld", epoch_kld / num_batches, epoch)
                writer.add_scalar("epoch/total_loss", (epoch_recon + epoch_kld) / num_batches, epoch)

            if writer is not None and sample_batch is not None and visualize_interval > 0:
                if (epoch + 1) % visualize_interval == 0 or epoch == epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        inputs = sample_batch.to(device)
                        recon, _, _, _ = model(inputs)
                    writer.add_images("samples/original", inputs.detach().cpu(), epoch + 1)
                    writer.add_images("samples/reconstructed", recon.clamp(0, 1).detach().cpu(), epoch + 1)
                    model.train()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved VAE weights to {output_path}")
    if writer is not None:
        writer.close()


def main():
    args = parse_args()

    if args.seed:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = create_vec_env(args.num_envs, args.sim_device, args.headless)
    env_info = env.get_env_info()
    rollout_device = torch.device(args.sim_device)

    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = "vae_planning" + datetime.now().strftime("_%d-%H-%M-%S")

    experiment_dir = os.path.join(args.train_dir, experiment_name)
    nn_dir = os.path.join(experiment_dir, "nn")
    summaries_dir = os.path.join(experiment_dir, "summaries")
    os.makedirs(nn_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(nn_dir, "vae_model.pth")

    log_dir = args.tb_logdir if args.tb_logdir else summaries_dir

    print(f"[INFO] VAE experiment directory: {experiment_dir}")
    print(f"[INFO] Model artifacts will be saved to: {output_path}")
    print(f"[INFO] TensorBoard logs will be written to: {log_dir}")

    policy_player: Optional[A2CPlayer] = None
    if args.policy_config and args.policy_checkpoint:
        with open(args.policy_config, "r") as stream:
            policy_config = yaml.safe_load(stream)
        policy_params = policy_config["params"]
        policy_params["config"]["env_info"] = env_info
        policy_params["config"]["player"] = policy_params["config"].get("player", {})
        policy_params["config"]["player"]["use_vecenv"] = False
        policy_params["config"]["player"]["deterministic"] = args.policy_deterministic
        policy_params["config"]["device_name"] = args.rl_device
        policy_params["config"]["device"] = args.rl_device
        policy_params["config"]["num_actors"] = args.num_envs
        try:
            ckpt = torch.load(args.policy_checkpoint, map_location="cpu")
            model_state = ckpt.get("model", {})
        except Exception as exc:
            print(f"[WARN] Failed to inspect policy checkpoint: {exc}. Falling back to random actions.")
            model_state = {}

        uses_cnn = any(key.startswith("actor_cnn") for key in model_state.keys())
        if uses_cnn:
            cnn_weight = model_state.get("actor_cnn.fc.weight", None)
            feature_dim = cnn_weight.shape[0] if cnn_weight is not None else 30
            policy_params["network"].pop("vae", None)
            policy_params["network"]["cnn"] = {"output_dim": feature_dim}
            print(f"[INFO] Detected CNN-based policy ({feature_dim} dims). Adjusted network config accordingly.")
        else:
            if "vae" not in policy_params["network"]:
                print("[INFO] Detected VAE-based policy.")

        policy_player = A2CPlayer(policy_params)
        policy_player.is_tensor_obses = True
        policy_player.has_batch_dimension = True
        try:
            policy_player.restore(args.policy_checkpoint)
            policy_player.model.to(torch.device(args.rl_device))
            policy_player.model.eval()
            print(f"[INFO] Loaded policy from {args.policy_checkpoint} for data collection "
                  f"({'deterministic' if args.policy_deterministic else 'stochastic'}) actions.")
        except Exception as exc:
            print(f"[WARN] Failed to load policy checkpoint ({exc}). Reverting to random actions.")
            policy_player = None
    else:
        print("[INFO] No policy supplied. Collecting data with random actions.")

    frames = collect_depth_samples(
        env,
        args.collection_steps,
        args.max_samples,
        args.num_envs,
        rollout_device,
        policy_player,
        args.policy_deterministic,
        args.policy_random_prob,
        args.random_prefill,
        args.loop_until_max,
    )
    if hasattr(env, "close"):
        env.close()

    frames = frames.unsqueeze(1) if frames.dim() == 3 else frames  # ensure (N, C, H, W)
    frames = torch.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
    frames = frames.clamp(0.0, 1.0)
    dataset = TensorDataset(frames.float())
    print(f"[INFO] Collected {len(dataset)} depth frames for VAE training.")
    if len(dataset) < args.max_samples:
        shortfall = args.max_samples - len(dataset)
        approx_steps = (shortfall + args.num_envs - 1) // max(args.num_envs, 1)
        print(f"[WARN] Requested max-samples={args.max_samples}, but only gathered {len(dataset)} frames. "
              f"Consider increasing --collection-steps by ≈{approx_steps}.")

    train_device = torch.device(args.rl_device)
    writer = SummaryWriter(log_dir)
    writer.add_scalar("dataset/min", frames.min().item(), 0)
    writer.add_scalar("dataset/max", frames.max().item(), 0)
    writer.add_scalar("dataset/mean", frames.mean().item(), 0)
    writer.add_histogram("dataset/depth_hist", frames.view(-1), 0)
    train_vae(
        dataset,
        args.latent_dims,
        args.epochs,
        args.batch_size,
        args.lr,
        args.kl_weight,
        args.kl_warmup_epochs,
        train_device,
        output_path,
        writer,
        max_grad_norm=args.max_grad_norm,
        amp_enabled=not args.no_amp,
        visualize_interval=args.visualize_interval,
        visualize_count=args.visualize_count,
    )


if __name__ == "__main__":
    main()
