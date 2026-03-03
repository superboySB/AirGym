# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from lib.core import torch_ext
from lib.utils import ivecenv 
from lib.utils import env_configurations

class AlgoObserver:
    def __init__(self):
        pass

    def before_init(self, base_name, config, experiment_name):
        pass

    def after_init(self, algo):
        pass

    def process_infos(self, infos, done_indices):
        pass

    def after_steps(self):
        pass

    def after_print_stats(self, frame, epoch_num, total_time):
        pass

class RLGPUAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats."""

    def __init__(self):
        self.tb_log_mode = "compact"
        self.tb_compact = True
        self.log_direct_info = False
        self.episode_info_keys = None
        self.done_reasons = ("goal", "collision", "timeout", "height", "oob", "heading", "unknown")
        self.done_reason_counts = {k: 0 for k in self.done_reasons}
        self.done_total = 0

    def before_init(self, base_name, config, experiment_name):
        self.tb_log_mode = str(config.get("tb_log_mode", "compact")).lower()
        self.tb_compact = self.tb_log_mode != "full"
        self.log_direct_info = config.get("tb_log_direct_info", not self.tb_compact)

        configured_keys = config.get("tb_episode_info_keys", None)
        if configured_keys is None:
            if self.tb_compact:
                self.episode_info_keys = ("reward", "forward_reward", "reach_goal_reward", "terminal_penalty")
            else:
                self.episode_info_keys = None
        else:
            self.episode_info_keys = tuple(configured_keys)

    def after_init(self, algo):
        self.algo = algo
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.ep_infos = []
        self.direct_info = {}
        self.writer = self.algo.writer
        self.done_reason_counts = {k: 0 for k in self.done_reasons}
        self.done_total = 0

    def _extract_flag_tensor(self, infos, key):
        value = infos.get(key, None)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value.to(self.algo.device)
        else:
            tensor = torch.as_tensor(value, device=self.algo.device)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        return tensor.reshape(-1).bool()

    def process_infos(self, infos, done_indices):
        assert isinstance(infos, dict), "RLGPUAlgoObserver expects dict info"
        if isinstance(infos, dict):
            if "item_reward_info" in infos:
                self.ep_infos.append(infos["item_reward_info"])

            if len(infos) > 0 and isinstance(infos, dict) and self.log_direct_info:  # allow direct logging from env
                self.direct_info = {}
                for k, v in infos.items():
                    # only log scalars
                    if (
                        isinstance(v, float)
                        or isinstance(v, int)
                        or (isinstance(v, torch.Tensor) and len(v.shape) == 0)
                    ):
                        self.direct_info[k] = v

            if done_indices.numel() > 0:
                done_idx = done_indices.squeeze(-1).long()
                done_count = int(done_idx.numel())
                self.done_total += done_count

                reason_map = {
                    "goal": "reach_goal",
                    "collision": "collision",
                    "timeout": "time_outs",
                    "height": "done_height",
                    "oob": "done_oob",
                    "heading": "done_heading",
                }
                known = torch.zeros(done_count, device=self.algo.device, dtype=torch.bool)
                for reason_name, info_key in reason_map.items():
                    flags = self._extract_flag_tensor(infos, info_key)
                    if flags is None or flags.numel() == 0:
                        continue
                    # align lengths conservatively if env returned a shorter tensor.
                    max_idx = min(flags.numel(), done_idx.max().item() + 1)
                    valid_mask = done_idx < max_idx
                    selected = torch.zeros(done_count, device=self.algo.device, dtype=torch.bool)
                    if valid_mask.any():
                        selected[valid_mask] = flags[done_idx[valid_mask]]
                    self.done_reason_counts[reason_name] += int(selected.sum().item())
                    known = torch.logical_or(known, selected)

                self.done_reason_counts["unknown"] += done_count - int(known.sum().item())

    def after_clear_stats(self):
        self.mean_scores.clear()
        self.done_reason_counts = {k: 0 for k in self.done_reasons}
        self.done_total = 0

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.ep_infos:
            if self.episode_info_keys is None:
                keys_to_log = tuple(self.ep_infos[0].keys())
            else:
                keys_to_log = tuple(k for k in self.episode_info_keys if k in self.ep_infos[0])
            for key in keys_to_log:
                infotensor = torch.tensor([], device=self.algo.device)
                for ep_info in self.ep_infos:
                    if key not in ep_info:
                        continue
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.algo.device)))
                if infotensor.numel() == 0:
                    continue
                value = torch.mean(infotensor)
                self.writer.add_scalar("episode_info/" + key + "/frame", value, frame)
                if not self.tb_compact:
                    self.writer.add_scalar("episode_info/" + key + "/iter", value, epoch_num)
                    self.writer.add_scalar("episode_info/" + key + "/time", value, total_time)
            self.ep_infos.clear()

        for k, v in self.direct_info.items():
            self.writer.add_scalar(f"env/{k}/frame", v, frame)
            if not self.tb_compact:
                self.writer.add_scalar(f"env/{k}/iter", v, epoch_num)
                self.writer.add_scalar(f"env/{k}/time", v, total_time)

        if self.done_total > 0:
            self.writer.add_scalar("done/episodes/frame", self.done_total, frame)
            for reason_name in self.done_reasons:
                count = self.done_reason_counts[reason_name]
                self.writer.add_scalar(f"done/count_{reason_name}/frame", count, frame)
                self.writer.add_scalar(f"done/rate_{reason_name}/frame", count / float(self.done_total), frame)
            if not self.tb_compact:
                self.writer.add_scalar("done/episodes/iter", self.done_total, epoch_num)
                self.writer.add_scalar("done/episodes/time", self.done_total, total_time)

        if self.mean_scores.current_size > 0:
            mean_scores = self.mean_scores.get_mean()
            self.writer.add_scalar("scores/mean/frame", mean_scores, frame)
            if not self.tb_compact:
                self.writer.add_scalar("scores/mean/iter", mean_scores, epoch_num)
                self.writer.add_scalar("scores/mean/time", mean_scores, total_time)

        self.done_reason_counts = {k: 0 for k in self.done_reasons}
        self.done_total = 0


class RLGPUEnv(ivecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = self.env.action_space
        info["observation_space"] = self.env.observation_space

        if self.env.num_states > 0:
            info["state_space"] = self.env.state_space
            print(info["action_space"], info["observation_space"], info["state_space"])
        else:
            print(info["action_space"], info["observation_space"])

        return info
