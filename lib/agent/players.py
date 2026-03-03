import gym
import numpy as np
import torch
import copy
from os.path import basename
from typing import Optional
import os
import shutil
import threading
import time

from lib.utils import vecenv
from lib.utils import env_configurations
from lib.core import torch_ext
from lib.utils.tr_helpers import unsqueeze_obs

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action


class BasePlayer(object):

    def __init__(self, params):
        self.config = config = params['config']
        self.env_name = self.config['env_name']
        self.player_config = self.config.get('player', {})
        self.env_config = self.config.get('env_config', {})
        self.env_config = self.player_config.get('env_config', self.env_config)
        self.env_info = self.config.get('env_info')
        self.clip_actions = config.get('clip_actions', True)
        self.seed = self.env_config.pop('seed', None)

        use_vecenv = self.player_config.get('use_vecenv', False)
        if use_vecenv:
            print('[BasePlayer] Creating vecenv: ', self.env_name)
            self.env = vecenv.create_vec_env(
                self.env_name, self.config['num_actors'], **self.env_config)
            self.env_info = self.env.get_env_info()


        self.num_agents = self.env_info.get('agents', 1)
        self.value_size = self.env_info.get('value_size', 1)
        self.action_space = self.env_info['action_space']

        self.observation_space = self.env_info['observation_space']
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape
        self.is_tensor_obses = False

        self.states = None
        self.player_config = self.config.get('player', {})
        self.use_cuda = True
        self.batch_size = 1
        self.has_batch_dimension = False
        self.device_name = self.config.get('device_name', 'cuda')
        self.render_env = self.player_config.get('render', False)
        self.games_num = self.player_config.get('games_num', 2000)

        if 'deterministic' in self.player_config:
            self.is_deterministic = self.player_config['deterministic']
        else:
            self.is_deterministic = self.player_config.get(
                'deterministic', True)

        self.n_game_life = self.player_config.get('n_game_life', 1)
        self.print_stats = self.player_config.get('print_stats', True)
        self.render_sleep = self.player_config.get('render_sleep', 0.002)
        self.max_steps = 108000 // 4
        self.device = torch.device(self.device_name)

    def wait_for_checkpoint(self):
        attempt = 0
        while True:
            attempt += 1
            with self.checkpoint_mutex:
                if self.checkpoint_to_load is not None:
                    if attempt % 10 == 0:
                        print(f"Evaluation: waiting for new checkpoint in {self.dir_to_monitor}...")
                    break
            time.sleep(1.0)

        print(f"Checkpoint {self.checkpoint_to_load} is available!")

    def process_new_eval_checkpoint(self, path):
        with self.checkpoint_mutex:
            # print(f"New checkpoint {path} available for evaluation")
            # copy file to eval_checkpoints dir using shutil
            # since we're running the evaluation worker in a separate process,
            # there is a chance that the file is changed/corrupted while we're copying it
            # not sure what we can do about this. In practice it never happened so far though
            try:
                eval_checkpoint_path = os.path.join(self.eval_checkpoint_dir, basename(path))
                shutil.copyfile(path, eval_checkpoint_path)
            except Exception as e:
                print(f"Failed to copy {path} to {eval_checkpoint_path}: {e}")
                return

            self.checkpoint_to_load = eval_checkpoint_path

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)
        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_torch(obs), rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def obs_to_torch(self, obs):
        if isinstance(obs, dict):
            if 'obs' in obs:
                obs = obs['obs']
            if isinstance(obs, dict):
                upd_obs = {}
                for key, value in obs.items():
                    upd_obs[key] = self._obs_to_tensors_internal(value, False)
            else:
                upd_obs = self.cast_obs(obs)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def _obs_to_tensors_internal(self, obs, cast_to_dict=True):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value, False)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert (obs.dtype != np.int8)
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        elif np.isscalar(obs):
            obs = torch.FloatTensor([obs]).to(self.device)
        return obs

    def preprocess_actions(self, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        return actions

    def env_reset(self, env):
        obs = env.reset()
        return self.obs_to_torch(obs)

    def restore(self, fn):
        raise NotImplementedError('restore')

    def get_weights(self):
        weights = {}
        weights['model'] = self.model.state_dict()
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])

    def create_env(self):
        return env_configurations.configurations[self.env_name]['env_creator'](**self.env_config)

    def get_action(self, obs, is_deterministic=False):
        raise NotImplementedError('step')

    def get_masked_action(self, obs, mask, is_deterministic=False):
        raise NotImplementedError('step')

    def reset(self):
        raise NotImplementedError('raise')

    def _extract_info_flag(self, info, key, batch_size):
        if not isinstance(info, dict):
            return torch.zeros(batch_size, dtype=torch.bool)

        value = info.get(key, None)
        if value is None:
            return torch.zeros(batch_size, dtype=torch.bool)

        if isinstance(value, torch.Tensor):
            flag = value.detach().cpu().reshape(-1).bool()
        elif isinstance(value, np.ndarray):
            flag = torch.from_numpy(value).reshape(-1).bool()
        elif isinstance(value, (list, tuple)):
            flag = torch.as_tensor(value).reshape(-1).bool()
        else:
            flag = torch.full((batch_size,), bool(value), dtype=torch.bool)

        if flag.numel() == 0:
            return torch.zeros(batch_size, dtype=torch.bool)
        if flag.numel() == 1 and batch_size > 1:
            flag = flag.repeat(batch_size)
        if flag.numel() < batch_size:
            pad = torch.zeros(batch_size - flag.numel(), dtype=torch.bool)
            flag = torch.cat((flag, pad), dim=0)
        if flag.numel() > batch_size:
            flag = flag[:batch_size]
        return flag

    def _extract_info_float(self, info, key, batch_size):
        if not isinstance(info, dict):
            return torch.full((batch_size,), float('nan'), dtype=torch.float32)

        value = info.get(key, None)
        if value is None:
            return torch.full((batch_size,), float('nan'), dtype=torch.float32)

        if isinstance(value, torch.Tensor):
            out = value.detach().cpu().reshape(-1).to(dtype=torch.float32)
        elif isinstance(value, np.ndarray):
            out = torch.from_numpy(value).reshape(-1).to(dtype=torch.float32)
        elif isinstance(value, (list, tuple)):
            out = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        else:
            out = torch.full((batch_size,), float(value), dtype=torch.float32)

        if out.numel() == 0:
            return torch.full((batch_size,), float('nan'), dtype=torch.float32)
        if out.numel() == 1 and batch_size > 1:
            out = out.repeat(batch_size)
        if out.numel() < batch_size:
            pad = torch.full((batch_size - out.numel(),), float('nan'), dtype=torch.float32)
            out = torch.cat((out, pad), dim=0)
        if out.numel() > batch_size:
            out = out[:batch_size]
        return out

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)
            print_game_res = False

            for n in range(self.max_steps):
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                obses, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1
                step_collision = self._extract_info_flag(info, 'collision', batch_size)
                step_goal = self._extract_info_flag(info, 'reach_goal', batch_size)
                step_timeout = self._extract_info_flag(info, 'time_outs', batch_size)
                step_height = self._extract_info_flag(info, 'done_height', batch_size)
                step_oob = self._extract_info_flag(info, 'done_oob', batch_size)
                step_heading = self._extract_info_flag(info, 'done_heading', batch_size)
                step_goal_dist = self._extract_info_float(info, 'goal_dist_m', batch_size)
                step_root_x = self._extract_info_float(info, 'root_x_m', batch_size)
                step_root_y = self._extract_info_float(info, 'root_y_m', batch_size)
                step_root_z = self._extract_info_float(info, 'root_z_m', batch_size)
                step_heading_reward = self._extract_info_float(info, 'heading_reward', batch_size)

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()
                    done_collision = int(step_collision[done_indices].sum().item())
                    done_goal = int(step_goal[done_indices].sum().item())
                    done_timeout = int(step_timeout[done_indices].sum().item())
                    done_height = int(step_height[done_indices].sum().item())
                    done_oob = int(step_oob[done_indices].sum().item())
                    done_heading = int(step_heading[done_indices].sum().item())
                    done_known = torch.logical_or(
                        torch.logical_or(
                            torch.logical_or(step_collision, step_goal),
                            torch.logical_or(step_timeout, step_height),
                        ),
                        torch.logical_or(step_oob, step_heading),
                    )
                    done_unknown = done_count - int(done_known[done_indices].sum().item())

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        print(
                            f'event: done={done_count} goal={done_goal} collision={done_collision} '
                            f'timeout={done_timeout} height={done_height} oob={done_oob} '
                            f'heading={done_heading} unknown={done_unknown}'
                        )
                        if done_count == 1:
                            done_env_idx = int(done_indices[0].item())
                            goal_dist = float(step_goal_dist[done_env_idx].item())
                            root_x = float(step_root_x[done_env_idx].item())
                            root_y = float(step_root_y[done_env_idx].item())
                            root_z = float(step_root_z[done_env_idx].item())
                            heading_reward_val = float(step_heading_reward[done_env_idx].item())

                            def fmt(v):
                                return "N/A" if not np.isfinite(v) else f"{v:.6f}"

                            reason_names = []
                            if done_goal > 0:
                                reason_names.append('goal')
                                print(f"event_reason: 达到目标，目标距离={fmt(goal_dist)} m，小于 0.5 m。")
                            if done_collision > 0:
                                reason_names.append('collision')
                                print("event_reason: 发生碰撞，至少一个无人机刚体链接的接触力范数 > 0.1。")
                            if done_timeout > 0:
                                reason_names.append('timeout')
                                print("event_reason: 超时结束，达到最大回合步数上限。")
                            if done_height > 0:
                                reason_names.append('height')
                                print(
                                    f"event_reason: 高度越界，当前 z={fmt(root_z)} m，"
                                    "安全范围是 [1.2, 1.8] m。"
                                )
                            if done_oob > 0:
                                reason_names.append('oob')
                                print(
                                    f"event_reason: 平面越界，当前 x={fmt(root_x)} m, y={fmt(root_y)} m，"
                                    "应满足 x∈[-8.5, 8.5], y∈[-4.0, 4.0]。"
                                )
                            if done_heading > 0:
                                reason_names.append('heading')
                                print(
                                    f"event_reason: 航向约束不满足，heading_reward={fmt(heading_reward_val)}，"
                                    "阈值是 0.25。"
                                )
                            if done_unknown > 0 or len(reason_names) == 0:
                                reason_names.append('unknown')
                                print("event_reason: 触发了未标注的结束条件（unknown）。")
                        if done_collision > 0:
                            print('event: collision detected')
                        cur_rewards_done = cur_rewards/done_count
                        cur_steps_done = cur_steps/done_count
                        if print_game_res:
                            print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f} w: {game_res}')
                        else:
                            print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f}')

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)

    def get_batch_size(self, obses, batch_size):
        obs_shape = self.obs_shape
        if type(self.obs_shape) is dict:
            if 'obs' in obses:
                obses = obses['obs']
            keys_view = self.obs_shape.keys()
            keys_iterator = iter(keys_view)
            if 'observation' in obses:
                first_key = 'observation'
            else:
                first_key = next(keys_iterator)
            obs_shape = self.obs_shape[first_key]
            obses = obses[first_key]

        if len(obses.size()) > len(obs_shape):
            batch_size = obses.size()[0]
            self.has_batch_dimension = True

        self.batch_size = batch_size

        return batch_size


class A2CPlayer(BasePlayer):

    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        from lib.model.a2c_continuous_logstd_model import ModelA2CContinuousLogStd
        keys = {
            'actions_num' : self.actions_num,
            'input_shape' : self.obs_shape,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.model = ModelA2CContinuousLogStd(params, keys)
        self.model.to(self.device)
        self.model.eval()

    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())
        
        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    # def restore(self, fn):
    #     checkpoint = torch_ext.load_checkpoint(fn)
    #     print(checkpoint['model'].keys())
    #     # print(self.model)
    #     self.model.load_state_dict(checkpoint['model'])
    #     if self.normalize_input and 'running_mean_std' in checkpoint:
    #         self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    #     env_state = checkpoint.get('env_state', None)
    #     if self.env is not None and env_state is not None:
    #         self.env.set_env_state(env_state)

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint)

    def set_full_state_weights(self, checkpoint):
        weights = checkpoint
        print(f"Checkpoint model tensors: {len(weights['model'])}")
        try:
            self.model.load_state_dict(weights['model'])
        except:
            """
            Load pretrained mlp.
            ['logstd', 
            'value_mean_std.running_mean', 
            'value_mean_std.running_var', 
            'value_mean_std.count', 
            'running_mean_std.running_mean', 
            'running_mean_std.running_var', 
            'running_mean_std.count', 
            'actor_mlp.layers.0.weight', 
            'actor_mlp.layers.0.bias', 
            'actor_mlp.layers.1.weight', 
            'actor_mlp.layers.1.bias', 
            'actor_mlp.layers.2.weight', 
            'actor_mlp.layers.2.bias', 
            'mu.weight', 
            'mu.bias', 
            'value_head.weight', 
            'value_head.bias']
            """
            print("Missing CNN part. Loading Pretrained MLP Model......")
            with torch.no_grad():
                self.model.logstd.copy_(weights['model']['logstd'])

            running_mean_std_state_dict = {'running_mean': weights['model']['running_mean_std.running_mean'], 
                                           'running_var': weights['model']['running_mean_std.running_var'], 
                                           'count': weights['model']['running_mean_std.count']}
            self.model.running_mean_std.running_mean_std["observation"].load_state_dict(running_mean_std_state_dict)

            value_mean_std_state_dict = {'running_mean': weights['model']['value_mean_std.running_mean'], 
                                         'running_var': weights['model']['value_mean_std.running_var'], 
                                         'count': weights['model']['value_mean_std.count']}
            self.model.value_mean_std.load_state_dict(value_mean_std_state_dict)

            mlp_keys = [key for key in weights['model'].keys() if 'actor_mlp' in key]
            mlp_state_dict = {key: weights['model'][key] for key in mlp_keys}
            self.model.actor_mlp.load_state_dict(mlp_state_dict, strict=False)

            mu_state_dict = {'weight': weights['model']['mu.weight'], 'bias': weights['model']['mu.bias']}
            self.model.mu.load_state_dict(mu_state_dict)
            
            value_state_dict = {'weight': weights['model']['value_head.weight'], 'bias': weights['model']['value_head.bias']}
            self.model.value_head.load_state_dict(value_state_dict)

            

    def reset(self):
        pass
