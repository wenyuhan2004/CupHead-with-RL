"""最小可跑的 Cuphead DQN 训练脚本，参考 HK 结构但保留现有环境/线程。"""
import time
import random
from collections import deque

import numpy as np
import torch
from cuphead_env import CupheadEnv
from .model import ConvDQN
from .memory import ReplayBuffer
from .agent import DQNAgent
from .actions import discrete_to_multibinary, n_actions


def preprocess_obs(obs: np.ndarray) -> torch.Tensor:
    # obs shape: (stack, H, W), 转成 (B, C, H, W)
    return torch.from_numpy(obs).unsqueeze(0)


def make_env():
    return CupheadEnv(
        decision_fps=10,
        frame_size=(128, 72),
        stack=4,
        auto_restart=True,
        debug=False,
        hp_every_n=1,
        parry_every_n=9999,
        x_every_n=0,
    )


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env()
    obs, _ = env.reset()
    obs_tensor = preprocess_obs(obs).float()

    q_net = ConvDQN(in_channels=obs_tensor.shape[1], n_actions=n_actions(), conv_input_shape=obs_tensor.shape[1:])
    target_net = ConvDQN(in_channels=obs_tensor.shape[1], n_actions=n_actions(), conv_input_shape=obs_tensor.shape[1:])
    agent = DQNAgent(q_net, target_net, lr=1e-4, gamma=0.99, device=device)
    replay = ReplayBuffer(capacity=50_000)

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 1e-4
    batch_size = 32
    target_update = 1000
    global_step = 0
    episode_rewards = []
    ep_reward = 0
    obs_buffer = deque(maxlen=1)
    obs_buffer.append(obs)

    try:
        while True:
            epsilon = max(epsilon_min, epsilon - epsilon_decay)
            obs_tensor = preprocess_obs(obs).float().to(device)
            action_idx = agent.select_action(obs_tensor, epsilon, n_actions())
            action = discrete_to_multibinary(action_idx)
            next_obs, reward, done, _, info = env.step(action)
            ep_reward += reward

            replay.push(obs, action_idx, reward, next_obs, float(done))
            obs = next_obs
            global_step += 1

            if len(replay) > 1000:
                batch = replay.sample(batch_size)
                loss = agent.learn(batch, n_actions())
                if global_step % target_update == 0:
                    agent.update_target()

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0
                obs, _ = env.reset()

            if global_step % 500 == 0:
                avg_r = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
                print(f"step={global_step}, eps={epsilon:.2f}, avgR10={avg_r:.2f}")

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        env.close()


if __name__ == "__main__":
    train()
