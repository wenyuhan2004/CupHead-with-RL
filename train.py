"""最小可跑的 Cuphead DQN 训练脚本，参考 HK 结构但保留现有环境/线程。"""
import os
import sys
import time
import random
from collections import deque

import numpy as np
import torch
torch.backends.cudnn.benchmark = True  # 让 cuDNN 自选最快算法
import os

# 兼容直接调用 `python train.py`（无 -m），手动把包根目录加入 sys.path
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from cuphead_env import CupheadEnv
    from model import Conv3DDQN
    from memory import ReplayBuffer
    from agent import DQNAgent
    from actions import discrete_to_multibinary, n_actions
    from cuphead_memory import CupheadMemoryReader
else:
    from cuphead_dqn.cuphead_env import CupheadEnv
    from cuphead_dqn.model import Conv3DDQN
    from cuphead_dqn.memory import ReplayBuffer
    from cuphead_dqn.agent import DQNAgent
    from cuphead_dqn.actions import discrete_to_multibinary, n_actions
    from cuphead_dqn.cuphead_memory import CupheadMemoryReader


def preprocess_obs(obs: np.ndarray) -> torch.Tensor:
    # obs shape: (stack, H, W), 转成 (B, stack, H, W)
    return torch.from_numpy(obs).unsqueeze(0)


def make_env():
    return CupheadEnv(
        decision_fps=12,
        frame_size=(192, 108),  # 降分辨率提速
        stack=3,                # 降 stack 减算力
        auto_restart=True,
        debug=False,
        hp_every_n=1,
        parry_every_n=9999,
        x_every_n=1,            # 启用X坐标读取
    )


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 调试：打印当前设备
    print("[Device]", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints_dqn")
    os.makedirs(ckpt_dir, exist_ok=True)
    env = make_env()
    obs, _ = env.reset()
    obs_tensor = preprocess_obs(obs).float()

    q_net = Conv3DDQN(stack=obs_tensor.shape[1], n_actions=n_actions())
    target_net = Conv3DDQN(stack=obs_tensor.shape[1], n_actions=n_actions())
    agent = DQNAgent(q_net, target_net, lr=1e-4, gamma=0.5, device=device)
    replay = ReplayBuffer(capacity=50_000)
    mem_reader = CupheadMemoryReader()

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 5e-5  # 更慢衰减，类似 HK 先充分探索
    batch_size = 8        # 降低 batch 提速
    target_update = 500   # 更频繁地同步目标网络
    memory_warmup = 1000  # 先攒一些样本再学习
    pre_learn_loops = 1   # 每局开始前学习次数
    post_learn_loops = 1  # 每局结束后学习次数

    global_step = 0
    learn_steps = 0
    episode_rewards = []
    ep_reward = 0
    obs_buffer = deque(maxlen=1)
    obs_buffer.append(obs)

    # 如存在 checkpoint，恢复训练
    ckpt_path = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=device)
            q_net.load_state_dict(state["q_net"])
            target_net.load_state_dict(state["target_net"])
            agent.optim.load_state_dict(state["optimizer"])
            global_step = state.get("global_step", 0)
            epsilon = state.get("epsilon", epsilon)
            episode_rewards = state.get("episode_rewards", episode_rewards)
            print(f"[CKPT] Loaded from {ckpt_path}, step={global_step}, eps={epsilon:.2f}")
        except Exception as e:
            print(f"[CKPT] Load failed: {e}")

    def learn_from_buffer():
        nonlocal learn_steps
        if len(replay) <= memory_warmup:
            return
        batch = replay.sample(batch_size)
        agent.learn(batch)
        learn_steps += 1
        if learn_steps % target_update == 0:
            agent.update_target()

    def save_checkpoint(tag="last"):
        path = os.path.join(ckpt_dir, "latest.pt")  # 始终覆盖一份，节省空间
        torch.save(
            {
                "q_net": q_net.state_dict(),
                "target_net": target_net.state_dict(),
                "optimizer": agent.optim.state_dict(),
                "global_step": global_step,
                "epsilon": epsilon,
                "episode_rewards": episode_rewards,
            },
            path,
        )
        print(f"[CKPT] Saved {path}")

    try:
        while True:
            # 局前预学习（参考 HK，在有样本后批量学习几次）
            for _ in range(pre_learn_loops):
                learn_from_buffer()

            epsilon = max(epsilon_min, epsilon - epsilon_decay)
            obs_tensor = preprocess_obs(obs).float().to(device)
            action_idx = agent.select_action(obs_tensor, epsilon, n_actions())
            action = discrete_to_multibinary(action_idx)

            penalty = 0.0
            try:
                x_val = mem_reader.read_player_x()
            except Exception:
                x_val = None
            if x_val is not None:
                if x_val < -610 and action[0] == 1:  # 向左越界
                    penalty -= 5.0
                if x_val > 228 and action[1] == 1:  # 向右越界
                    penalty -= 5.0

            next_obs, reward, done, _, info = env.step(action)
            reward += penalty
            ep_reward += reward

            replay.push(obs, action_idx, reward, next_obs, float(done))
            obs = next_obs
            global_step += 1

            if done:
                # 局后批量学习
                for _ in range(post_learn_loops):
                    learn_from_buffer()

                episode_rewards.append(ep_reward)
                ep_reward = 0
                obs, _ = env.reset()

            if global_step % 500 == 0:
                avg_r = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
                fps_val = getattr(env, "_last_fps", 0.0)
                print(f"step={global_step}, eps={epsilon:.2f}, avgR10={avg_r:.2f}, fps={fps_val:.2f}")
                save_checkpoint(tag="last")

    except KeyboardInterrupt:
        print("Interrupted, saving checkpoint...")
        save_checkpoint(tag="interrupt")
    finally:
        try:
            save_checkpoint(tag="final")
        except Exception as e:
            print(f"[CKPT] Save failed: {e}")
        env.close()


if __name__ == "__main__":
    train()
