"""Replay one episode with the latest checkpoint and print rewards."""
import os
import sys
import torch


# 兼容直接运行
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from train import make_env, preprocess_obs  # type: ignore
    from model import Conv3DDQN  # type: ignore
    from actions import discrete_to_multibinary, n_actions  # type: ignore
else:
    from cuphead_dqn.train import make_env, preprocess_obs  # type: ignore
    from cuphead_dqn.model import Conv3DDQN  # type: ignore
    from cuphead_dqn.actions import discrete_to_multibinary, n_actions  # type: ignore


def load_policy(device="cpu"):
    env = make_env()
    obs, _ = env.reset()
    obs_tensor = preprocess_obs(obs).float()

    q_net = Conv3DDQN(stack=obs_tensor.shape[1], n_actions=n_actions())
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints_dqn", "latest.pt")
    if os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=device)
            q_net.load_state_dict(state["q_net"])
            print(f"[CKPT] Loaded {ckpt_path}")
        except Exception as exc:  # pragma: no cover - 仅日志
            print(f"[CKPT] Load failed: {exc}")
    q_net.to(device)
    q_net.eval()
    return env, q_net


def run_episode(max_steps=5000, device="cpu"):
    env, q_net = load_policy(device=device)
    obs, _ = env.reset()
    total_reward = 0.0

    with torch.no_grad():
        for step in range(max_steps):
            obs_tensor = preprocess_obs(obs).float().to(device)
            q_values = q_net(obs_tensor)
            action_idx = int(q_values.argmax(dim=1).item())
            action = discrete_to_multibinary(action_idx)

            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            print(f"step={step:04d} r={reward:+.3f} total={total_reward:+.3f} info={info}")

            if done:
                print(f"[EP END] done_reason={info.get('done_reason')} total_reward={total_reward:+.3f}")
                break
        else:
            print(f"[MAX STEPS REACHED] total_reward={total_reward:+.3f}")
    env.close()


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_episode(device=dev)
