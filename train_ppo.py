# train_ppo.py
import os, re, time, numpy as np
import pygetwindow as gw
import pyautogui as pag
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from cuphead_env import CupheadEnv

# ------------------ 聚焦窗口 ------------------
def focus_cuphead_window():
    """尝试把 Cuphead 窗口置顶、激活"""
    try:
        titles = [t for t in gw.getAllTitles() if t and "cuphead" in t.lower()]
        if titles:
            w = gw.getWindowsWithTitle(titles[0])[0]
            w.activate(); w.restore()
            time.sleep(0.2)
            print("[INFO] Cuphead 窗口已激活。");  return True
    except Exception:
        pass
    try:
        sw, sh = pag.size()
        pag.moveTo(sw // 2, sh // 2, duration=0.05)
        pag.click(); time.sleep(0.1)
        print("[INFO] 使用鼠标点击方式激活窗口。");  return True
    except Exception:
        print("[WARN] 无法激活窗口。");  return False

# ------------------ 训练回调 ------------------
class CupheadTrainCallback(BaseCallback):
    def __init__(self, print_freq: int = 2000, verbose: int = 1):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.ep_rews, self.ep_lens = [], []
        self.last_info = None
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.ep_rews.append(ep["r"]); self.ep_lens.append(ep["l"])
            self.last_info = info
        if self.n_calls % self.print_freq == 0:
            print(f"\n=== Global Step: {self.num_timesteps} ===")
            if self.ep_rews:
                print(f"Recent Avg Reward(10): {np.mean(self.ep_rews[-10:]):.3f}")
                print(f"Recent Avg EpLen(10):  {np.mean(self.ep_lens[-10:]):.1f}")
            if self.last_info:
                bhp = self.last_info.get("boss_hp", None)
                php = self.last_info.get("player_hp", None)
                parry = self.last_info.get("parry", None)
                xval = self.last_info.get("x", None)
                print(f"HP snapshot -> Boss:{'None' if bhp is None else f'{bhp:.1f}'}, "
                      f"Player:{'None' if php is None else str(php)}")
                print(f"Parry:{parry}   X:{'None' if xval is None else f'{xval:.1f}'}   "
                      f"Phase2:{self.last_info.get('phase2', False)}   "
                      f"StillSteps:{self.last_info.get('x_still_steps', 0)}")
            print("========================================")
        return True

# ------------------ VecEnv 工厂 ------------------
def make_env(rank: int):
    def _init():
        return CupheadEnv(
            decision_fps=15,
            frame_size=(96, 96),
            stack=4,
            auto_restart=True,
            debug=False,
            # ↓↓↓ 降采样读取（与 read_hp.py 的限频配合，保证吞吐）
            hp_every_n=3,
            parry_every_n=3,
            x_every_n=2,
            # ↓↓↓ 奖励塑形参数（可按需要微调）
            x_still_thresh=50.0,
            reward_parry_gain=0.20,
            reward_duck_dash=0.02,
            reward_phase_bonus=0.50,
            still_penalty_unit=0.01,
        )
    return _init

# ------------------ 自动续训工具 ------------------
def find_latest_checkpoint(folder="models", prefix="cuphead_ppo_"):
    if not os.path.isdir(folder):
        return None
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)_steps\.zip$")
    best = None; best_steps = -1
    for f in os.listdir(folder):
        m = pat.match(f)
        if m:
            steps = int(m.group(1))
            if steps > best_steps:
                best_steps, best = steps, os.path.join(folder, f)
    return best

# ------------------ 主程序 ------------------
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    focus_cuphead_window()

    # 单环境
    vec = DummyVecEnv([make_env(0)])
    vec = VecMonitor(vec)

    # —— 从头训练 —— #
    # 若以后需要续训，把 next line 改为：ckpt_path = find_latest_checkpoint()
    ckpt_path = None

    if ckpt_path:
        print(f"[INFO] 载入最新检查点继续训练: {ckpt_path}")
        model = PPO.load(ckpt_path, env=vec, tensorboard_log="./logs_ppo/", device="auto")
    else:
        print("[INFO] 未加载检查点，从头开始训练。")
        # 可选：自定义网络结构；默认 CnnPolicy 已支持 MultiBinary 动作
        model = PPO(
            "CnnPolicy",
            vec,
            verbose=1,
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            ent_coef=0.01,      # MultiBinary 稍强探索
            clip_range=0.2,
            n_epochs=10,
            tensorboard_log="./logs_ppo/",
            device="auto",
        )

    # 定时保存 + 控制台打印
    ckpt_cb  = CheckpointCallback(save_freq=10_000, save_path="./models/", name_prefix="cuphead_ppo")
    train_cb = CupheadTrainCallback(print_freq=1000)

    # 开训
    model.learn(total_timesteps=1_000_000, callback=[ckpt_cb, train_cb])
    model.save("./models/cuphead_ppo_final")
    vec.close()
