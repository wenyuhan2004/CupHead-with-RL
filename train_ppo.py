# train_ppo.py
import os, re, time, numpy as np
import torch
import pygetwindow as gw
import pyautogui as pag
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# ✅ 使用异步OCR版环境
from cuphead_env_async import CupheadEnv


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
    def __init__(self, print_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.ep_rews, self.ep_lens = [], []
        self.last_info = None
        
        # FPS监控
        self.fps_start_time = time.time()
        self.fps_last_print = time.time()
        self.fps_step_count = 0

    def _on_step(self) -> bool:
        # FPS计算
        self.fps_step_count += 1
        current_time = time.time()
        
        # 每1秒打印一次FPS
        if current_time - self.fps_last_print >= 1.0:
            elapsed = current_time - self.fps_start_time
            fps = self.fps_step_count / elapsed if elapsed > 0 else 0
            print(f"[FPS] 平均FPS: {fps:.2f} | 总步数: {self.fps_step_count} | 运行时间: {elapsed:.1f}s")
            self.fps_last_print = current_time
        
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
                xstill = self.last_info.get("x_still", 0)
                print(f"HP -> Boss:{'None' if bhp is None else f'{bhp:.1f}'}, "
                      f"Player:{'None' if php is None else str(php)}")
                print(f"Parry:{parry}   X:{'None' if xval is None else f'{xval:.1f}'}   "
                      f"StillSteps:{xstill}")
            print("========================================")
        return True


# ------------------ VecEnv 工厂 ------------------
def make_env(rank: int):
    def _init():
        # ✅ 参数与异步环境匹配：仅 hp_every_n 控制 OCR 线程频率
        return CupheadEnv(
            decision_fps=12,         # 优化：降低目标FPS减少计算压力
            frame_size=(48, 48),     # 优化：更小的图像尺寸
            stack=3,                 # 优化：降低帧堆叠数量
            auto_restart=True,
            debug=False,
            hp_every_n=20,           # 优化：提高死亡监测频率
            # —— 奖励塑形参数（与同步版一致）——
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
    # GPU检测和设备配置
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[INFO] 检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # 设置GPU内存优化
        torch.cuda.empty_cache()
    else:
        device = "cpu"
        print("[INFO] 未检测到GPU，使用CPU")
    
    os.makedirs("models", exist_ok=True)
    focus_cuphead_window()

    # 单环境
    vec = DummyVecEnv([make_env(0)])
    vec = VecMonitor(vec)
    
    # GPU优化设置
    if device == "cuda":
        # PyTorch GPU优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("[INFO] 启用CUDNN优化")
        
        # 如果有足够GPU内存，可以增加并行环境数量
        # vec = SubprocVecEnv([make_env(i) for i in range(2)])  # 可选：2个并行环境

    # —— 从头训练 —— #
    # 若以后需要续训，把 next line 改为：ckpt_path = find_latest_checkpoint()
    ckpt_path = None

    if ckpt_path:
        print(f"[INFO] 载入最新检查点继续训练: {ckpt_path}")
        model = PPO.load(ckpt_path, env=vec, tensorboard_log="./logs_ppo/", device=device)
    else:
        print(f"[INFO] 未加载检查点，从头开始训练。使用设备: {device}")
        model = PPO(
            "CnnPolicy",
            vec,
            verbose=1,
            n_steps=512,           # 优化：适中的批次大小平衡性能和内存
            batch_size=128,        # 优化：提高batch size充分利用GPU
            learning_rate=3e-4,
            gamma=0.99,
            ent_coef=0.01,         # MultiBinary 稍强探索
            clip_range=0.2,
            n_epochs=3,            # 优化：更少的epochs提高速度
            tensorboard_log="./logs_ppo/",
            device=device,      # 使用检测到的设备
            policy_kwargs=dict(
                net_arch=[128, 128] if device == "cuda" else [64, 64],  # GPU时使用更大网络
                activation_fn=torch.nn.ReLU,
            ),
        )

    # 定时保存 + 控制台打印
    ckpt_cb  = CheckpointCallback(save_freq=10_000, save_path="./models/", name_prefix="cuphead_ppo")
    train_cb = CupheadTrainCallback(print_freq=1000)

    # 开训
    model.learn(total_timesteps=1_000_000, callback=[ckpt_cb, train_cb])
    model.save("./models/cuphead_ppo_final")
    vec.close()
 