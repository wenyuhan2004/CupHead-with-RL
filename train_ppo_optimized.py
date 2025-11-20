# train_ppo_optimized.py - 高性能优化版本
# 目标：保持9+ FPS同时具备完整的监控和调试功能

import os
import re
import time
import subprocess
import webbrowser
import shutil
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from cuphead_env import CupheadEnv
from cuphead_model import CupheadHKStyleExtractor
import warnings
warnings.filterwarnings("ignore")

def launch_tensorboard(logdir: str, port: int = 6006):
    """启动 TensorBoard 并尝试打开浏览器"""
    if shutil.which("tensorboard") is None:
        print("[WARN] 未找到 tensorboard 命令，跳过监控网页启动")
        return None
    cmd = ["tensorboard", f"--logdir={logdir}", f"--port={port}", "--reload_multifile=true"]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        url = f"http://localhost:{port}"
        print(f"[INFO] TensorBoard 已启动: {url}")
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass
        return proc
    except Exception as exc:
        print(f"[WARN] 启动 TensorBoard 失败: {exc}")
        return None

class HudConvExtractor(BaseFeaturesExtractor):
    """更深的卷积特征提取器，兼顾弹幕识别"""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_input_channels, observation_space.shape[1], observation_space.shape[2])
            conv_out = self.conv(dummy).view(1, -1).shape[1]
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations):
        return self.linear(self.conv(observations))

def make_env(rank: int):
    """创建高性能优化环境"""
    def _init():
        env = CupheadEnv(
            decision_fps=12,
            frame_size=(400, 225),
            stack=4,
            auto_restart=True,
            debug=False,
            hp_every_n=18,
            parry_every_n=30,
            x_every_n=30,
        )
        return Monitor(env)
    return _init

class CupheadOptimizedCallback(BaseCallback):
    """高性能优化回调 - 平衡性能与监控"""
    
    def __init__(self, 
                 check_freq: int = 1000,
                 save_path: str = "./checkpoints_optimized",
                 max_no_improvement_evals: int = 40,
                 min_evals: int = 5,
                 episode_buffer_size: int = 200,
                 early_stop_threshold: float = 0.0,
                 verbose: int = 1):
        super().__init__(verbose)
        
        # 基础设置
        self.check_freq = check_freq
        self.save_path = save_path
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.episode_buffer_size = episode_buffer_size
        self.early_stop_threshold = early_stop_threshold
        
        # 性能监控
        self.episode_rewards = []
        self.episode_lengths = []
        self.fps_history = []
        self.last_info = {}
        
        # FPS计算
        self.fps_start_time = time.time()
        self.fps_last_print = time.time()
        self.fps_step_count = 0
        
        # 检查点管理
        self.best_mean_reward = -np.inf
        self.evaluations_since_best = 0
        self.early_stop_enabled = False
        
        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)
        
        print(f"[优化回调] 初始化完成")
        print(f"[优化回调] 保存路径: {save_path}")
        print(f"[优化回调] 检查频率: {check_freq}")

    def _on_step(self) -> bool:
        # FPS监控（高频）
        self.fps_step_count += 1
        current_time = time.time()
        
        # 收集episode信息
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_info = info["episode"]
                self.episode_rewards.append(ep_info["r"])
                self.episode_lengths.append(ep_info["l"])
                
                # 保持缓冲区大小
                if len(self.episode_rewards) > self.episode_buffer_size:
                    self.episode_rewards.pop(0)
                    self.episode_lengths.pop(0)
            
            # 更新最新信息
            self.last_info.update(info)
        
        # 定期FPS报告（每30秒）
        if current_time - self.fps_last_print >= 30.0:
            elapsed = current_time - self.fps_start_time
            current_fps = self.fps_step_count / elapsed if elapsed > 0 else 0
            self.fps_history.append(current_fps)
            
            print(f"\n[性能监控] 步数: {self.num_timesteps:,}")
            print(f"[性能监控] 当前FPS: {current_fps:.2f}")
            
            if self.episode_rewards:
                recent_rewards = self.episode_rewards[-20:] if len(self.episode_rewards) >= 20 else self.episode_rewards
                print(f"[性能监控] 最近{len(recent_rewards)}轮平均奖励: {np.mean(recent_rewards):.3f}")
            
            self.fps_last_print = current_time
        
        # 定期详细报告和检查点
        if self.n_calls % self.check_freq == 0:
            self._detailed_report()
            self._save_checkpoint()
        
        return True
    
    def _detailed_report(self):
        """详细报告"""
        print(f"\n{'='*60}")
        print(f"详细训练报告 - 步数: {self.num_timesteps:,}")
        print(f"{'='*60}")
        
        # 奖励统计
        if self.episode_rewards:
            rewards = np.array(self.episode_rewards)
            print(f"奖励统计:")
            print(f"  - 最近10轮平均: {np.mean(rewards[-10:]):.3f}")
            print(f"  - 最近50轮平均: {np.mean(rewards[-50:]):.3f}")
            print(f"  - 全部平均: {np.mean(rewards):.3f}")
            print(f"  - 最高奖励: {np.max(rewards):.3f}")
            print(f"  - 总轮数: {len(rewards)}")
        
        # FPS统计
        if self.fps_history:
            fps_array = np.array(self.fps_history)
            print(f"FPS统计:")
            print(f"  - 当前FPS: {fps_array[-1]:.2f}")
            print(f"  - 平均FPS: {np.mean(fps_array):.2f}")
            print(f"  - 最高FPS: {np.max(fps_array):.2f}")
        
        # 游戏状态
        if self.last_info:
            print(f"游戏状态:")
            boss_hp = self.last_info.get("boss_hp", "Unknown")
            player_hp = self.last_info.get("player_hp", "Unknown")
            parry = self.last_info.get("parry", "Unknown")
            x_pos = self.last_info.get("x", "Unknown")
            
            print(f"  - Boss血量: {boss_hp}")
            print(f"  - 玩家血量: {player_hp}")
            print(f"  - Parry状态: {parry}")
            print(f"  - X坐标: {x_pos}")

        print(f"{'='*60}\n")
    
    def _save_checkpoint(self):
        """保存检查点"""
        if not self.episode_rewards:
            return
        
        # 计算当前平均奖励
        current_mean_reward = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
        
        # 检查是否是最佳模型
        if current_mean_reward > self.best_mean_reward:
            self.best_mean_reward = current_mean_reward
            self.evaluations_since_best = 0
            if self.best_mean_reward >= self.early_stop_threshold:
                self.early_stop_enabled = True
            
            # 保存最佳模型
            best_model_path = os.path.join(self.save_path, f"best_model_{self.num_timesteps}_steps")
            self.model.save(best_model_path)
            print(f"[检查点] 新的最佳模型已保存: {best_model_path}")
            print(f"[检查点] 最佳平均奖励: {self.best_mean_reward:.3f}")
        else:
            self.evaluations_since_best += 1
        
        # 定期保存检查点
        checkpoint_path = os.path.join(self.save_path, f"checkpoint_{self.num_timesteps}_steps")
        self.model.save(checkpoint_path)
        print(f"[检查点] 定期检查点已保存: {checkpoint_path}")
        
        # 早停检查（仅在达到门槛后启用）
        if (self.early_stop_enabled and
            self.evaluations_since_best >= self.max_no_improvement_evals and 
            len(self.episode_rewards) >= self.min_evals * 10):
            print(f"[早停] {self.max_no_improvement_evals}次评估无改善，建议考虑停止训练")


def find_latest_checkpoint(folder="./checkpoints_optimized", prefix="checkpoint_"):
    """查找最新的检查点"""
    if not os.path.isdir(folder):
        return None
    
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)_steps\.zip$")
    best_file = None
    best_steps = -1
    
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            steps = int(match.group(1))
            if steps > best_steps:
                best_steps = steps
                best_file = os.path.join(folder, filename)
    
    return best_file

def main():
    """主训练函数"""
    print("="*60)
    print("CupHead 强化学习训练 - 高性能优化版")
    print("目标: 保持9+ FPS的高性能训练")
    print("="*60)
    
    # GPU检测和配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] 使用GPU: {gpu_name}")
        print(f"[GPU] 显存: {gpu_memory:.1f}GB")
        
        # 优化GPU设置
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    else:
        print("[INFO] 使用CPU训练")
    
    print(f"[INFO] 训练设备: {device}")
    
    log_dir = "./logs_optimized/"
    os.makedirs(log_dir, exist_ok=True)
    tb_port = int(os.environ.get("TENSORBOARD_PORT", "6006"))
    tb_proc = launch_tensorboard(log_dir, tb_port)

    # 创建环境
    print("[INFO] 创建优化环境...")
    num_envs = int(os.environ.get("CUPHEAD_NUM_ENVS", "1"))
    env_fns = [make_env(i) for i in range(num_envs)]
    if num_envs > 1:
        print(f"[INFO] 使用 SubprocVecEnv 并行环境数量: {num_envs}")
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    
    # 检查是否有现有检查点
    checkpoint_dir = "./checkpoints_optimized"
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint:
        print(f"[INFO] 发现现有检查点: {latest_checkpoint}")
        print("[INFO] 加载模型继续训练...")
        model = PPO.load(latest_checkpoint, env=env, device=device)
    else:
        print("[INFO] 创建新模型...")
        # 高性能优化的PPO参数
        model = PPO(
            "CnnPolicy",
            env,
            verbose=0,
            device=device,
            
            # 核心超参数 - 更偏向在线、低延迟训练
            n_steps=128,
            batch_size=32,
            n_epochs=4,
            learning_rate=1e-5,
            
            # 策略参数
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.0,
            max_grad_norm=0.5,
            gae_lambda=0.95,
            gamma=0.99,
            
            # 网络架构 - 快速推理
            policy_kwargs=dict(
                features_extractor_class=CupheadHKStyleExtractor,
                features_extractor_kwargs=dict(features_dim=256),
                net_arch=[dict(vf=[256, 128], pi=[256, 128])]
            ),
            
            tensorboard_log=log_dir
        )
    
    # 创建优化回调
    callback = CupheadOptimizedCallback(
        check_freq=1000,
        save_path=checkpoint_dir,
        min_evals=5,
        episode_buffer_size=200,
        early_stop_threshold=0.0
    )
    
    # 训练配置
    total_timesteps = 1_000_000
    
    print(f"[INFO] 开始高性能训练...")
    print(f"[INFO] 总训练步数: {total_timesteps:,}")
    print(f"[INFO] 检查点目录: {checkpoint_dir}")
    print(f"[INFO] 目标FPS: 9+")
    print("="*60)
    
    # 开始训练
    try:
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False  # 关闭进度条提升性能
        )
        
        training_time = time.time() - start_time
        print(f"\n[完成] 训练完成！总用时: {training_time/3600:.2f}小时")
        
    except KeyboardInterrupt:
        print("\n[中断] 训练被用户中断")
        interrupt_path = os.path.join(checkpoint_dir, "interrupted_model")
        model.save(interrupt_path)
        print(f"[保存] 中断模型已保存: {interrupt_path}")
        
    except Exception as e:
        print(f"\n[错误] 训练过程中出错: {e}")
        error_path = os.path.join(checkpoint_dir, "error_model")
        model.save(error_path)
        print(f"[保存] 错误模型已保存: {error_path}")
        raise
    finally:
        env.close()
        if tb_proc and tb_proc.poll() is None:
            tb_proc.terminate()
            print("[INFO] TensorBoard 已关闭")
    
    # 保存最终模型
    final_model_path = "cuphead_ppo_optimized_final"
    model.save(final_model_path)
    print(f"[保存] 最终模型已保存: {final_model_path}")
    
    # 训练总结
    print("\n" + "="*60)
    print("训练完成总结")
    print("="*60)
    if hasattr(callback, 'episode_rewards') and callback.episode_rewards:
        avg_reward = np.mean(callback.episode_rewards[-50:])
        max_reward = max(callback.episode_rewards)
        print(f"最近50轮平均奖励: {avg_reward:.2f}")
        print(f"最高奖励: {max_reward:.2f}")
        print(f"总训练轮数: {len(callback.episode_rewards)}")
    
    if hasattr(callback, 'fps_history') and callback.fps_history:
        avg_fps = np.mean(callback.fps_history)
        max_fps = max(callback.fps_history)
        print(f"平均训练FPS: {avg_fps:.2f}")
        print(f"最高FPS: {max_fps:.2f}")
    
    print(f"总训练步数: {total_timesteps:,}")
    print(f"检查点目录: {checkpoint_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
