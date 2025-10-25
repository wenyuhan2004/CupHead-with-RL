import os
import time
import numpy as np
import pygetwindow as gw
import pyautogui as pag
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from cuphead_env import CupheadEnv


# ------------------ 聚焦窗口 ------------------
def focus_cuphead_window():
    """确保 Cuphead 游戏窗口被激活"""
    try:
        titles = [t for t in gw.getAllTitles() if t and "cuphead" in t.lower()]
        if titles:
            w = gw.getWindowsWithTitle(titles[0])[0]
            w.activate(); w.restore()
            time.sleep(0.2)
            print("[INFO] Cuphead 窗口已激活。")
            return True
    except Exception:
        pass
    # 兜底：点击屏幕中心
    try:
        sw, sh = pag.size()
        pag.moveTo(sw // 2, sh // 2, duration=0.05)
        pag.click()
        print("[INFO] 使用鼠标点击方式激活窗口。")
        return True
    except Exception:
        print("[WARN] 无法激活窗口。")
        return False


# ------------------ 查找最新模型 ------------------
def get_latest_model(path="./models"):
    """返回 models/ 中步数最大的模型路径"""
    ckpts = [f for f in os.listdir(path) if f.endswith(".zip") and "cuphead_ppo" in f]
    if not ckpts:
        return None
    ckpts.sort(key=lambda n: int("".join([c for c in n if c.isdigit()])))
    latest = os.path.join(path, ckpts[-1])
    return latest


# ------------------ 主评估逻辑 ------------------
if __name__ == "__main__":
    focus_cuphead_window()

    latest_ckpt = get_latest_model("./models")
    if latest_ckpt is None:
        raise FileNotFoundError("❌ 未找到模型，请先训练或放入 models/ 文件夹中。")

    print(f"[INFO] 正在加载模型: {latest_ckpt}")
    env = DummyVecEnv([lambda: CupheadEnv(
        decision_fps=15,
        frame_size=(96, 96),
        stack=4,
        auto_restart=True,
        debug=False
    )])

    model = PPO.load(latest_ckpt, env=env, device="cpu")
    print("[INFO] 模型加载完成，开始执行策略...")

    obs = env.reset()
    episode_rewards = []
    total_reward = 0
    step_count = 0
    start_time = time.time()

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward[0]
            step_count += 1

            # 每 N 步打印一次状态
            if step_count % 200 == 0:
                bhp = info[0].get("boss_hp", None)
                php = info[0].get("player_hp", None)
                print(f"[STEP {step_count}] Reward={total_reward:.2f}, BossHP={bhp}, PlayerHP={php}")

            # 如果回合结束
            if done[0]:
                episode_rewards.append(total_reward)
                print(f"[EPISODE END] Total reward={total_reward:.2f}, len={step_count}")
                total_reward = 0
                step_count = 0
                obs = env.reset()

            # 若要测试 10 分钟后自动退出
            if time.time() - start_time > 600:  # 600 秒 = 10 分钟
                break

    except KeyboardInterrupt:
        print("\n[INFO] 手动中断。")

    finally:
        env.close()
        if episode_rewards:
            print(f"\n✅ 平均回合奖励: {np.mean(episode_rewards):.3f}")
            print(f"共完成 {len(episode_rewards)} 回合。")
        print("[INFO] 评估结束。")
