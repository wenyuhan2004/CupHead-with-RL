# CupHead_RL（Windows）

使用强化学习（PPO, Stable-Baselines3）让智能体在《茶杯头（Cuphead）》中学习通关 Boss。

通过 **dxcam** 抓屏、**PyAutoGUI** 模拟按键、**Tesseract OCR** 读取 HP 条，并利用 **Gymnasium** + **Stable-Baselines3** 进行训练。

---

## 🧩 环境要求

- **系统**：Windows 10/11（桌面模式）
- **Python**：推荐 3.10（Conda 环境）
- **GPU**：可选（训练瓶颈在屏幕采样和 OCR，CPU 也能训练）

---

## ⚙️ 安装步骤

### 1️⃣ 创建环境
```bash
conda env create -f environment.yml
conda activate cuphead_rl
```

### 2️⃣ 安装 Tesseract OCR

下载并安装：
> https://github.com/UB-Mannheim/tesseract/wiki  

默认路径应为：
```
C:\Program Files\Tesseract-OCR\tesseract.exe
```
如果不同，请修改 `read_hp.py`：
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## 🧩 必需 Mod

使用 **BepInEx 5 + Cuphead DebugMod v1.6.1**  
（无需修改游戏 DLL）

### 安装方式
1. 打开游戏根目录（含 `Cuphead.exe`）。  
2. 将 `BepInEx/`、`doorstop_config.ini`、`winhttp.dll` 等文件解压进去。  
3. 启动游戏，首次会生成 `BepInEx/config/`。  
4. 按 `F1` 打开插件菜单，确认存在 **Cuphead.DebugMod 1.6.1**。

> Debug HUD 显示 Boss HP 和玩家 HP，本项目通过 OCR 读取这些数据。

---

## 🧭 代码结构

```
CupHead_RL/
├─ calibrate_roi.py
├─ read_hp.py
├─ cuphead_env.py
├─ train_ppo.py
├─ eval_ppo.py
├─ control_smoketest.py
└─ cuphead_roi.json
```

---

## 🚀 使用流程

### 第一步：标定 ROI
```bash
python calibrate_roi.py
```

### 第二步：验证识别
```bash
python read_hp.py
```

### 第三步：按键测试
```bash
python control_smoketest.py
```

### 第四步：训练智能体
```bash
python train_ppo.py
```

模型自动保存到 `models/`：
```
models/cuphead_ppo_10000_steps.zip
models/cuphead_ppo_20000_steps.zip
...
```

### 第五步：断点续训
自动加载最新 checkpoint：
```python
model = PPO.load("./models/cuphead_ppo_40000_steps.zip", env=vec)
```

### 第六步：仅执行策略（评估）
```bash
python eval_ppo.py
```

---

## 🧮 控制台输出说明

训练过程中会打印：
```
=== Global Step: 10000 ===
Recent Avg Reward (last 10 eps): 55.9
Recent Avg EpLen (last 10 eps): 23.3
HP snapshot -> Boss: 865.8, Player: 2
```

---

## 🧠 常见问题

**FPS 低**
- 关闭 `imshow`
- 设置 `hp_every_n=3`
- 游戏窗口化 1280×720

**抓屏黑屏**
- 仅支持本地桌面，关闭远程桌面。

**按键无效**
- 确认窗口激活，输入法为英文。

---

## ⚙️ PPO 参数建议

| 参数 | 推荐值 |
|------|---------|
| decision_fps | 15 |
| frame_size | (96,96) |
| stack | 4 |
| hp_every_n | 3 |
| n_steps | 2048 |
| batch_size | 256 |
| ent_coef | 0.01 |
| gamma | 0.99 |
| n_epochs | 10 |

---

## 🧰 快速命令

```bash
conda env create -f environment.yml
conda activate cuphead_rl
python calibrate_roi.py
python read_hp.py
python control_smoketest.py
python train_ppo.py
python eval_ppo.py
```
