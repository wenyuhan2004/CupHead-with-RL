# CupHead-with-RL

使用深度强化学习（DQN）自动游玩《茶杯头》（Cuphead）的完整训练系统，基于 PyTorch 实现。

## 项目简介

本项目实现了一个端到端的强化学习系统，能够通过视觉输入学习如何游玩《茶杯头》游戏。系统使用：
- **深度 Q 网络（DQN）** 作为核心算法
- **3D 卷积神经网络** 处理时序帧堆叠
- **内存读取 + OCR** 双重方案获取游戏状态
- **复杂奖励系统** 引导智能体学习有效策略

## 核心功能

### 🎮 环境系统 (`cuphead_env.py`)
- **视觉输入**：使用 dxcam 高性能屏幕捕获（~20 FPS），RGB 帧处理
- **动作空间**：10 维 MultiBinary 向量
  - 方向移动：左、右、上、下
  - 战斗动作：跳跃（小跳/大跳）、射击、冲刺、锁定、技能
- **状态读取**：
  - 优先使用 Windows API 内存读取（Boss/玩家 HP、坐标、技能点）
  - 备用 OCR 方案（需 Tesseract）
- **智能奖励系统**：
  - Boss 伤害奖励：每 9 点伤害给予正向奖励
  - 生存奖励：根据战斗时间和 Boss 血量进度动态调整
  - 穿越奖励：鼓励横向移动探索（左右穿越地图）
  - 边界惩罚：防止卡边（左边界 -655，右边界 +125）
  - 受伤惩罚：玩家掉血时给予 -20 惩罚

### 🧠 模型架构 (`model.py`)
- **Conv3DDQN**：3D 卷积 + 残差块网络
  - 输入：(Batch, 3, Stack, Height, Width) - RGB 时序堆叠
  - 主干：5 层 ResidualBlock3D，通道数 [16, 32, 64, 96, 128]
  - 时空下采样：保留时间维度信息，空间分辨率逐步降低
  - 输出：14 个离散动作的 Q 值
- **特点**：
  - 批归一化（BatchNorm3D）加速训练
  - 自适应平均池化提取全局特征
  - 残差连接缓解梯度消失

### 🚀 训练系统 (`train.py`)
- **DQN 算法**：经验回放 + 目标网络 + Huber 损失
- **超参数**：
  - 学习率：1e-4，优化器：Adam
  - 折扣因子 γ：0.2（注重短期奖励）
  - 探索策略：ε 从 1.0 线性衰减至 0.15（每步 -5e-5）
  - 经验池容量：50,000 条转移
  - 批量大小：8
  - 目标网络更新频率：500 步
- **训练流程**：
  - 预热阶段：积累 1000 条经验后开始学习
  - 局前/局后学习：每局额外执行 1 轮梯度更新
  - 自动检查点：每 500 步更新 `latest.pt`，每 50,000 步归档
  - TensorBoard 可视化：奖励、长度、胜率、FPS 实时监控
- **断点续训**：自动加载 `checkpoints_dqn/latest.pt` 继续训练

### 🎯 动作映射 (`actions.py`)
定义 14 个离散动作组合：
- 0-2：静止/左走/右走 + 射击
- 3-5：左小跳/原地小跳/右小跳 + 射击
- 6-8：左大跳/原地大跳/右大跳 + 射击
- 9-10：左冲刺/右冲刺（不射击）
- 11-12：下蹲左冲刺/下蹲右冲刺（不射击）
- 13：下蹲射击

**设计亮点**：
- 方向互斥（左右/上下不可同时触发）
- 跳跃强度区分（长按/短按）
- 冲刺冷却机制（5 步）
- 防卡键保护（20 步方向限制）

### 🔧 工具脚本

#### `test_cuphead_hp.py` - 游戏状态读取
- **内存读取**：通过 Windows API 直接读取游戏内存
  - Boss HP：`mono.dll + 0x00264A68 → ... → float`
  - Player HP：`mono.dll + 0x002675D8 → ... → int`
  - Player X 坐标：`UnityPlayer.dll + 0x01468F30 → ... → float`
  - 技能点数：`mono.dll + 0x002BFB40 → ... → float`
- **OCR 读取**：使用 Tesseract 识别血条文字（备用方案）
- **去抖动处理**：连续 3 帧一致才更新数值，避免误读

#### `cuphead_memory.py` - 内存读取封装
- 面向对象的内存读取接口
- 支持配置文件 `cuphead_mem.json` 自定义偏移量
- 自动枚举进程模块（mono.dll、UnityPlayer.dll）
- 线程安全的指针链解析

#### `calibrate_roi.py` - ROI 标定工具
- 交互式框选 Boss 血条区域
- 生成 `roi_selected.json` 配置文件
- 支持 dxcam 实时截图预览

#### `replay.py` - 策略回放
- 加载训练好的模型执行单局游戏
- 关闭探索（ε=0），展示最优策略
- 实时打印每步奖励和游戏状态

#### `agent.py` - DQN 智能体
- 封装 Q 网络和目标网络
- ε-贪心动作选择
- 梯度裁剪（max_norm=5.0）防止梯度爆炸

#### `memory.py` - 经验回放缓冲区
- 使用 deque 实现 FIFO 缓冲区
- 批量采样时预堆叠为 numpy 数组，避免转换警告
- 存储五元组 (obs, action, reward, next_obs, done)

## 环境要求

### 系统环境
- **操作系统**：Windows 10/11（需内存读取权限）
- **Python**：3.8+
- **游戏**：Cuphead（Steam/GOG 版本均可）

### 依赖库
核心依赖：
- `torch>=1.12.0` - PyTorch 深度学习框架
- `gymnasium>=0.26.0` - 强化学习环境接口
- `numpy>=1.21.0` - 数值计算
- `opencv-python>=4.6.0` - 图像处理
- `dxcam>=0.0.5` - 高性能屏幕捕获
- `pyautogui>=0.9.53` - 键盘输入模拟
- `pygetwindow>=0.0.9` - 窗口管理
- `pywin32>=304` - Windows API（内存读取）
- `tensorboard>=2.10.0` - 训练可视化

可选依赖：
- `pytesseract>=0.3.10` - OCR 引擎接口（需安装 Tesseract-OCR）

### Tesseract OCR（可选）
如需使用 OCR 读取血量，请安装 [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract)：
- 默认路径：`C:\Program Files\Tesseract-OCR\tesseract.exe`
- 自定义路径：设置环境变量 `TESSERACT_CMD`

## 快速开始

### 1. 安装依赖
```bash
# 使用 conda（推荐）
conda env create -f environment.yml
conda activate cuphead_rl

# 或使用 pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium numpy opencv-python dxcam pyautogui pywin32 tensorboard
```

### 2. 标定 Boss 血条 ROI（首次运行）
```bash
python calibrate_roi.py
```
- 在弹出窗口中拖拽框选 Boss 血条区域
- 按回车确认，生成 `roi_selected.json`

### 3. 启动训练
```bash
python train.py
```
训练过程中会：
- 每 500 步打印统计信息（平均奖励、局长、胜率、FPS）
- 自动保存 `checkpoints_dqn/latest.pt`（断点续训）
- 每 50,000 步归档一次模型

### 4. 监控训练进度
```bash
# 新开终端
tensorboard --logdir checkpoints_dqn/tb
```
访问 `http://localhost:6006` 查看：
- `episode/reward`：单局总奖励
- `episode/length`：单局步数
- `episode/win`：胜率
- `stats/fps`：训练速度

### 5. 回放最优策略
```bash
python replay.py
```
加载 `checkpoints_dqn/latest.pt`，运行一局游戏并打印详细信息。

## 项目结构
```
CupHead-with-RL/
├── cuphead_env.py          # Gym 环境主体（奖励、动作、状态管理）
├── train.py                # DQN 训练主循环
├── model.py                # Conv3D DQN 网络定义
├── agent.py                # DQN 智能体（ε-贪心、学习逻辑）
├── memory.py               # 经验回放缓冲区
├── actions.py              # 离散动作映射
├── cuphead_memory.py       # 内存读取封装类
├── test_cuphead_hp.py      # 游戏状态读取工具
├── calibrate_roi.py        # ROI 交互式标定
├── replay.py               # 策略回放脚本
├── environment.yml         # Conda 环境配置
├── roi_selected.json       # Boss 血条 ROI 坐标（首次运行生成）
├── checkpoints_dqn/        # 模型检查点目录
│   ├── latest.pt           # 最新模型（自动更新）
│   ├── ckpt_step_*.pt      # 归档模型
│   └── tb/                 # TensorBoard 日志
└── README.md               # 本文档
```

## 配置与调参

### 环境参数 (`cuphead_env.py`)
```python
CupheadEnv(
    decision_fps=12,                    # 决策频率（Hz）
    frame_size=(192, 108),              # 观测分辨率
    stack=4,                            # 帧堆叠数量
    reward_boss_damage_mul=0.1,         # Boss 伤害奖励系数
    reward_progress_bonus=1.0,          # 生存奖励（每秒）
    reward_player_damage_penalty=10.0,  # 受伤惩罚
    reward_wall_penalty=5.0,            # 边界惩罚
    x_min=-655.0, x_max=125.0,          # X 坐标边界
)
```

### 训练参数 (`train.py`)
```python
# DQN 超参数
lr=1e-4                     # 学习率
gamma=0.2                   # 折扣因子
epsilon_decay=5e-5          # 探索衰减率
batch_size=8                # 批量大小
capacity=50_000             # 经验池容量
target_update=500           # 目标网络更新频率

# 训练控制
memory_warmup=1000          # 预热步数
pre_learn_loops=1           # 局前学习次数
post_learn_loops=1          # 局后学习次数
```

### 内存地址配置 (`cuphead_mem.json`，可选)
如游戏版本更新导致内存地址变化，创建此文件覆盖默认配置：
```json
{
  "boss": {
    "module": "mono.dll",
    "base_offset": "0x00264A68",
    "offsets": [160, 2880, 368],
    "final_offset": 60,
    "type": "float"
  },
  "player": {
    "module": "mono.dll",
    "base_offset": "0x002675D8",
    "offsets": [3240, 32, 96, 96],
    "final_offset": 180,
    "type": "int"
  }
}
```

## 性能指标

### 训练效率
- **抓帧速度**：~20 FPS（dxcam 硬件加速）
- **决策频率**：12 Hz（实际约 10-15 Hz 含推理）
- **GPU 占用**：~2 GB（NVIDIA RTX 30/40 系列）
- **单局时长**：30-180 秒（取决于表现）

### 收敛情况
- **初期探索**（0-50k 步）：随机动作为主，ε > 0.75
- **策略形成**（50k-150k 步）：开始避开子弹，ε 降至 0.3
- **策略优化**（150k+ 步）：稳定输出，偶尔获胜

## 常见问题

### Q: 训练时 FPS 很低（<5）
A: 
- 检查 GPU 是否启用：`print(torch.cuda.is_available())`
- 降低分辨率：`frame_size=(128, 72)`
- 减少帧堆叠：`stack=3`

### Q: 内存读取失败
A:
- 以管理员身份运行 Python
- 确认游戏窗口标题为 "Cuphead"
- 检查游戏版本（仅支持 Steam/GOG 标准版）


### Q: 训练中断后如何恢复
A: 直接再次运行 `python train.py`，会自动加载 `checkpoints_dqn/latest.pt`

### Q: OCR 识别不准确
A:
- 重新标定 ROI：`python calibrate_roi.py`
- 确保 Boss 血条清晰可见（无遮挡）
- 优先使用内存读取（更稳定）

## 技术亮点

1. **高效视觉处理**：dxcam 硬件加速 + 低分辨率输入，平衡速度与信息量
2. **时序特征提取**：3D 卷积保留帧间时序关系，捕捉运动模式
3. **双重状态感知**：内存读取（精确）+ OCR（兼容），鲁棒性强
4. **复杂奖励工程**：多维度奖励（伤害/生存/探索/惩罚）引导策略多样性
5. **防卡键机制**：方向互斥 + 冷却系统 + 自动抬键，避免操作死锁
6. **断点续训**：自动保存/加载，支持长时间训练

## 后续改进方向

- [ ] 引入 PPO/SAC 等 on-policy 算法
- [ ] 多 Boss 关卡泛化能力
- [ ] 注意力机制定位关键目标
- [ ] 层次强化学习拆解复杂动作
- [ ] 模仿学习预训练加速收敛



