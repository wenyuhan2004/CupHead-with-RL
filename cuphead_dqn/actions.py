"""离散动作到 MultiBinary 的映射。"""

# MultiBinary 10: [L,R,Up,Down,Jump,Shoot,Dash,Lock,Special,JumpHold]
# 根据当前环境自定义一组高层动作

ACTION_MAP = [
    [0,0,0,0,0,0,0,0,0,0],  # 静止
    [1,0,0,0,0,0,0,0,0,0],  # 左
    [0,1,0,0,0,0,0,0,0,0],  # 右
    [0,0,0,0,1,0,0,0,0,0],  # 跳
    [0,0,0,0,0,1,0,0,0,0],  # 射击
    [0,1,0,0,1,1,0,0,0,0],  # 右跳射
    [0,0,0,1,0,0,1,0,0,0],  # 下蹲+dash（S 映射为 S+shift）
    [0,0,1,0,0,0,0,1,0,0],  # 抬枪+锁定射
]


def discrete_to_multibinary(idx: int):
    return ACTION_MAP[idx]


def n_actions():
    return len(ACTION_MAP)
