# cuphead_env.py
import time
import json
import cv2
import numpy as np
import dxcam
import pyautogui as pag
import gymnasium as gym
from gymnasium import spaces

pag.FAILSAFE = False

# 复用识别函数（与 read_hp.py 同目录）
from read_hp import (
    grab, ocr_number_block, parse_boss_hp,
    read_player_digit_only, extract_digit_bin, is_digit_one_by_shape,
    detect_dead_text, is_hp_badge_red, is_dead_fast,
    # 新增：Parry 与 X 坐标
    read_parry_count_only, read_xcoord_only
)

def focus_cuphead_window():
    try:
        import pygetwindow as gw
        wins = [w for w in gw.getAllTitles() if w and 'cuphead' in w.lower()]
        if wins:
            w = gw.getWindowsWithTitle(wins[0])[0]
            w.activate(); w.restore()
            time.sleep(0.2)
            return True
    except Exception:
        pass
    try:
        sw, sh = pag.size()
        pag.moveTo(sw//2, sh//2, duration=0.05)
        pag.click()
        time.sleep(0.1)
        return True
    except Exception:
        return False


class CupheadEnv(gym.Env):
    """
    Cuphead 强化学习环境（并发按键 + 高吞吐）
    - 动作空间: MultiBinary(9) -> [L,R,Up,Down,Jump,Shoot,Dash,Lock,Special(K)]
    - 观测: (stack, H, W) 灰度
    - 降采样读取HP/Parry/X 以减少OCR开销
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        decision_fps: int = 15,
        frame_size=(96, 96),
        stack: int = 4,
        debug: bool = False,
        auto_restart: bool = True,
        hp_every_n: int = 2,        # 每 N 步读取一次 HP（>=1）
        parry_every_n: int = 2,     # 每 N 步读取一次 Parry
        x_every_n: int = 2,         # 每 N 步读取一次 X 坐标
        x_still_thresh: float = 50.0,   # |Δx| < 阈值 视为“几乎不动”
        reward_parry_gain: float = 0.20,  # 每次 parry 计数上升的奖励
        reward_duck_dash: float = 0.02,   # 下蹲+冲刺的小额奖励
        reward_phase_bonus: float = 0.50, # Boss HP 从 >450 降到 ≤450 的一次性奖励
        still_penalty_unit: float = 0.01, # “连续不动”惩罚单位，每步叠加至最多5倍
    ):
        super().__init__()
        self.debug = debug
        self.auto_restart = auto_restart
        self.dt = 1.0 / max(1, decision_fps)
        self.W, self.H, self.stack = frame_size[0], frame_size[1], stack

        self.hp_every_n = max(1, hp_every_n)
        self.parry_every_n = max(1, parry_every_n)
        self.x_every_n = max(1, x_every_n)
        self.x_still_thresh = float(x_still_thresh)

        self.reward_parry_gain = float(reward_parry_gain)
        self.reward_duck_dash = float(reward_duck_dash)
        self.reward_phase_bonus = float(reward_phase_bonus)
        self.still_penalty_unit = float(still_penalty_unit)

        # —— 动作 / 观测空间 —— #
        # L,R,Up,Down,Jump,Shoot,Dash,Lock,Special(K)
        self.action_space = spaces.MultiBinary(9)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.stack, self.H, self.W), dtype=np.uint8
        )

        # —— 相机 —— #
        self.cam = dxcam.create(output_color="BGR")
        # video_mode=True 只取最新帧，吞吐更高
        self.cam.start(target_fps=0, video_mode=True)
        time.sleep(0.15)

        # —— ROI —— #
        roi = json.load(open("cuphead_roi.json", "r", encoding="utf-8"))
        self.bx, self.by, self.bw, self.bh = roi["boss_hp_roi"]
        self.px, self.py, self.pw, self.ph = roi["player_hp_roi"]

        if "parry_roi" in roi:
            self.sx, self.sy, self.sw, self.sh = roi["parry_roi"]
            self.has_parry = True
        else:
            self.sx = self.sy = self.sw = self.sh = 0
            self.has_parry = False

        if "xcoord_roi" in roi:
            self.xx, self.xy, self.xw, self.xh = roi["xcoord_roi"]
            self.has_x = True
        else:
            self.xx = self.xy = self.xw = self.xh = 0
            self.has_x = False

        # —— 键位 —— #
        self.key_left   = "a"
        self.key_right  = "d"
        self.key_up     = "w"
        self.key_down   = "s"
        self.key_jump   = "space"
        self.key_shoot  = "j"
        self.key_dash   = "shift"        # 左Shift
        self.key_lock   = "rightshift"   # 右Shift（若你设置是右Shift）
        self.key_special= "k"            # 特殊（释放技能）
        self.key_reset  = "r"

        # 当前按下状态（避免重复发事件）
        self._held = {
            self.key_left: False, self.key_right: False,
            self.key_up: False,   self.key_down: False,
            self.key_jump: False, self.key_shoot: False,
            self.key_dash: False, self.key_lock: False,
            # K 为瞬时按键，不放入 _held 管理（直接 tap）
        }

        # —— 内部状态 —— #
        self.last_boss = (0.0, 1.0)
        self.stackbuf = None
        self._skip_reset_once = False
        self._step_count = 0

        # Parry / K 使用计数
        self.parry_last = 0        # 上一次读取到的 parry 计数
        self.parry_used = 0        # 本局内已消耗的 K 次数（≤ parry_last）
        # X 坐标
        self.x_last = None         # 上一次读取到的 X
        self.x_still_steps = 0     # 连续“几乎不动”的步数（用于叠加惩罚）
        # Boss 阶段标志
        self.in_phase2 = False     # 是否已进入 ≤450 的阶段（触发一次性奖励后置 True）

    # -------------- 帧与堆栈 -------------- #
    def _obs_from_frame(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return g

    def _stack_init(self, g):
        self.stackbuf = np.repeat(g[None, ...], self.stack, axis=0)

    def _stack_push(self, g):
        self.stackbuf = np.concatenate([self.stackbuf[1:], g[None, ...]], axis=0)

    # -------------- 键盘并发控制 -------------- #
    def _press(self, key):
        if key not in self._held:
            pag.keyDown(key)  # 即使未托管，也允许手动按下
            return
        if not self._held[key]:
            pag.keyDown(key)
            self._held[key] = True

    def _release(self, key):
        if key not in self._held:
            pag.keyUp(key)
            return
        if self._held[key]:
            pag.keyUp(key)
            self._held[key] = False

    def _tap(self, key, dur=0.04):
        """瞬时按键（用于 K）。"""
        try:
            pag.keyDown(key)
            time.sleep(max(0.01, float(dur)))
            pag.keyUp(key)
        except Exception:
            pass

    def _apply_action(self, mask, allow_k=True):
        """
        mask: ndarray/sequence 长度9, 0/1
        顺序: [L,R,Up,Down,Jump,Shoot,Dash,Lock,Special(K)]
        冲突处理：若 L/R 或 Up/Down 同时为1，均不按（避免抖动）
        K：瞬时按键（不维持按下）；且受 parry 限制
        """
        if len(mask) < 9:
            # 兼容旧 8 位动作，自动补 0
            mask = list(mask) + [0]
        L,R,Up,Down,Jump,Shoot,Dash,Lock,Special = [int(x) for x in mask]

        # 互斥轴
        if L and R:   L, R   = 0, 0
        if Up and Down: Up, Down = 0, 0

        # 并发维持键（不含 K）
        target_state = {
            self.key_left:  bool(L),
            self.key_right: bool(R),
            self.key_up:    bool(Up),
            self.key_down:  bool(Down),
            self.key_jump:  bool(Jump),
            self.key_shoot: bool(Shoot),
            self.key_dash:  bool(Dash),
            self.key_lock:  bool(Lock),
        }
        for k, want in target_state.items():
            if want and not self._held[k]:
                self._press(k)
            elif (not want) and self._held[k]:
                self._release(k)

        # 处理 K（瞬时、受 Parry 限制）
        if allow_k and Special:
            if self.parry_last is not None and (self.parry_used < int(self.parry_last)):
                self._tap(self.key_special, dur=0.04)
                self.parry_used += 1
            # 否则忽略 K，不做惩罚（避免噪声误导）

    def _release_all(self):
        for k in list(self._held.keys()):
            self._release(k)

    # -------------- OCR 读取 -------------- #
    def _read_boss_hp(self, frame):
        roi = frame[self.by:self.by+self.bh, self.bx:self.bx+self.bw]
        raw = ocr_number_block(roi)
        return parse_boss_hp(raw)

    def _read_player_hp(self, frame):
        roi = frame[self.py:self.py+self.ph, self.px:self.px+self.pw]
        if is_dead_fast(roi, thresh=0.72) is True:
            return 0, "DEAD_fast"
        if is_hp_badge_red(roi, frac_thresh=0.22):
            return 1, "red_bg"
        p, _ = read_player_digit_only(roi)
        if p is not None:
            return p, "ocr"
        bin_img = extract_digit_bin(roi, right_ratio=0.50)
        if is_digit_one_by_shape(bin_img):
            return 1, "shape_1"
        ok, _ = detect_dead_text(roi)
        if ok:
            return 0, "DEAD_ocr"
        return None, "none"

    def _read_parry(self, frame):
        if not self.has_parry:
            return None, "parry_na"
        roi = frame[self.sy:self.sy+self.sh, self.sx:self.sx+self.sw]
        val, tag = read_parry_count_only(roi)  # 内部含缓存/限频
        # read_parry_count_only 返回 int 或 None
        return (int(val) if isinstance(val, (int, np.integer)) else val), tag

    def _read_xcoord(self, frame):
        if not self.has_x:
            return None, "x_na"
        roi = frame[self.xy:self.xy+self.xh, self.xx:self.xx+self.xw]
        val, tag = read_xcoord_only(roi)       # 内部含缓存/限频
        # 返回 float 或 None
        return (float(val) if isinstance(val, (int, float, np.floating)) else val), tag

    # -------------- Gym API -------------- #
    def reset(self, *, seed=None, options=None):
        self._release_all()

        if not self._skip_reset_once:
            pag.press(self.key_reset)
            time.sleep(0.85)   # 关卡加载；按你的机器调整
        else:
            self._skip_reset_once = False
            time.sleep(0.35)

        frame = grab(self.cam)
        g = self._obs_from_frame(frame)
        self._stack_init(g)

        c, m = self._read_boss_hp(frame)
        if c is None or m is None or m <= 0:
            c, m = 0.0, 1.0
        self.last_boss = (c, m)
        self._step_count = 0

        # 重置 Parry / K / X 状态
        self.parry_last = 0
        self.parry_used = 0
        self.x_last = None
        self.x_still_steps = 0
        self.in_phase2 = (c is not None and c <= 450.0)

        return self.stackbuf.copy(), {}

    def step(self, action):
        t0 = time.perf_counter()

        # 1) 并发按键（含 K，受 parry_used 限制）
        self._apply_action(action, allow_k=True)

        # 2) 等待到决策周期
        elapsed = time.perf_counter() - t0
        remain = self.dt - elapsed
        if remain > 0:
            time.sleep(remain)

        # 3) 取帧 + 观测
        frame = grab(self.cam)
        g = self._obs_from_frame(frame)
        self._stack_push(g)

        # 4) 降采样读取
        read_hp_now    = (self._step_count % self.hp_every_n    == 0)
        read_parry_now = (self._step_count % self.parry_every_n == 0)
        read_x_now     = (self._step_count % self.x_every_n     == 0)

        # HP
        if read_hp_now:
            boss_c, boss_m = self._read_boss_hp(frame)
            if boss_c is None or boss_m is None or boss_m <= 0:
                boss_c, boss_m = self.last_boss
            ply_hp, _ = self._read_player_hp(frame)
        else:
            boss_c, boss_m = self.last_boss
            ply_hp = None  # 未更新

        # Parry
        if read_parry_now:
            parry_cur, parry_tag = self._read_parry(frame)
        else:
            parry_cur, parry_tag = self.parry_last, "parry_skip"

        # X
        if read_x_now:
            x_cur, x_tag = self._read_xcoord(frame)
        else:
            x_cur, x_tag = self.x_last, "x_skip"

        # 5) 奖励
        last_c, last_m = self.last_boss
        boss_damage = max(0.0, (last_c - boss_c)) if (boss_c is not None and last_c is not None) else 0.0
        reward = 1.0 * boss_damage - 0.002  # 你的基础 shaping

        # 5.1 Parry 增加（格挡成功）奖励
        if read_parry_now and (parry_cur is not None) and (parry_cur > self.parry_last):
            reward += self.reward_parry_gain * (parry_cur - self.parry_last)

        # 5.2 鼓励下蹲+冲刺（s + 左Shift）
        if self._held.get(self.key_down, False) and self._held.get(self.key_dash, False):
            reward += self.reward_duck_dash

        # 5.3 Boss 进入 ≤450 一次性奖励（只触发一次）
        if (boss_c is not None):
            if (not self.in_phase2) and (boss_c <= 450.0):
                reward += self.reward_phase_bonus
                self.in_phase2 = True

        # 5.4 X 不动惩罚（仅当这帧有新 X）
        if read_x_now and (x_cur is not None) and (self.x_last is not None):
            dx = abs(float(x_cur) - float(self.x_last))
            if dx < self.x_still_thresh:
                self.x_still_steps = min(self.x_still_steps + 1, 5)
                reward -= self.still_penalty_unit * self.x_still_steps
            else:
                self.x_still_steps = 0

        # 6) 终止条件
        win  = (boss_c is not None and boss_c <= 0.0)
        dead = (ply_hp == 0) if read_hp_now else False
        done = bool(win or dead)

        # 7) 快速重启
        if self.auto_restart and done:
            self._release_all()
            wait = 0.15 if dead else 0.35
            pag.press(self.key_reset)
            self._skip_reset_once = True
            time.sleep(wait)

        # —— 状态更新 —— #
        self.last_boss = (boss_c, boss_m)
        if read_parry_now and (parry_cur is not None):
            # 注意：parry_used 不回退，只在 reset 时清零
            self.parry_last = int(parry_cur)
            # 若 OCR 抖动导致 parry_last < parry_used，做保护
            if self.parry_last < self.parry_used:
                self.parry_last = self.parry_used
        if read_x_now and (x_cur is not None):
            self.x_last = float(x_cur)

        self._step_count += 1

        info = {
            "boss_hp": boss_c, "boss_max": boss_m,
            "player_hp": ply_hp, "win": win, "dead": dead,
            "hp_read": read_hp_now,
            "parry": self.parry_last, "parry_used": self.parry_used, "parry_tag": parry_tag,
            "phase2": self.in_phase2,
            "x": self.x_last, "x_read": read_x_now, "x_tag": x_tag,
            "x_still_steps": self.x_still_steps,
        }
        if self.debug:
            print(info)

        return self.stackbuf.copy(), reward, done, False, info

    def render(self):
        # 为了性能，训练时建议不要显示
        pass

    def close(self):
        self._release_all()
        if self.cam:
            self.cam.stop()
        cv2.destroyAllWindows()
