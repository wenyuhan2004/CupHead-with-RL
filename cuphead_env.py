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
    detect_dead_text, is_hp_badge_red, is_dead_fast
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
    - 动作空间: MultiBinary(8) -> [L,R,Up,Down,Jump,Shoot,Dash,Lock]
    - 观测: (stack, H, W) 灰度
    - 降采样读取HP (hp_every_n) 以减少OCR开销
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        decision_fps: int = 15,
        frame_size=(96, 96),
        stack: int = 4,
        debug: bool = False,
        auto_restart: bool = True,
        hp_every_n: int = 2,      # 每 N 步读取一次 HP（>=1）
    ):
        super().__init__()
        self.debug = debug
        self.auto_restart = auto_restart
        self.dt = 1.0 / max(1, decision_fps)
        self.W, self.H, self.stack = frame_size[0], frame_size[1], stack
        self.hp_every_n = max(1, hp_every_n)

        # —— 动作 / 观测空间 —— #
        # L,R,Up,Down,Jump,Shoot,Dash,Lock (8 bits)
        self.action_space = spaces.MultiBinary(8)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.stack, self.H, self.W), dtype=np.uint8
        )

        # —— 相机 —— #
        self.cam = dxcam.create(output_color="BGR")
        # 尽可能快，video_mode=True 只取最新帧
        self.cam.start(target_fps=0, video_mode=True)
        time.sleep(0.15)

        # —— ROI —— #
        roi = json.load(open("cuphead_roi.json", "r", encoding="utf-8"))
        self.bx, self.by, self.bw, self.bh = roi["boss_hp_roi"]
        self.px, self.py, self.pw, self.ph = roi["player_hp_roi"]

        # —— 键位 —— #
        self.key_left   = "a"
        self.key_right  = "d"
        self.key_up     = "w"
        self.key_down   = "s"
        self.key_jump   = "space"
        self.key_shoot  = "j"
        self.key_dash   = "shift"        # 左Shift
        self.key_lock   = "rightshift"   # 右Shift（若你设置是右Shift）
        self.key_reset  = "r"

        # 当前按下状态（避免重复发事件）
        self._held = {
            self.key_left: False, self.key_right: False,
            self.key_up: False,   self.key_down: False,
            self.key_jump: False, self.key_shoot: False,
            self.key_dash: False, self.key_lock: False,
        }

        # —— 内部状态 —— #
        self.last_boss = (0.0, 1.0)
        self.stackbuf = None
        self._skip_reset_once = False
        self._step_count = 0

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
        if not self._held[key]:
            pag.keyDown(key)
            self._held[key] = True

    def _release(self, key):
        if self._held[key]:
            pag.keyUp(key)
            self._held[key] = False

    def _apply_action(self, mask):
        """
        mask: ndarray/sequence 长度8, 0/1
        顺序: [L,R,Up,Down,Jump,Shoot,Dash,Lock]
        冲突处理：若 L/R 或 Up/Down 同时为1，均不按（避免抖动）
        """
        L,R,Up,Down,Jump,Shoot,Dash,Lock = [int(x) for x in mask]

        # 互斥轴
        if L and R:   L, R   = 0, 0
        if Up and Down: Up, Down = 0, 0

        # 按需变更状态（只对状态变化派发事件）
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

    def _release_all(self):
        for k in list(self._held.keys()):
            self._release(k)

    # -------------- HP 读取（降采样） -------------- #
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
        return self.stackbuf.copy(), {}

    def step(self, action):
        t0 = time.perf_counter()

        # 1) 并发按键
        self._apply_action(action)

        # 2) 补齐决策周期
        elapsed = time.perf_counter() - t0
        remain = self.dt - elapsed
        if remain > 0:
            time.sleep(remain)

        # 3) 取帧 + 观测
        frame = grab(self.cam)
        g = self._obs_from_frame(frame)
        self._stack_push(g)

        # 4) 降采样读取 HP
        read_hp_now = (self._step_count % self.hp_every_n == 0)
        if read_hp_now:
            boss_c, boss_m = self._read_boss_hp(frame)
            if boss_c is None or boss_m is None or boss_m <= 0:
                boss_c, boss_m = self.last_boss
            ply_hp, _ = self._read_player_hp(frame)
        else:
            # 不读取时复用上次（更快）
            boss_c, boss_m = self.last_boss
            ply_hp = None  # 未更新

        # 5) 奖励
        last_c, last_m = self.last_boss
        delta = max(0.0, (last_c - boss_c)) if (boss_c is not None and last_c is not None) else 0.0
        reward = 1.0 * delta - 0.002

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

        self.last_boss = (boss_c, boss_m)
        self._step_count += 1

        info = {
            "boss_hp": boss_c, "boss_max": boss_m,
            "player_hp": ply_hp, "win": win, "dead": dead,
            "hp_read": read_hp_now,
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
