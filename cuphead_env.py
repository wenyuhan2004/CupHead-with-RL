# cuphead_env.py
import time
import json
import csv
import threading
import queue
import collections
import os
import sys
import cv2
import numpy as np
import dxcam
import pyautogui as pag
import gymnasium as gym
from gymnasium import spaces
# 确保能找到上级目录的辅助模块（如 read_hp.py）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from cuphead_dqn.test_cuphead_hp import (
        read_player_hp as test_read_player_hp,
        read_boss_hp as test_read_boss_hp,
        read_player_x as test_read_player_x,
        read_skill_points as test_read_skill_points,
        open_cuphead as test_open_cuphead,
        enum_module as test_enum_module,
        update_hp_with_debounce as test_update_hp_with_debounce,
    )
except Exception:
    try:
        from test_cuphead_hp import (  # type: ignore
            read_player_hp as test_read_player_hp,
            read_boss_hp as test_read_boss_hp,
            read_player_x as test_read_player_x,
            read_skill_points as test_read_skill_points,
            open_cuphead as test_open_cuphead,
            enum_module as test_enum_module,
            update_hp_with_debounce as test_update_hp_with_debounce,
        )
    except Exception:  # pragma: no cover - optional dependency
        test_read_player_hp = None
        test_read_boss_hp = None
        test_read_player_x = None
        test_open_cuphead = None
        test_enum_module = None
        test_update_hp_with_debounce = None

try:
    # 优先使用本目录内的内存读取实现
    from cuphead_dqn.cuphead_memory import CupheadMemoryReader  # type: ignore
except Exception:
    try:
        from cuphead_memory import CupheadMemoryReader  # type: ignore
    except Exception:  # pragma: no cover
        CupheadMemoryReader = None

pag.FAILSAFE = False
pag.PAUSE = 0  # 移除pyautogui的自动延迟

# 本地抓帧（替代 OCR 模块的 grab），其余 OCR 功能已停用
def grab(cam):
    frame = cam.get_latest_frame()
    while frame is None:
        frame = cam.get_latest_frame()
        time.sleep(0.005)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame

def focus_cuphead_window():
    """尝试激活 Cuphead 窗口"""
    try:
        import pygetwindow as gw
        wins = [w for w in gw.getAllTitles() if w and 'cuphead' in w.lower()]
        if wins:
            w = gw.getWindowsWithTitle(wins[0])[0]
            w.activate(); w.restore()
            # 提高内存读取频率（约30FPS），便于每步都拿到最新缓存
            time.sleep(1.0 / 30.0)
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
    Cuphead 强化学习环境（多键并发 + OCR 感知）
    - 动作空间: MultiBinary(9) -> [L,R,Up,Down,Jump,Shoot,Dash,Lock,Special(K)]
    - 支持方向互斥、tap 键、持续键与技能触发
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        decision_fps: int = 12,
        frame_size=(192, 108),
        stack: int = 4,
        debug: bool = False,
        auto_restart: bool = True,
        hp_every_n: int = 1,
        parry_every_n: int = 9999,
        x_every_n: int = 0,
        reward_parry_gain: float = 0.30,
        reward_duck_dash: float = 0.01,
        # 新增奖励调参项：
        reward_boss_damage_mul: float = 0.1,
        reward_progress_bonus: float = 0.004,
        reward_player_damage_penalty: float = 10.0,
        reward_skill_use: float = 0.5,
        reward_shoot_hold: float = 0.0,
        reward_shoot_hit: float = 0.3,
        reward_dash_safe: float = 0.05,
        async_ocr: bool = False,
        ocr_max_delay: float = 0.3,
        use_memory_hp: bool = True,
        player_hp_max: int = 4,
        warmup_steps: int = 5,
        min_episode_steps: int = 25,
        boss_max_drop: float = 120.0,
        hp_valid_age: float = 1.0,
        step_log_path=None,
        # 位置相关
        x_min: float = -615.0,
        x_max: float = 615.0,
        x_margin: float = 5.0,
        reward_wall_penalty: float = 5.0,
        dir_hold_limit: int = 20,
    ):
        super().__init__()
        self.debug = debug
        self.auto_restart = auto_restart
        # HK 风格：若 decision_fps<=0 则不做固定步频控制
        self.dt = 1.0 / max(1, decision_fps) if decision_fps and decision_fps > 0 else 0.0
        self.W, self.H, self.stack = frame_size[0], frame_size[1], stack

        self.hp_every_n = max(1, hp_every_n)
        self.parry_every_n = max(1, parry_every_n)
        self.x_every_n = max(0, x_every_n)

        self.reward_parry_gain = float(reward_parry_gain)
        self.reward_duck_dash = float(reward_duck_dash)
        # 新增参数
        self.reward_boss_damage_mul = float(reward_boss_damage_mul)
        self.reward_progress_bonus = float(reward_progress_bonus)
        self.reward_player_damage_penalty = float(reward_player_damage_penalty)
        self.reward_skill_use = float(reward_skill_use)
        self.reward_shoot_hold = float(reward_shoot_hold)
        self.reward_shoot_hit = float(reward_shoot_hit)
        self.reward_dash_safe = float(reward_dash_safe)
        # 完全弃用 OCR
        self.async_ocr = False
        self.ocr_max_delay = float(max(0.05, ocr_max_delay))
        self.use_memory_hp = bool(use_memory_hp)
        self.player_hp_max = int(max(1, player_hp_max))
        self.warmup_steps = int(max(0, warmup_steps))
        self.min_episode_steps = int(max(0, min_episode_steps))
        self.boss_max_drop = float(max(0.0, boss_max_drop))
        self.hp_valid_age = float(max(0.05, hp_valid_age))
        self.step_log_path = step_log_path
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.x_margin = float(max(0.0, x_margin))
        self.reward_wall_penalty = float(reward_wall_penalty)
        self.dir_hold_limit = int(max(1, dir_hold_limit))

        # —— 动作 / 观测空间 —— #
        # 扩展为10维：原9维 + 跳跃强度
        self.action_space = spaces.MultiBinary(10)  # [L,R,Up,Down,Jump,Shoot,Dash,Lock,Special,JumpHold]
        # RGB + 堆叠：shape = (stack, 3, H, W)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.stack, 3, self.H, self.W), dtype=np.uint8
        )

        # —— 相机 —— #
        self.cam = dxcam.create(output_color="BGR")
        # 限帧以降低抓帧开销
        self.cam.start(target_fps=20, video_mode=True)
        time.sleep(0.15)

        # —— ROI —— #
        # OCR 已禁用，这里的 ROI 仅占位，无需文件
        self.bx = self.by = self.bw = self.bh = 0
        self.px = self.py = self.pw = self.ph = 0
        self.has_parry = False
        self.sx = self.sy = self.sw = self.sh = 0
        self.has_x = False
        self.xx = self.xy = self.xw = self.xh = 0

        # —— 键位 —— #
        self.key_left   = "a"
        self.key_right  = "d"
        self.key_up     = "w"
        self.key_down   = "s"
        self.key_jump   = "space"
        self.key_shoot  = "j"
        self.key_dash   = "shift"        # 左Shift
        self.key_lock   = "rightshift"   # 右Shift
        self.key_special= "k"            # 特殊技能（当前禁用）
        self.key_reset  = "r"

        # 当前按下状态
        self._held = {
            self.key_left: False, self.key_right: False,
            self.key_up: False,
            self.key_shoot: False, self.key_lock: False,
        }

        # —— 状态变量 —— #
        self._boss_default = (1000.0, 1000.0)
        self.last_boss = self._boss_default
        self.stackbuf = None
        self._skip_reset_once = False
        self._pending_reset = False
        self._step_count = 0

        self.parry_last = 0
        self.parry_used = 0
        self.x_last = None
        self.facing_dir = None  # 不再记录朝向
        self._recent_crouch_dash = False
        # 记录玩家血量用于检测受伤惩罚
        self.last_player_hp = self.player_hp_max
        # OCR 异步支持（已弃用）
        self._ocr_queue = None
        self._ocr_lock = None
        self._ocr_result = None
        self._ocr_stop = threading.Event()
        self._ocr_thread = None
        # Frame buffer thread
        self._frame_queue = collections.deque(maxlen=2)
        self._frame_lock = threading.Lock()
        self._frame_stop = threading.Event()
        self._frame_thread = threading.Thread(
            target=self._frame_worker, name="CupheadFrame", daemon=True
        )
        self._frame_thread.start()
        self.mem_reader = None
        self._mem_data = {"boss": None, "player_stable": None, "player_raw": None, "skill": None, "timestamp": 0.0}
        self._mem_lock = threading.Lock()
        self._mem_stop = threading.Event()
        self._mem_thread = None
        # 取消按键队列，直接主线程发送
        self._player_hp_state = {
            "stable": self.player_hp_max,
            "candidate": None,
            "count": 0,
        }
        self._player_hp_state_test = {
            "stable": None,
            "candidate": None,
            "count": 0,
        }
        self._skill_points = 0.0
        self._last_player_print = None
        self._dash_triggered = False
        self._last_player_hp_raw = None
        self._step_log_fp = None
        self._step_log_writer = None
        focus_cuphead_window()  # 初始化时也尝试激活窗口
        # FPS 统计
        self._fps_counter = 0  # 跨 episode 统计总步数
        self._fps_last_ts = time.time()
        self._last_fps = 0.0
        # 直接使用 test_* 读取，跳过内存线程
        self.use_memory_hp = False
        self._hp_hproc = None
        self._hp_mono_base = None
        self._hp_unity_base = None
        if test_read_player_hp and test_open_cuphead and test_enum_module:
            try:
                self._hp_hproc, _pid = test_open_cuphead()
                self._hp_mono_base = test_enum_module(self._hp_hproc, _pid, "mono.dll")
                try:
                    self._hp_unity_base = test_enum_module(self._hp_hproc, _pid, "UnityPlayer.dll")
                except Exception:
                    self._hp_unity_base = None
                print("[INFO] test_cuphead_hp handle ready for player HP")
            except Exception as exc:
                self._hp_hproc = None
                self._hp_mono_base = None
                self._hp_unity_base = None
                print(f"[WARN] test_cuphead_hp init failed: {exc}")
        # 关闭步级日志，需时可自行开启
        self._step_log_fp = None
        self._step_log_writer = None

    # ====== 帧堆栈处理 ====== #
    def _obs_from_frame(self, frame):
        # 保持 RGB，转为 (3, H, W)
        g = cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
        g = np.transpose(g, (2, 0, 1))
        return g

    def _stack_init(self, g):
        self.stackbuf = np.repeat(g[None, ...], self.stack, axis=0)  # (stack, 3, H, W)

    def _stack_push(self, g):
        self.stackbuf = np.concatenate([self.stackbuf[1:], g[None, ...]], axis=0)
    def _frame_worker(self):
        while not self._frame_stop.is_set():
            frame = grab(self.cam)
            with self._frame_lock:
                self._frame_queue.append(frame)
            time.sleep(0.001)

    def _get_latest_frame(self):
        # 去掉截屏缓存，直接抓取当前帧
        return grab(self.cam)

    def _memory_worker(self):
        if not self.mem_reader:
            return
        while not self._mem_stop.is_set():
            boss = None
            player_raw = None
            x_raw = None
            skill_raw = None
            try:
                boss = self.mem_reader.read_boss_hp()
            except Exception:
                boss = None
            try:
                player_raw = self.mem_reader.read_player_hp()
            except Exception:
                player_raw = None
            if (player_raw is None) and self._hp_hproc and test_read_player_hp:
                try:
                    player_raw = test_read_player_hp(self._hp_hproc, self._hp_mono_base)
                except Exception:
                    player_raw = None
            if (boss is None) and self._hp_hproc and self._hp_mono_base and test_read_boss_hp:
                try:
                    boss = test_read_boss_hp(self._hp_hproc, self._hp_mono_base)
                except Exception:
                    boss = None
            if self._hp_hproc and self._hp_unity_base and test_read_player_x:
                try:
                    x_raw = test_read_player_x(self._hp_hproc, self._hp_unity_base)
                except Exception:
                    x_raw = None
            if self._hp_hproc and self._hp_mono_base and test_read_skill_points:
                try:
                    skill_raw = test_read_skill_points(self._hp_hproc, self._hp_mono_base)
                except Exception:
                    skill_raw = None
            # 使用 test_cuphead_hp 的去抖逻辑，允许原样保留异常值以便排查
            if test_update_hp_with_debounce:
                st = self._player_hp_state_test
                stable_hp, cand_hp, cand_cnt = test_update_hp_with_debounce(
                    st.get("stable"),
                    st.get("candidate"),
                    st.get("count", 0),
                    player_raw,
                    min_val=-1e6,
                    max_val=1e6,
                    threshold=1,
                )
                self._player_hp_state_test.update(
                    stable=stable_hp,
                    candidate=cand_hp,
                    count=cand_cnt,
                )
                player_stable = stable_hp
            else:
                player_stable = self._debounce_player_hp(player_raw, threshold=1)
            # 写入缓存均使用稳定值
            # 缓存不再使用，仅保留为兼容
            self._last_player_print = player_stable
            time.sleep(0.2)

    def _enqueue_ocr_frame(self, frame):
        return None

    def _get_async_ocr(self):
        return None

    def _ocr_worker(self):
        return None

    # ====== 按键操作 ====== #
    def _press(self, key):
        if key not in self._held:
            pag.keyDown(key)
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

    def _tap(self, key, dur=0.06):
        try:
            pag.keyDown(key)
            time.sleep(dur)
            pag.keyUp(key)
        except Exception:
            pass

    def _tap_combo(self, keys, dur=0.10):
        try:
            for k in keys:
                pag.keyDown(k)
            time.sleep(dur)
            for k in reversed(keys):
                pag.keyUp(k)
        except Exception:
            pass

    def _release_all(self):
        for k in list(self._held.keys()):
            self._release(k)
        # 额外保险：把常用按键都抬起，防止系统漏松键
        for k in [self.key_left, self.key_right, self.key_up, self.key_down, self.key_dash, self.key_jump, self.key_lock, self.key_shoot]:
            try:
                pag.keyUp(k)
            except Exception:
                pass

    def _release_safety(self, L, R, Up, Down, Dash, Jump, Lock, Shoot):
        """防止长时间运行后键位卡住：凡本步不需要的键一律抬起。"""
        want = {
            self.key_left:  L,
            self.key_right: R,
            self.key_up:    Up,
            self.key_down:  Down,
            self.key_dash:  Dash,
            self.key_jump:  Jump,
            self.key_lock:  Lock,
            self.key_shoot: Shoot,
        }
        for k, flag in want.items():
            if not flag:
                try:
                    pag.keyUp(k)
                except Exception:
                    pass

    def _add_safe_window(self, dur: float = 1.0):
        exp = time.time() + max(0.0, dur)
        self._pending_safe.append({"expire": exp, "failed": False})
        if len(self._pending_safe) > 10:
            self._pending_safe = self._pending_safe[-10:]
        # 保留函数以兼容，但不再使用

    # ====== 核心输入逻辑 ====== #
    def _apply_action(self, mask, allow_k=True):
        if len(mask) < 10:
            mask = list(mask) + [0] * (10 - len(mask))
        L, R, Up, Down, Jump, Shoot, Dash, Lock, Special, JumpHold = [int(x) for x in mask]
        # 攻击键改为两步一按（1/0交替）
        if not hasattr(self, "_shoot_phase"):
            self._shoot_phase = False
        if self._shoot_phase:
            Shoot = 1
        else:
            Shoot = 0
        self._shoot_phase = not self._shoot_phase

        if not hasattr(self, "_prev_action"):
            self._prev_action = [0] * 10

        # --- 方向互斥 (L/R, Up/Down) ---
        if L and R:
            # 同时按左右，置为空方向避免卡死
            L = R = 0
        if Up and Down:
            if self._prev_action[2] == 1 and self._prev_action[3] == 0:
                Up = 0
            elif self._prev_action[3] == 1 and self._prev_action[2] == 0:
                Down = 0
            else:
                Up = Down = 0

        # --- 持续维持型键 ---
        # 按键顺序：先左右方向，再其他（避免后续动作覆盖方向未按下）
        ordered_targets = [
            (self.key_left, bool(L)),
            (self.key_right, bool(R)),
            (self.key_up, bool(Up)),
            (self.key_shoot, bool(Shoot)),  # J持续按
            (self.key_lock, bool(Lock)),
        ]
        for k, want in ordered_targets:
            if want and not self._held[k]:
                self._press(k)
            elif (not want) and self._held[k]:
                self._release(k)

        # --- 方向卡死防抖：同向持续过长则强制松开 ---
        dir_cmd = -1 if L and not R else (1 if R and not L else 0)
        if dir_cmd == self._dir_hold_dir and dir_cmd != 0:
            self._dir_hold_steps += 1
        else:
            self._dir_hold_dir = dir_cmd
            self._dir_hold_steps = 0
        if self._dir_hold_steps >= self.dir_hold_limit:
            self._release(self.key_left)
            self._release(self.key_right)
            L = R = 0
            self._dir_hold_steps = 0
            self._dir_hold_dir = 0

        # --- Duck Dash 触发（S自动触发Shift）---
        down_trigger = bool(Down)
        # 仅在 Dash=1 时触发下蹲冲刺，并保证先按方向键再按 Shift
        if down_trigger and Dash and not self._prev_action[3]:
            self._tap_combo([self.key_down, self.key_dash], dur=0.10)  # 适中按压
            self._recent_crouch_dash = True
        # 不再添加安全窗奖励
        Down = 0

        # --- 智能跳跃系统 ---
        if Jump and not self._prev_action[4]:
            # 根据JumpHold决定跳跃时长：短跳/长跳
            jump_duration = 0.20 if JumpHold else 0.12
            self._tap(self.key_jump, dur=jump_duration)
            Jump = 0
        if Dash and not self._prev_action[6]:
            self._tap(self.key_dash, dur=0.06)  # 适中 dash 按压
            self._recent_crouch_dash = bool(self._held.get(self.key_down, False))
            self._dash_triggered = True
            Dash = 0

        # --- 技能键 ---（禁用技能逻辑）
        Special = 0

        self._prev_action = [L, R, Up, down_trigger, Jump, Shoot, Dash, Lock, Special, JumpHold]
        # 防卡键保险：本步无需的键立即抬起
        self._release_safety(L, R, Up, False, False, False, Lock, Shoot)

    # ====== OCR读取 ====== #
    def _read_boss_hp(self, frame):
        # OCR 已停用
        return None, "ocr_disabled"

    def _read_player_hp(self, frame):
        # OCR 已停用
        return None, "ocr_disabled"

    def _read_parry(self, frame):
        # OCR 已停用
        return None, "ocr_disabled"

    def _valid_async_data(self):
        return None

    def _get_mem_snapshot(self):
        return None  # 不再使用内存缓存

    def _debounce_player_hp(self, raw_hp, min_val=0, max_val=None, threshold=3):
        if max_val is None:
            max_val = self.player_hp_max
        state = self._player_hp_state
        stable = state.get("stable")
        cand = state.get("candidate")
        cnt = state.get("count", 0)
        if raw_hp is None or not (min_val <= raw_hp <= max_val):
            state.update(stable=stable, candidate=cand, count=cnt)
            return stable
        if stable is None:
            stable = raw_hp
            cand = None
            cnt = 0
        elif raw_hp == stable:
            cand = None
            cnt = 0
        else:
            if cand is None or raw_hp != cand:
                cand = raw_hp
                cnt = 1
            else:
                cnt += 1
                if cnt >= threshold:
                    stable = cand
                    cand = None
                    cnt = 0
        state.update(stable=stable, candidate=cand, count=cnt)
        return stable

    def _fetch_boss_hp(self, frame):
        # 直接从内存读取，不使用缓存/线程
        boss_val = None
        if self._hp_hproc and self._hp_mono_base and test_read_boss_hp:
            try:
                boss_val = test_read_boss_hp(self._hp_hproc, self._hp_mono_base)
            except Exception:
                boss_val = None
        if boss_val is not None:
            boss_m = max(boss_val, self.last_boss[1]) if self.last_boss[1] is not None else boss_val
            return float(boss_val), float(boss_m)
        return self.last_boss

    def _fetch_player_hp(self, frame):
        val = None
        if self._hp_hproc and test_read_player_hp:
            try:
                val = test_read_player_hp(self._hp_hproc, self._hp_mono_base)
            except Exception:
                val = None
        if val is not None and 0 <= val <= self.player_hp_max:
            return val
        # 异常值视为0（触发重开）
        return 0

    def _fetch_parry(self, frame):
        return None

    def _fetch_xcoord(self, frame):  # legacy stub for compatibility
        if self._hp_hproc and self._hp_unity_base and test_read_player_x:
            try:
                return float(test_read_player_x(self._hp_hproc, self._hp_unity_base))
            except Exception:
                return None
        return None


    # ====== Gym API ====== #
    def reset(self, *, seed=None, options=None):
        self._release_all()
        pag.press(self.key_reset)
        time.sleep(0.3)
        self._skip_reset_once = False
        self._pending_reset = False

        frame = self._get_latest_frame()
        g = self._obs_from_frame(frame)
        self._stack_init(g)
        c, m = self._fetch_boss_hp(frame)
        if c is None or m is None or m <= 0:
            c, m = self._boss_default
        else:
            c = float(c)
            m = float(m)
        self.last_boss = (c, m)
        self._step_count = 0
        self.parry_last = 0
        self.parry_used = 0
        self.x_last = None
        self.facing_dir = None
        self._recent_crouch_dash = False
        self._dash_triggered = False
        self.last_player_hp = self.player_hp_max
        self._last_player_hp_raw = None
        self._pending_safe = []
        self._dir_hold_steps = 0
        self._dir_hold_dir = 0
        focus_cuphead_window()  # 保持窗口前台，避免按键丢失
        return self.stackbuf.copy(), {}

    def step(self, action):
        t0 = time.perf_counter()
        # 记录使用技能前的计数，以便在后续奖励中判断是否消耗技能点
        parry_used_before = self.parry_used
        self._apply_action(action, allow_k=True)
        # 移除额外等待，按键线程已取消，尽量提升循环速度

        frame = self._get_latest_frame()
        g = self._obs_from_frame(frame)
        self._stack_push(g)

        read_parry_now = (self._step_count % self.parry_every_n == 0)
        read_x_now = (self.x_every_n > 0) and (self._step_count % self.x_every_n == 0)

        if (self._step_count % 3) == 0:
            boss_c, boss_m = self._fetch_boss_hp(frame)
            if boss_c is None or boss_m is None or boss_m <= 0:
                boss_c, boss_m = self.last_boss
        else:
            boss_c, boss_m = self.last_boss
        boss_c = float(boss_c)
        boss_m = float(boss_m)
        # 持续打印 Boss HP

        raw_hp = self._fetch_player_hp(frame)
        self._last_player_hp_raw = raw_hp
        if raw_hp is None:
            ply_hp = self.last_player_hp
        else:
            if self.last_player_hp is not None and raw_hp < self.last_player_hp:
                ply_hp = max(self.last_player_hp - 1, raw_hp)
            elif self.last_player_hp is not None and raw_hp > self.last_player_hp:
                ply_hp = self.last_player_hp
            else:
                ply_hp = raw_hp


        parry_cur = 0

        # 当前动作左右键状态（用本步动作，避免方向判断反转）
        act_mask = list(action) if isinstance(action, (list, tuple)) else np.array(action).tolist()
        if len(act_mask) < 7:
            act_mask = act_mask + [0] * (7 - len(act_mask))
        cur_left, cur_right = bool(int(act_mask[0])), bool(int(act_mask[1]))
        cur_dash = bool(int(act_mask[6]))

        if read_x_now:
            x_cur = self._fetch_xcoord(frame)
        else:
            x_cur = self.x_last
        # 技能点（原始，直接内存读取）
        skill_pts = None
        if self._hp_hproc and self._hp_mono_base and test_read_skill_points:
            try:
                skill_pts = test_read_skill_points(self._hp_hproc, self._hp_mono_base)
            except Exception:
                skill_pts = None
        if skill_pts is not None:
            self._skill_points = skill_pts
        # 即时血量差奖励：Boss 掉血正向，玩家掉血负向
        last_c, last_m = self.last_boss
        boss_drop = max(0.0, (last_c - boss_c))
        boss_term = boss_drop / 9.0  # HK 风格
        hp_term = 0.0
        if ply_hp is not None and self.last_player_hp is not None:
            hp_drop = max(0.0, self.last_player_hp - ply_hp)
            hp_term = -11.0 * hp_drop
        reward = boss_term + hp_term
        # 边界推进惩罚/奖励：在边缘向外走扣分，向内走奖励
        if x_cur is not None:
            at_left_edge = (x_cur <= self.x_min + self.x_margin)
            at_right_edge = (x_cur >= self.x_max - self.x_margin)
            if at_left_edge:
                if cur_left:
                    reward -= self.reward_wall_penalty
                    print(0)
                if cur_right:
                    reward += 2.0
                    print(1)
            if at_right_edge:
                if cur_right:
                    reward -= self.reward_wall_penalty
                    print(0)
                if cur_left:
                    reward += 2.0
                    print(1)
        if ply_hp is not None:
            self.last_player_hp = ply_hp

        # 提前结束以避免击杀后无法重开：Boss 血量低于 50 即视为胜利
        win = (boss_c <= 50.0)
        if ply_hp is None:
            ply_hp = self.last_player_hp
        dead = False
        if ply_hp is not None:
            dead = (ply_hp <= 1)
        done = win or dead
        done_reason = "win" if win else ("dead" if dead else None)

        if self.auto_restart and done:
            self._release_all()
            pag.press(self.key_reset)
            self._skip_reset_once = True
            self._pending_reset = False
            time.sleep(0.3)
        # 若上一帧已标记重置，确保按一次 R
        if self._pending_reset and not done:
            self._release_all()
            pag.press(self.key_reset)
            self._pending_reset = False
            time.sleep(0.1)

        # 更新状态
        self.last_boss = (boss_c, boss_m)
        self.parry_last = 0
        if read_x_now and (x_cur is not None):
            self.x_last = float(x_cur)
        info = {
            "boss_hp": boss_c,
            "boss_hp_max": boss_m,
            "player_hp": ply_hp,
            "parry": self.parry_last,
            "x": self.x_last,
            "facing": None,
            "skill": self._skill_points,
            "win": win,
            "dead": dead,
            "done_reason": done_reason,
        }
        # 不写入步级 CSV 日志
        self._step_count += 1
        # 打印总 FPS（包含抓帧+推理周期），每 50 步统计一次
        self._fps_counter += 1
        if self._fps_counter >= 50:
            now_ts = time.time()
            elapsed = now_ts - self._fps_last_ts
            if elapsed > 0:
                fps = self._fps_counter / elapsed
                self._last_fps = fps
            self._fps_counter = 0
            self._fps_last_ts = now_ts

        return self.stackbuf.copy(), reward, done, False, info

    def render(self):
        pass

    def close(self):
        self._release_all()
        self._frame_stop.set()
        if self._frame_thread and self._frame_thread.is_alive():
            self._frame_thread.join(timeout=0.5)
        self._mem_stop.set()
        if self._mem_thread and self._mem_thread.is_alive():
            self._mem_thread.join(timeout=0.5)
        if self.cam:
            self.cam.stop()
        self._ocr_stop.set()
        if self._ocr_thread and self._ocr_thread.is_alive():
            self._ocr_thread.join(timeout=0.5)
        if self._hp_hproc:
            try:
                import win32api  # type: ignore
                win32api.CloseHandle(self._hp_hproc)
            except Exception:
                pass
        if self._step_log_fp:
            try:
                self._step_log_fp.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
