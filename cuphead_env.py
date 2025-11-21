# cuphead_env.py
import time
import json
import csv
import threading
import queue
import collections
import cv2
import numpy as np
import dxcam
import pyautogui as pag
import gymnasium as gym
from gymnasium import spaces
try:
    from test_cuphead_hp import (
        read_player_hp as test_read_player_hp,
        read_boss_hp as test_read_boss_hp,
        read_player_x as test_read_player_x,
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
    from cuphead_memory import CupheadMemoryReader
except Exception:  # pragma: no cover
    CupheadMemoryReader = None

pag.FAILSAFE = False
pag.PAUSE = 0  # 移除pyautogui的自动延迟

# 复用识别函数（与 read_hp.py 同目录）
from read_hp import (
    grab, ocr_number_block, parse_boss_hp,
    read_player_digit_only, extract_digit_bin, is_digit_one_by_shape,
    detect_dead_text, is_hp_badge_red, is_dead_fast,
    # 新增：Parry 与 X 坐标
    read_parry_count_only, read_xcoord_only
)

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
        reward_boss_damage_mul: float = 0.04,
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
        x_max: float = 228.0,
        x_margin: float = 5.0,
        reward_wall_penalty: float = 1.0,
        dir_hold_limit: int = 20,
    ):
        super().__init__()
        self.debug = debug
        self.auto_restart = auto_restart
        self.dt = 1.0 / max(1, decision_fps)
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
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.stack, self.H, self.W), dtype=np.uint8
        )

        # —— 相机 —— #
        self.cam = dxcam.create(output_color="BGR")
        # 限帧以降低抓帧开销
        self.cam.start(target_fps=20, video_mode=True)
        time.sleep(0.15)

        # —— ROI —— #
        roi = json.load(open("cuphead_roi.json", "r", encoding="utf-8"))
        self.bx, self.by, self.bw, self.bh = roi["boss_hp_roi"]
        self.px, self.py, self.pw, self.ph = roi["player_hp_roi"]

        self.has_parry = "parry_roi" in roi
        if self.has_parry:
            self.sx, self.sy, self.sw, self.sh = roi["parry_roi"]
        else:
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
        self.key_special= "k"            # 特殊技能
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
        self._step_count = 0

        self.parry_last = 0
        self.parry_used = 0
        self.x_last = None
        self.facing_dir = 1  # 1: right, -1: left
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
        self._mem_data = {"boss": None, "player_stable": None, "player_raw": None, "timestamp": 0.0}
        self._mem_lock = threading.Lock()
        self._mem_stop = threading.Event()
        self._mem_thread = None
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
        self._last_player_print = None
        self._dash_triggered = False
        self._last_player_hp_raw = None
        self._step_log_fp = None
        self._step_log_writer = None
        if self.use_memory_hp and CupheadMemoryReader is not None:
            try:
                self.mem_reader = CupheadMemoryReader()
                print("[INFO] Cuphead memory reader initialized")
                self._mem_thread = threading.Thread(
                    target=self._memory_worker, name="CupheadMem", daemon=True
                )
                self._mem_thread.start()
            except Exception as exc:
                self.mem_reader = None
                print(f"[WARN] Memory reader init failed: {exc}")
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
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return g

    def _stack_init(self, g):
        self.stackbuf = np.repeat(g[None, ...], self.stack, axis=0)

    def _stack_push(self, g):
        self.stackbuf = np.concatenate([self.stackbuf[1:], g[None, ...]], axis=0)
    def _frame_worker(self):
        while not self._frame_stop.is_set():
            frame = grab(self.cam)
            with self._frame_lock:
                self._frame_queue.append(frame)
            time.sleep(0.001)

    def _get_latest_frame(self):
        with self._frame_lock:
            if self._frame_queue:
                return self._frame_queue[-1].copy()
        return grab(self.cam)

    def _memory_worker(self):
        if not self.mem_reader:
            return
        while not self._mem_stop.is_set():
            boss = None
            player_raw = None
            x_raw = None
            try:
                boss = self.mem_reader.read_boss_hp()
            except Exception:
                boss = None
            try:
                player_raw = self.mem_reader.read_player_hp()
            except Exception:
                player_raw = None
            if (player_raw is None) and self._hp_hproc and self._hp_mono_base and test_read_player_hp:
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
                    threshold=3,
                )
                self._player_hp_state_test.update(
                    stable=stable_hp,
                    candidate=cand_hp,
                    count=cand_cnt,
                )
                player_stable = stable_hp
            else:
                player_stable = self._debounce_player_hp(player_raw)
            # 写入缓存均使用稳定值
            with self._mem_lock:
                self._mem_data = {
                    "boss": boss,
                    "player_raw": player_raw,
                    "player_stable": player_stable,
                    "x_raw": x_raw,
                    "timestamp": time.time(),
                }
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

    def _tap(self, key, dur=0.04):
        try:
            pag.keyDown(key)
            time.sleep(dur)
            pag.keyUp(key)
        except Exception:
            pass

    def _tap_combo(self, keys, dur=0.04):
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

    def _add_safe_window(self, dur: float = 1.0):
        exp = time.time() + max(0.0, dur)
        self._pending_safe.append({"expire": exp, "failed": False})
        if len(self._pending_safe) > 10:
            self._pending_safe = self._pending_safe[-10:]

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
        target_state = {
            self.key_left:  bool(L),
            self.key_right: bool(R),
            self.key_up:    bool(Up),
            self.key_shoot: bool(Shoot),  # J持续按
            self.key_lock:  bool(Lock),
        }
        for k, want in target_state.items():
            if want and not self._held[k]:
                self._press(k)
            elif (not want) and self._held[k]:
                self._release(k)

        # --- 更新朝向 ---
        if L and not R:
            self.facing_dir = -1
        elif R and not L:
            self.facing_dir = 1

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
        if down_trigger and not self._prev_action[3]:
            self._tap_combo([self.key_down, self.key_dash], dur=0.05)
            self._recent_crouch_dash = True
            self._add_safe_window()
        Down = 0

        # --- 智能跳跃系统 ---
        if Jump and not self._prev_action[4]:
            # 根据JumpHold决定跳跃时长：短跳0.03s，长跳0.08s
            jump_duration = 0.08 if JumpHold else 0.03
            self._tap(self.key_jump, dur=jump_duration)
            self._add_safe_window()
            Jump = 0
        if Dash and not self._prev_action[6]:
            self._tap(self.key_dash, dur=0.04)
            self._recent_crouch_dash = bool(self._held.get(self.key_down, False))
            self._dash_triggered = True
            self._add_safe_window()
            Dash = 0

        # --- 技能键 ---
        if allow_k and Special and not self._prev_action[8]:
            if self.parry_last is not None and (self.parry_used < int(self.parry_last)):
                self._tap(self.key_special, dur=0.04)
                self.parry_used += 1
            Special = 0

        self._prev_action = [L, R, Up, down_trigger, Jump, Shoot, Dash, Lock, Special, JumpHold]

    # ====== OCR读取 ====== #
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
        val, tag = read_parry_count_only(roi)
        return (int(val) if isinstance(val, (int, np.integer)) else val), tag

    def _valid_async_data(self):
        return None

    def _get_mem_snapshot(self):
        if self.mem_reader is None:
            return None
        with self._mem_lock:
            return dict(self._mem_data)

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
        mem_snapshot = self._get_mem_snapshot()
        now = time.time()
        last_c, last_m = self.last_boss

        def _valid(cur, mx, ts=None):
            if cur is None or mx is None or mx <= 0:
                return False
            if ts is not None and (now - ts) > self.hp_valid_age:
                return False
            if last_c is not None and (last_c - float(cur)) > self.boss_max_drop:
                return False
            return True

        if mem_snapshot and mem_snapshot.get("boss") is not None:
            boss_val = float(mem_snapshot["boss"])
            boss_m = max(boss_val, last_m) if last_m is not None else boss_val
            if _valid(boss_val, boss_m, mem_snapshot.get("timestamp", 0)):
                return boss_val, boss_m
        # 无 OCR 回退，直接使用上次的值
        return self.last_boss

    def _fetch_player_hp(self, frame):
        mem_snapshot = self._get_mem_snapshot()
        if mem_snapshot and mem_snapshot.get("player_stable") is not None:
            return mem_snapshot["player_stable"]
        return int(self.last_player_hp) if self.last_player_hp is not None else self.player_hp_max

    def _fetch_parry(self, frame):
        return None

    def _fetch_xcoord(self, frame):  # legacy stub for compatibility
        snap = self._get_mem_snapshot()
        if snap and snap.get("x_raw") is not None:
            return float(snap["x_raw"])
        return None


    # ====== Gym API ====== #
    def reset(self, *, seed=None, options=None):
        self._release_all()
        if not self._skip_reset_once:
            pag.press(self.key_reset)
            time.sleep(0.3)  # 减少等待时间
        else:
            self._skip_reset_once = False
            time.sleep(0.1)  # 减少等待时间

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
        self.facing_dir = 1
        self._recent_crouch_dash = False
        self._dash_triggered = False
        self.last_player_hp = self.player_hp_max
        self._last_player_hp_raw = None
        self._pending_safe = []
        self._dir_hold_steps = 0
        self._dir_hold_dir = 0
        return self.stackbuf.copy(), {}

    def step(self, action):
        t0 = time.perf_counter()
        # 记录使用技能前的计数，以便在后续奖励中判断是否消耗技能点
        parry_used_before = self.parry_used
        self._apply_action(action, allow_k=True)
        elapsed = time.perf_counter() - t0
        remain = self.dt - elapsed
        # 大幅减少等待时间以提高FPS
        if remain > 0.001:  # 只有当剩余时间大于1ms时才等待
            time.sleep(min(remain, 0.01))  # 最多等待10ms

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
        # 初始化奖励
        reward = 0.0
        # 边界推进惩罚：在边缘仍向外走
        if x_cur is not None:
            at_left_edge = (x_cur <= self.x_min + self.x_margin)
            at_right_edge = (x_cur >= self.x_max - self.x_margin)
            if at_left_edge and (cur_left or (cur_dash and self.facing_dir < 0)):
                reward -= self.reward_wall_penalty
            if at_right_edge and (cur_right or (cur_dash and self.facing_dir > 0)):
                reward -= self.reward_wall_penalty

        # ==== 奖励 shaping ====
        last_c, last_m = self.last_boss
        # boss 掉血奖励（按绝对血量差给分）
        boss_damage = max(0.0, (last_c - boss_c))
        reward += boss_damage * self.reward_boss_damage_mul

        # 存活奖励：Boss 越残血奖励越高，鼓励持续输出
        boss_ratio = float(boss_c / boss_m) if boss_m and boss_m > 0 else 1.0
        progress = 1.0 - np.clip(boss_ratio, 0.0, 1.0)
        reward += self.reward_progress_bonus * progress

        # 技能使用奖励（若消耗了技能点则鼓励）
        if self.parry_used > parry_used_before:
            reward += self.reward_skill_use

        # parry获得奖励（OCR检测到parry上升）
        if False and read_parry_now:
            pass

        # Duck+Dash 小加分（触发时奖励）
        if self._recent_crouch_dash:
            reward += self.reward_duck_dash
            self._recent_crouch_dash = False
        # Dash 成功且未掉血奖励
        if self._dash_triggered:
            if (ply_hp is not None) and (self.last_player_hp is not None) and (ply_hp >= self.last_player_hp):
                reward += self.reward_dash_safe
            self._dash_triggered = False

        # 命中奖励：在射击且 Boss 掉血时额外加分
        if self._held.get(self.key_shoot, False) and boss_damage > 0:
            reward += self.reward_shoot_hit

        took_damage = False
        # 玩家受伤惩罚（一次掉血扣固定分）
        if ply_hp is not None:
            if self.last_player_hp is not None and ply_hp < self.last_player_hp:
                dmg = float(self.last_player_hp - ply_hp)
                if abs(dmg - 1.0) < 1e-6:
                    reward -= self.reward_player_damage_penalty
                    took_damage = True
            self.last_player_hp = ply_hp

        # 安全窗口奖励：Dash/Jump/下蹲冲刺后1s内未掉血
        if self._pending_safe:
            now = time.time()
            remaining = []
            for item in self._pending_safe:
                exp = item.get("expire", 0)
                failed = item.get("failed", False)
                if took_damage:
                    failed = True
                if now >= exp:
                    if not failed:
                        reward += 1.0
                    continue
                remaining.append({"expire": exp, "failed": failed})
            self._pending_safe = remaining

        # 提前结束以避免击杀后无法重开：Boss 血量低于 50 即视为胜利
        win = (boss_c <= 50.0)
        if ply_hp is None:
            ply_hp = self.last_player_hp
        dead = False
        if ply_hp is not None:
            dead = (ply_hp <= 1)
        done = win or dead
        done_reason = "win" if win else ("dead" if dead else None)

        # 跳过前几步的奖励与终止，等待感知稳定（死亡情况不延迟重开）
        if self._step_count < self.warmup_steps and not dead:
            reward = 0.0
            done = False
            done_reason = "warmup_skip"

        # 限制最短 episode，避免极短局（死亡不延迟重开）
        if done and (done_reason != "dead") and self._step_count < self.min_episode_steps:
            done = False
            done_reason = "min_steps_block"

        if self.auto_restart and done:
            self._release_all()
            pag.press(self.key_reset)
            self._skip_reset_once = True
            time.sleep(1.0)

        # 更新状态
        self.last_boss = (boss_c, boss_m)
        self.parry_last = 0
        if read_x_now and (x_cur is not None):
            x_val = float(x_cur)
            if self.x_last is not None:
                dx = x_val - float(self.x_last)
                if abs(dx) > 1e-3:
                    self.facing_dir = 1 if dx > 0 else -1
            self.x_last = x_val
        info = {
            "boss_hp": boss_c,
            "boss_hp_max": boss_m,
            "player_hp": ply_hp,
            "parry": self.parry_last,
            "x": self.x_last,
            "facing": self.facing_dir,
            "win": win,
            "dead": dead,
            "done_reason": done_reason,
        }
        # 不写入步级 CSV 日志
        self._step_count += 1
        if self.debug:
            print(info)

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
