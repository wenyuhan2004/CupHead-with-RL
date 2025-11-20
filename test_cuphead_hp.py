import time
import ctypes
from ctypes import wintypes
import win32gui, win32process, win32api

Psapi = ctypes.WinDLL('Psapi.dll')
kernel32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010

kernel32.ReadProcessMemory.argtypes = [
    wintypes.HANDLE,      # hProcess
    wintypes.LPCVOID,     # lpBaseAddress
    wintypes.LPVOID,      # lpBuffer
    ctypes.c_size_t,      # nSize
    ctypes.POINTER(ctypes.c_size_t)  # lpNumberOfBytesRead
]
kernel32.ReadProcessMemory.restype = wintypes.BOOL


def rpm(hproc, address, buf):
    """简单封装 ReadProcessMemory，不打印调试信息"""
    size = ctypes.sizeof(buf)
    read = ctypes.c_size_t()

    h = wintypes.HANDLE(int(hproc))
    addr = ctypes.c_void_p(int(address))

    ok = kernel32.ReadProcessMemory(
        h,
        addr,
        ctypes.byref(buf),
        size,
        ctypes.byref(read)
    )
    return bool(ok)


def enum_module(hproc, pid, suffix):
    hProcess = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    buf_count = 256
    while True:
        HMODULE_ARR = wintypes.HMODULE * buf_count
        buf = HMODULE_ARR()
        needed = wintypes.DWORD()
        if not Psapi.EnumProcessModulesEx(
            hProcess,
            ctypes.byref(buf),
            ctypes.sizeof(buf),
            ctypes.byref(needed),
            0x03
        ):
            raise OSError("EnumProcessModulesEx failed")

        if ctypes.sizeof(buf) < needed.value:
            buf_count = needed.value // ctypes.sizeof(wintypes.HMODULE)
            continue

        count = needed.value // ctypes.sizeof(wintypes.HMODULE)
        for i in range(count):
            h_mod = buf[i]
            path = win32process.GetModuleFileNameEx(hproc, h_mod)
            if path.lower().endswith(suffix.lower()):
                return int(h_mod)

        raise RuntimeError(f"未找到模块 {suffix}")


def open_cuphead():
    hwnd = win32gui.FindWindow(None, "Cuphead")
    if not hwnd:
        raise RuntimeError("找不到 Cuphead 窗口")
    pid = win32process.GetWindowThreadProcessId(hwnd)[1]
    proc = win32api.OpenProcess(0x1F0FFF, False, pid)
    return proc, pid


# -------- Boss HP (float) --------
def read_boss_hp(hproc, mono_base):
    base_offset = 0x00264A68
    deref_offsets = [0xA0, 0xB40, 0x170]
    last_offset = 0x3C

    addr = mono_base + base_offset
    ptr = ctypes.c_uint64()
    if not rpm(hproc, addr, ptr):
        return None

    for off in deref_offsets:
        next_addr = ptr.value + off
        if not rpm(hproc, next_addr, ptr):
            return None

    hp_addr = ptr.value + last_offset
    hp = ctypes.c_float()
    if not rpm(hproc, hp_addr, hp):
        return None

    return hp.value
def update_hp_with_debounce(stable_hp, candidate_hp, candidate_count, raw_hp,
                            min_val=0, max_val=4, threshold=3):
    """
    stable_hp      : 当前确认的 HP（对外使用）
    candidate_hp   : 正在观测中的候选值
    candidate_count: 候选值已经连续出现的次数
    raw_hp         : 本帧从内存读到的原始值

    规则：
      - 只接受 [min_val, max_val] 范围内的值，其他视为噪声
      - 若 raw_hp == stable_hp，直接认定为稳定值，清空候选
      - 若 raw_hp != stable_hp，则作为候选，要求连读 'threshold' 次才更新 stable_hp
    """

    # 1) 原始值不在合法范围内：直接忽略，保持现状
    if raw_hp is None or not (min_val <= raw_hp <= max_val):
        return stable_hp, candidate_hp, candidate_count

    # 2) 目前还没有稳定值：第一次直接接受
    if stable_hp is None:
        stable_hp = raw_hp
        candidate_hp = None
        candidate_count = 0
        return stable_hp, candidate_hp, candidate_count

    # 3) 与当前稳定值一致：说明状态没变，清空候选
    if raw_hp == stable_hp:
        candidate_hp = None
        candidate_count = 0
        return stable_hp, candidate_hp, candidate_count

    # 4) 与 stable_hp 不一致：作为候选值
    if candidate_hp is None or raw_hp != candidate_hp:
        # 换了新候选，从 1 开始计数
        candidate_hp = raw_hp
        candidate_count = 1
    else:
        # 候选值和上一次相同，计数 +1
        candidate_count += 1

        # 连续出现次数达到阈值：确认更新稳定值
        if candidate_count >= threshold:
            stable_hp = candidate_hp
            candidate_hp = None
            candidate_count = 0

    return stable_hp, candidate_hp, candidate_count


# -------- Player HP (2 bytes) --------
# -------- Player HP (2 bytes，使用新的指针链) --------
# -------- Player HP (2 bytes，使用更稳定的指针链) --------
def read_player_hp(hproc, mono_base):
    """
    读取玩家 HP（0~4）：
        指针链：mono.dll+0x0027BAA0 -> 0xC8 -> 0x60 -> +0xB4
        最终类型：4 Bytes（int）
    约定：读取失败或值不在 0~4 时返回 0（视为死亡）
    """
    base_offset = 0x0027BAA0
    deref_offsets = [0xC8, 0x60]  # 需要解引用的两层
    last_offset = 0xB4            # 最后只做 +B4，不再解引用

    # level0: [mono.dll + base_offset]
    addr = mono_base + base_offset
    ptr = ctypes.c_uint64()
    if not rpm(hproc, addr, ptr):
        return 0

    # level1~2: 依次解引用
    for off in deref_offsets:
        next_addr = ptr.value + off
        if not rpm(hproc, next_addr, ptr):
            return 0

    # 最后一层：只 +B4，得到最终 HP 地址
    hp_addr = ptr.value + last_offset
    hp4 = ctypes.c_uint32()  # 4 Bytes
    if not rpm(hproc, hp_addr, hp4):
        return 0

    val = hp4.value
    # 保险起见做一次范围检查
    if 0 <= val <= 4:
        return val
    else:
        return 0
def read_player_x(hproc, unity_base):
    """
    读取茶杯头 X 坐标（float）：
        UnityPlayer.dll+0x0147B7B8 -> 0x0 -> 0x38 -> 0x4B0 -> +0x250
    失败时返回 None
    """
    base_offset = 0x0147B7B8
    deref_offsets = [0x0, 0x38, 0x4B0]
    last_offset = 0x250

    # level0: [unity_base + base_offset]
    addr = unity_base + base_offset
    ptr = ctypes.c_uint64()
    if not rpm(hproc, addr, ptr):
        return None

    # 依次解引用几层指针
    for off in deref_offsets:
        next_addr = ptr.value + off
        if not rpm(hproc, next_addr, ptr):
            return None

    # 最后一层：只加 0x250，不再解引用
    x_addr = ptr.value + last_offset
    xf = ctypes.c_float()
    if not rpm(hproc, x_addr, xf):
        return None

    return xf.value

def monitor_cuphead_hp(duration: float = 10.0, interval: float = 0.2):
    """
    连续读取 Boss/Player HP 并做去抖动，返回 (mono_base, samples)。
    samples: [{'ts', 'boss_hp', 'player_hp_raw', 'player_hp', 'candidate', 'candidate_count'}, ...]
    """
    hproc, pid = open_cuphead()
    mono_base = enum_module(hproc, pid, "mono.dll")

    stable_hp = None
    candidate_hp = None
    candidate_count = 0
    samples = []

    start = time.time()
    try:
        while time.time() - start < duration:
            raw_hp = read_player_hp(hproc, mono_base)
            stable_hp, candidate_hp, candidate_count = update_hp_with_debounce(
                stable_hp, candidate_hp, candidate_count, raw_hp
            )
            boss_hp = read_boss_hp(hproc, mono_base)
            samples.append(
                {
                    "ts": time.time(),
                    "boss_hp": boss_hp,
                    "player_hp_raw": raw_hp,
                    "player_hp": stable_hp,
                    "candidate": candidate_hp,
                    "candidate_count": candidate_count,
                }
            )
            time.sleep(interval)
    finally:
        try:
            win32api.CloseHandle(hproc)
        except Exception:
            pass

    return mono_base, samples


if __name__ == "__main__":
    hproc, pid = open_cuphead()
    mono_base  = enum_module(hproc, pid, "mono.dll")
    unity_base = enum_module(hproc, pid, "UnityPlayer.dll")
    print("mono base :", hex(mono_base))
    print("unity base:", hex(unity_base))

    # 去抖的 HP 状态（如果你还在用）
    stable_hp = None
    candidate_hp = None
    candidate_count = 0

    try:
        while True:
            boss_hp  = read_boss_hp(hproc, mono_base)
            raw_hp   = read_player_hp(hproc, mono_base)
            x_pos    = read_player_x(hproc, unity_base)

            # 如果你还用去抖动 HP：
            stable_hp, candidate_hp, candidate_count = update_hp_with_debounce(
                stable_hp, candidate_hp, candidate_count, raw_hp,
                min_val=0, max_val=4, threshold=3
            )
            hp_for_env = stable_hp if stable_hp is not None else 0
            dead = (hp_for_env == 0)

            print(f"BossHP={boss_hp:.1f}, HP_raw={raw_hp}, HP={hp_for_env}, "
                  f"X={x_pos}, Dead={dead}")

            time.sleep(0.2)
    except KeyboardInterrupt:
        print("stop")
