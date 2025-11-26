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
    if raw_hp is None or not (min_val <= raw_hp <= max_val):
        return stable_hp, candidate_hp, candidate_count

    if stable_hp is None:
        stable_hp = raw_hp
        candidate_hp = None
        candidate_count = 0
        return stable_hp, candidate_hp, candidate_count

    if raw_hp == stable_hp:
        candidate_hp = None
        candidate_count = 0
        return stable_hp, candidate_hp, candidate_count

    if candidate_hp is None or raw_hp != candidate_hp:
        candidate_hp = raw_hp
        candidate_count = 1
    else:
        candidate_count += 1
        if candidate_count >= threshold:
            stable_hp = candidate_hp
            candidate_hp = None
            candidate_count = 0

    return stable_hp, candidate_hp, candidate_count

# -------- Player HP (int, mono.dll 链) --------
def read_player_hp(hproc, mono_base):
    """
    mono.dll + 0x002BD940 -> 0x688 -> 0xE8 -> 0x120 -> +0xB4 (int)
    对应 CE 图：15A63D1245C，值为 4。
    """
    base_offset = 0x002BD940
    deref_offsets = [0x688, 0xE8, 0x120]
    last_offset = 0xB4

    addr = mono_base + base_offset
    ptr = ctypes.c_uint64()

    # 第一次：读 mono.dll + base_offset
    if not rpm(hproc, addr, ptr):
        return 0

    # 依次解引用 0x890 -> 0xE8 -> 0xBC0
    for off in deref_offsets:
        next_addr = ptr.value + off
        if not rpm(hproc, next_addr, ptr):
            return 0

    # 最后一段：+0xB4 直接读 int
    hp4 = ctypes.c_uint32()
    if not rpm(hproc, ptr.value + last_offset, hp4):
        return 0

    return int(hp4.value)

# -------- Player X (float) --------
def read_player_x(hproc, unity_base):

    base_addr = unity_base + 0x01468F30

    ptr = ctypes.c_uint64()
    if not rpm(hproc, base_addr, ptr):
        return None

    # ptr → 18167C1D300
    ptr1_addr = ptr.value + 0x38
    if not rpm(hproc, ptr1_addr, ptr):
        return None

    # ptr → 18195A540CD0
    ptr2_addr = ptr.value + 0x140
    if not rpm(hproc, ptr2_addr, ptr):
        return None

    # ptr → 1895A5F91B0
    final_addr = ptr.value + 0x688   # 最后一级不是 pointer!!

    x_val = ctypes.c_float()
    if not rpm(hproc, final_addr, x_val):
        return None

    return x_val.value


# -------- Skill points (float, max 50, every +10 one skill) --------
def read_skill_points(hproc, mono_base):
    """
    mono.dll + 0x002BFB40 -> 0x898 -> 0x78 -> 0xBC0 -> +0xD8 (float)
    对应 CE 图二：最终地址 20373CFD210，值为 10.0。
    """
    base_offset = 0x002BFB40
    deref_offsets = [0x898, 0x78, 0xBC0]
    last_offset = 0xD8

    addr = mono_base + base_offset
    ptr = ctypes.c_uint64()

    # 第一次：读 mono.dll + base_offset
    if not rpm(hproc, addr, ptr):
        return None

    # 依次解引用 0x898 -> 0x78 -> 0xBC0
    for off in deref_offsets:
        next_addr = ptr.value + off
        if not rpm(hproc, next_addr, ptr):
            return None

    # 最后一段：+0xD8 读 float
    val = ctypes.c_float()
    if not rpm(hproc, ptr.value + last_offset, val):
        return None

    return float(val.value)


def monitor_cuphead_hp(duration: float = 10.0, interval: float = 0.2):
    hproc, pid = open_cuphead()
    mono_base = enum_module(hproc, pid, "mono.dll")
    unity_base = enum_module(hproc, pid, "UnityPlayer.dll")

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
            skill = read_skill_points(hproc, mono_base)
            x = read_player_x(hproc, unity_base)
            samples.append(
                {
                    "ts": time.time(),
                    "boss_hp": boss_hp,
                    "player_hp_raw": raw_hp,
                    "player_hp": stable_hp,
                    "candidate": candidate_hp,
                    "candidate_count": candidate_count,
                    "skill": skill,
                    "x": x,
                }
            )
            print(f"Boss={boss_hp}, HP_raw={raw_hp}, HP={stable_hp}, skill={skill}, x={x}")
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

    stable_hp = None
    candidate_hp = None
    candidate_count = 0

    monitor_cuphead_hp()
