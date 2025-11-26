"""Memory reader for Cuphead HP values."""
from __future__ import annotations

import json
import os
import ctypes
from ctypes import wintypes
from typing import Dict, Any, Optional, Sequence

try:
    import win32gui
    import win32process
    import win32api
except ImportError:  # pragma: no cover - only on Windows
    win32gui = None
    win32process = None
    win32api = None


Psapi = ctypes.WinDLL('Psapi.dll') if os.name == "nt" else None
Kernel32 = ctypes.WinDLL('kernel32.dll') if os.name == "nt" else None
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
kernel32 = ctypes.windll.kernel32 if os.name == "nt" else None


def _load_config() -> Dict[str, Any]:
    default = {
        "boss": {
            "module": "mono.dll",
            "base_offset": 0x00264A68,
            "offsets": [0xA0, 0xB40, 0x170],
            "final_offset": 0x3C,
            "type": "float",
        },
        "player": {
            "module": "mono.dll",
            "base_offset": 0x002BD940,
            "offsets": [0x688, 0xE8, 0x120],
            "final_offset": 0xB4,
            "type": "int",
        },
        "player_x": {
            "module": "UnityPlayer.dll",
            "base_offset": 0x01468F30,
            "offsets": [0x38],
            "final_offset": 0x18C,
            "type": "float",
        },
    }
    cfg_path = "cuphead_mem.json"
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            default.update(data)
        except Exception:
            pass
    return default


class CupheadMemoryReader:
    def __init__(self, window_title: str = "Cuphead", config: Optional[Dict[str, Any]] = None):
        if win32gui is None:
            raise ImportError("pywin32 is required for memory reading")
        self.window_title = window_title
        self.config = config or _load_config()
        self.process_handle = None
        self.pid = None
        self.modules: Dict[str, int] = {}
        self._open_process()

    def _open_process(self) -> None:
        hwnd = win32gui.FindWindow(None, self.window_title)
        if not hwnd:
            raise RuntimeError("Cuphead window not found")
        pid = win32process.GetWindowThreadProcessId(hwnd)[1]
        self.process_handle = win32api.OpenProcess(0x1F0FFF, False, pid)
        self.pid = pid
        self.modules.clear()

    def _get_module_base(self, suffix: str) -> Optional[int]:
        suffix = suffix.lower()
        if suffix in self.modules:
            return self.modules[suffix]
        hProcess = Kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, self.pid)
        buf_count = 256
        while True:
            HMODULE_ARR = wintypes.HMODULE * buf_count
            buf = HMODULE_ARR()
            needed = wintypes.DWORD()
            if not Psapi.EnumProcessModulesEx(hProcess, ctypes.byref(buf), ctypes.sizeof(buf), ctypes.byref(needed), 0x03):
                raise OSError("EnumProcessModulesEx failed")
            if ctypes.sizeof(buf) < needed.value:
                buf_count = needed.value // ctypes.sizeof(wintypes.HMODULE)
                continue
            count = needed.value // ctypes.sizeof(wintypes.HMODULE)
            for i in range(count):
                h_mod = buf[i]
                path = win32process.GetModuleFileNameEx(self.process_handle, h_mod)
                if path.lower().endswith(suffix):
                    self.modules[suffix] = h_mod
                    return h_mod
            break
        return None

    def _read_pointer_chain(self, entry: Dict[str, Any]) -> Optional[int]:
        module = entry.get("module")
        base_offset = entry.get("base_offset")
        offsets = entry.get("offsets", [])
        final_offset = entry.get("final_offset", 0)
        if module is None or base_offset is None:
            return None
        base = self._get_module_base(module)
        if base is None:
            return None
        addr = ctypes.c_void_p(base + base_offset)
        ptr = ctypes.c_ulonglong()
        if not kernel32.ReadProcessMemory(int(self.process_handle), addr, ctypes.byref(ptr), 8, None):
            return None
        for off in offsets:
            next_addr = ctypes.c_void_p(ptr.value + off)
            if not kernel32.ReadProcessMemory(int(self.process_handle), next_addr, ctypes.byref(ptr), 8, None):
                return None
        return ptr.value + final_offset

    def _read_value(self, key: str) -> Optional[float]:
        entry = self.config.get(key)
        if not entry:
            return None
        addr = self._read_pointer_chain(entry)
        if addr is None:
            return None
        dtype = entry.get("type", "float").lower()
        if dtype == "float":
            buf = ctypes.c_float()
            size = 4
        else:
            buf = ctypes.c_int()
            size = 4
        if not kernel32.ReadProcessMemory(int(self.process_handle), ctypes.c_void_p(addr), ctypes.byref(buf), size, None):
            return None
        return float(buf.value)

    def read_boss_hp(self) -> Optional[float]:
        return self._read_value("boss")

    def read_player_hp(self) -> Optional[float]:
        return self._read_value("player")

    def read_player_x(self) -> Optional[float]:
        return self._read_value("player_x")
