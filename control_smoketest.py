import time
import sys
import pyautogui as pag

# 可选：更快的窗口聚焦
def focus_cuphead_window():
    try:
        import pygetwindow as gw
        titles = [t for t in gw.getAllTitles() if t and 'cuphead' in t.lower()]
        if titles:
            w = gw.getWindowsWithTitle(titles[0])[0]
            # 尝试激活/还原/置顶
            try: w.activate()
            except: pass
            try: w.restore()
            except: pass
            time.sleep(0.2)
            return True
    except Exception:
        pass
    # 兜底：点击屏幕中心激活前台
    try:
        sw, sh = pag.size()
        pag.moveTo(sw//2, sh//2, duration=0.05)
        pag.click()
        time.sleep(0.1)
        return True
    except Exception:
        return False

# —— 你的键位（与你截图一致）——
KEY_Invincible   = "5"
KEY_LEFT   = "a"
KEY_RIGHT  = "d"
KEY_UP     = "w"
KEY_DOWN   = "s"
KEY_JUMP   = "space"       # 跳跃
KEY_SHOOT  = "j"           # 射击
KEY_SUPER  = "k"           # 强力（本测试未用）
KEY_SWAP   = "q"           # 换武器（本测试未用）
KEY_LOCK   = "shiftright"  # 锁定（注意：pyautogui 使用 shiftright）
KEY_DASH   = "shift"       # 左 Shift（或用 "shiftleft"）
KEY_RESET  = "r"           # 重开

pag.FAILSAFE = False   # 鼠标移角落不触发异常
pag.PAUSE = 0.02       # 连续按键的小间隔

def tap(key: str, hold: float = 0.06):
    pag.keyDown(key); time.sleep(hold); pag.keyUp(key)

def chord(keys_down, hold: float = 0.08):
    for k in keys_down: pag.keyDown(k)
    time.sleep(hold)
    for k in reversed(keys_down): pag.keyUp(k)

def main():
    print("[INFO] 尝试聚焦 Cuphead 窗口…")
    focused = focus_cuphead_window()
    print(f"[INFO] 聚焦结果: {focused}")

    for i in (3, 2, 1):
        print(f"[INFO] {i}…"); time.sleep(1.0)

    # 可选：重开一次，确保在关卡内
    print("[INFO] 按 R 重开关卡")
    tap(KEY_RESET, hold=0.05)
    time.sleep(1.0)

    print("[INFO] 演示开始：右移 → 跳 → 射击 → 冲刺 → 上/下射 → 跳射 → 锁定扫射")
    pag.keyDown(KEY_Invincible); time.sleep(0.8); pag.keyUp(KEY_Invincible); time.sleep(0.2)
    # 1) 右移 0.8s
    pag.keyDown(KEY_RIGHT); time.sleep(0.8); pag.keyUp(KEY_RIGHT); time.sleep(0.2)

    # 2) 跳
    tap(KEY_JUMP, hold=0.08); time.sleep(0.2)

    # 3) 连续射击 0.6s
    pag.keyDown(KEY_SHOOT); time.sleep(0.6); pag.keyUp(KEY_SHOOT); time.sleep(0.2)

    # 4) 冲刺
    tap(KEY_DASH, hold=0.08); time.sleep(0.25)

    # 5) 向上射（抬枪射击）
    pag.keyDown(KEY_UP); tap(KEY_SHOOT, hold=0.06); pag.keyUp(KEY_UP); time.sleep(0.2)

    # 6) 向下射（压枪射击）
    pag.keyDown(KEY_DOWN); tap(KEY_SHOOT, hold=0.06); pag.keyUp(KEY_DOWN); time.sleep(0.2)

    # 7) 跳跃 + 射击（空中输出）
    pag.keyDown(KEY_JUMP); pag.keyDown(KEY_SHOOT)
    time.sleep(0.25)
    pag.keyUp(KEY_SHOOT); pag.keyUp(KEY_JUMP)
    time.sleep(0.3)

    # 8) 锁定扫射（在原地向左/向右扫两次）
    pag.keyDown(KEY_LOCK)
    for _ in range(2):
        pag.keyDown(KEY_LEFT); tap(KEY_SHOOT, hold=0.08); pag.keyUp(KEY_LEFT); time.sleep(0.15)
        pag.keyDown(KEY_RIGHT); tap(KEY_SHOOT, hold=0.08); pag.keyUp(KEY_RIGHT); time.sleep(0.15)
    pag.keyUp(KEY_LOCK)

    print("[DONE] 动作演示完成。若人物未动，请确认：游戏在前台、权限一致（管理员 ↔ 普通）、键位与脚本一致。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[USER] 中断。")
        sys.exit(0)
