# calibrate_roi.py
import json, cv2, numpy as np, dxcam, time, os
CONF = "cuphead_roi.json"

def grab_frame(cam):
    frame = cam.get_latest_frame()
    while frame is None:
        frame = cam.get_latest_frame()
        time.sleep(0.01)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2]==4 else frame

def select_roi(name, img):
    r = cv2.selectROI(f"Select ROI - {name}", img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(f"Select ROI - {name}")
    return [int(r[0]), int(r[1]), int(r[2]), int(r[3])]

def main():
    cam = dxcam.create(output_color="BGR")
    cam.start(target_fps=60, video_mode=True)
    time.sleep(0.2)
    img = grab_frame(cam)
    print("请框选【Boss HP 数字】区域")
    boss = select_roi("BossHP", img)
    print("请框选【玩家HP（数字或心形）】区域")
    player = select_roi("PlayerHP", img)
    with open(CONF, "w", encoding="utf-8") as f:
        json.dump({"boss_hp_roi": boss, "player_hp_roi": player}, f, ensure_ascii=False, indent=2)
    print(f"ROI 已保存到 {os.path.abspath(CONF)}")
    cam.stop()

if __name__ == "__main__":
    main()
