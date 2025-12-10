"""Interactively select an ROI and print (x, y, w, h) for OCR/Boss HP."""
import time
import json
import os

import cv2
import dxcam


def grab_frame(cam):
    frame = cam.get_latest_frame()
    while frame is None:
        time.sleep(0.05)
        frame = cam.get_latest_frame()
    # dxcam returns BGRA 有时；转 BGR 方便显示/保存
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def main():
    cam = dxcam.create(output_color="BGR")
    cam.start(target_fps=10, video_mode=True)
    time.sleep(0.2)

    frame = grab_frame(cam)
    print("[INFO] 截取到一帧，按窗口中拖拽选择 ROI，回车确认，ESC 取消")
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    cam.stop()

    x, y, w, h = roi
    print(f"[ROI] (x, y, w, h) = ({x}, {y}, {w}, {h})")

    # 可选：写入文件
    out_path = os.path.join(os.path.dirname(__file__), "roi_selected.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"roi": [int(x), int(y), int(w), int(h)]}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已保存到 {out_path}")


if __name__ == "__main__":
    main()
