# -*- coding: utf-8 -*-
"""
fps_check.py — 复用 read_hp.py 的函数，基准测试各阶段 FPS（不训练）
用法示例：
  python fps_check.py --stage raw --seconds 15
  python fps_check.py --stage roi_player --seconds 15
  python fps_check.py --stage ocr_boss --seconds 15
  python fps_check.py --stage ocr_player --seconds 15
"""
import time, json, argparse, os
import numpy as np
import cv2, dxcam

# 复用你已实现的函数
from read_hp import grab, ocr_number_block, parse_boss_hp, read_player_digit_only

def load_roi(path, key):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 ROI 文件：{path}")
    roi = json.load(open(path, "r", encoding="utf-8"))
    if key not in roi:
        raise KeyError(f"ROI 缺少字段：{key}")
    return roi[key]  # x, y, w, h

def bench(stage: str, seconds: int, show: bool, roi_path: str):
    cam = dxcam.create(output_color="BGR")
    # target_fps=0 表示尽可能快；video_mode=True 用非阻塞队列
    cam.start(target_fps=0, video_mode=True)
    time.sleep(0.2)

    rx_ply = ry_ply = rw_ply = rh_ply = None
    rx_bos = ry_bos = rw_bos = rh_bos = None
    if stage in ("roi_player", "ocr_player"):
        rx_ply, ry_ply, rw_ply, rh_ply = load_roi(roi_path, "player_hp_roi")
    if stage in ("roi_boss", "ocr_boss"):
        rx_bos, ry_bos, rw_bos, rh_bos = load_roi(roi_path, "boss_hp_roi")

    print(f"[INFO] stage={stage} seconds={seconds} show={show}")
    t0 = time.perf_counter()
    frames = 0
    tick = []

    try:
        while True:
            if time.perf_counter() - t0 >= seconds:
                break

            frame = grab(cam)  # 使用你封装的抓屏函数（含 BGRA→BGR 处理）

            if stage == "raw":
                pass

            elif stage == "roi_player":
                _ = frame[ry_ply:ry_ply+rh_ply, rx_ply:rx_ply+rw_ply]

            elif stage == "roi_boss":
                _ = frame[ry_bos:ry_bos+rh_bos, rx_bos:rx_bos+rw_bos]

            elif stage == "ocr_boss":
                boss = frame[ry_bos:ry_bos+rh_bos, rx_bos:rx_bos+rw_bos]
                raw = ocr_number_block(boss)
                _cur, _mx = parse_boss_hp(raw)  # 仅为模拟完整链路

            elif stage == "ocr_player":
                ply = frame[ry_ply:ry_ply+rh_ply, rx_ply:rx_ply+rw_ply]
                _pcur, _rawp = read_player_digit_only(ply)

            else:
                raise ValueError(f"未知阶段：{stage}")

            frames += 1
            tick.append(time.perf_counter())

            if show:
                vis = frame.copy()
                if stage in ("roi_player", "ocr_player"):
                    cv2.rectangle(vis, (rx_ply, ry_ply), (rx_ply+rw_ply, ry_ply+rh_ply), (0,255,0), 2)
                if stage in ("roi_boss", "ocr_boss"):
                    cv2.rectangle(vis, (rx_bos, ry_bos), (rx_bos+rw_bos, ry_bos+rh_bos), (0,0,255), 2)
                cv2.imshow("fps_check", cv2.resize(vis, (960, 540)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cam.stop()
        cv2.destroyAllWindows()

    dur = max(time.perf_counter() - t0, 1e-9)
    avg_fps = frames / dur
    # 即时 FPS 的分位数（更稳）
    inst_fps = []
    for i in range(1, len(tick)):
        dt = tick[i] - tick[i-1]
        if dt > 0:
            inst_fps.append(1.0 / dt)
    p50 = float(np.median(inst_fps)) if inst_fps else avg_fps
    p10 = float(np.percentile(inst_fps, 10)) if inst_fps else avg_fps
    p90 = float(np.percentile(inst_fps, 90)) if inst_fps else avg_fps

    print("\n=== FPS Report ===")
    print(f"Frames:     {frames}")
    print(f"Duration:   {dur:.3f}s")
    print(f"Average:    {avg_fps:.2f} fps")
    print(f"Instant p10/p50/p90: {p10:.2f} / {p50:.2f} / {p90:.2f} fps")
    print("注：关闭 --show、置顶/全屏 Cuphead，可测到更真实上限。")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["raw","roi_player","roi_boss","ocr_player","ocr_boss"], default="raw")
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--roi", type=str, default="cuphead_roi.json")
    args = ap.parse_args()
    bench(args.stage, args.seconds, args.show, args.roi)
