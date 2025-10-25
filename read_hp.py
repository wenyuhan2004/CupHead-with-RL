# read_hp.py
import json, time, re, os,collections
import cv2, numpy as np, dxcam, pytesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CONF = "cuphead_roi.json"
# 如未加入 PATH，请手动指定 Tesseract 路径
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def grab(cam):
    frame = cam.get_latest_frame()
    while frame is None:
        frame = cam.get_latest_frame()
        time.sleep(0.005)
    # dxcam 可能返回 BGRA
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame

def ocr_number_block(img_bgr):
    """Boss 区域：读 '123.45/1234' 形式文本"""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cfg = "--psm 6 -c tessedit_char_whitelist=0123456789./"
    return pytesseract.image_to_string(th, config=cfg).strip()

def parse_boss_hp(text):
    """解析 Boss HP；容忍前导噪点，如 '.21.56/1235'"""
    text = text.strip()
    text = re.sub(r"^[^\d]+", "", text)  # 去非数字前缀
    m = re.search(r"(\d+(?:\.\d+)?)[^\d]+(\d+)", text)
    if not m:
        return None, None
    cur = float(m.group(1)); mx = float(m.group(2))
    if mx <= 0: 
        return None, None
    cur = min(cur, mx)
    return cur, mx

# -------------------- 玩家HP：OCR优先 --------------------
def read_player_digit_only(img_bgr):
    """
    仅识别玩家 HP 数字（忽略 'HP.'），颜色无关，对红/黄/白底闪烁鲁棒。
    返回 (pcur:int|None, raw_text:str)
    """
    h, w = img_bgr.shape[:2]
    right_ratio = 0.50  # 右侧数字区占比（必要时调 0.45~0.55）
    rw = max(12, int(w * right_ratio))
    digit_roi = img_bgr[:, w - rw : w]

    digit_roi = cv2.resize(digit_roi, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(digit_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    def try_ocr(bin_img):
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, np.ones((2, 2), np.uint8), 1)
        cfg = r'--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789'
        raw = pytesseract.image_to_string(bin_img, config=cfg).strip()
        m = re.search(r'(\d+)', raw)
        return (int(m.group(1)), raw) if m else (None, raw)

    # 尝试顺序：OTSU-INV → ADAPT-INV → OTSU → ADAPT
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    p, raw = try_ocr(th)
    if p is not None: return p, raw

    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, blockSize=15, C=5)
    p, raw = try_ocr(th)
    if p is not None: return p, raw

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    p, raw = try_ocr(th)
    if p is not None: return p, raw

    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, blockSize=15, C=5)
    p, raw = try_ocr(th)
    if p is not None: return p, raw

    return None, "ocr_fail"

def extract_digit_bin(ply_roi_bgr, right_ratio=0.50):
    """裁剪出数字区并得到稳定的二值图，用于形状/模板兜底"""
    h, w = ply_roi_bgr.shape[:2]
    rw = max(12, int(w * right_ratio))
    roi = ply_roi_bgr[:, w - rw : w]
    roi = cv2.resize(roi, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, blockSize=15, C=5)
    th = cv2.morphologyEx(th, cv2.MORPH_DILATE, np.ones((2, 2), np.uint8), 1)
    return th

def is_digit_one_by_shape(bin_img):
    """启发式判 '1'：高瘦、前景稀疏、面积适中"""
    img = cv2.resize(bin_img, (56, 96), interpolation=cv2.INTER_AREA)
    img = cv2.medianBlur(img, 3)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    aspect = h / (w + 1e-6)              # 期望 >= 2.4
    area_ratio = area / (img.shape[0] * img.shape[1])  # 0.03~0.35
    fg_ratio = (img > 0).mean()          # 稀疏笔画（避免把 4/5/8 误判为 1）
    return (aspect >= 2.4) and (0.03 <= area_ratio <= 0.35) and (fg_ratio <= 0.28)

# -------------------- 模板库与滤波 --------------------
class DigitTemplateBank:
    """玩家数字模板库（0-9），用于 OCR 失败时兜底。"""
    def __init__(self):
        self.templates = {}   # {digit -> [bin_img,...]}
        self.size = (28, 48)

    def add(self, bin_img, value):
        if value is None:
            return
        d = int(value)
        if d < 0 or d > 9:
            return
        t = cv2.resize(bin_img, self.size, interpolation=cv2.INTER_AREA)
        self.templates.setdefault(d, []).append(t)

    def match(self, bin_img, thresh=0.62):
        if not self.templates:
            return None
        img = cv2.resize(bin_img, self.size, interpolation=cv2.INTER_AREA)
        best_d, best_s = None, -1.0
        for d, templ_list in self.templates.items():
            for tmpl in templ_list:
                s = float(cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED).max())
                if s > best_s:
                    best_s, best_d = s, d
        return best_d if best_s >= thresh else None

class PlayerHPFilter:
    """多数表决 + 粘滞保持（抗闪烁/掉帧）"""
    def __init__(self, hold_ms=500, require_k=3, window_size=5):
        self.hold_ms = hold_ms
        self.require_k = require_k
        self.window_size = window_size
        self.hist = collections.deque(maxlen=window_size)
        self.last_value = None
        self.last_ok_ts = 0.0

    def update(self, candidate):
        now = time.time() * 1000
        if candidate is not None:
            self.hist.append(candidate)
            vals, cnts = np.unique(self.hist, return_counts=True)
            maj = int(vals[np.argmax(cnts)])
            if cnts.max() >= self.require_k or (self.last_value is None):
                self.last_value = maj
                self.last_ok_ts = now
                return self.last_value
        if self.last_value is not None and (now - self.last_ok_ts) <= self.hold_ms:
            return self.last_value
        return None

digit_bank = DigitTemplateBank()
hp_filter  = PlayerHPFilter(hold_ms=500, require_k=3, window_size=5)

# -------------------- DEAD 快速判定（模板自举+匹配） --------------------
DEAD_TEMPLATE = None
DEAD_TSIZE = (180, 60)

def make_dead_template(badge_bgr):
    """从牌匾 ROI 自举 DEAD 模板"""
    roi = cv2.resize(badge_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, blockSize=17, C=5)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1)
    return cv2.resize(th, DEAD_TSIZE, interpolation=cv2.INTER_AREA)

def is_dead_fast(badge_bgr, thresh=0.72):
    """若已有模板，则用模板匹配快速判定 DEAD；无模板返回 None"""
    global DEAD_TEMPLATE
    if DEAD_TEMPLATE is None:
        return None
    roi = cv2.resize(badge_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, blockSize=17, C=5)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1)
    img = cv2.resize(th, DEAD_TSIZE, interpolation=cv2.INTER_AREA)
    s = float(cv2.matchTemplate(img, DEAD_TEMPLATE, cv2.TM_CCOEFF_NORMED).max())
    return s >= thresh

def detect_dead_text(img_bgr):
    """OCR 慢路径检测 DEAD（大小写不敏感）"""
    h, w = img_bgr.shape[:2]
    roi = cv2.resize(img_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    outs = []
    for inv in (False, True):
        if inv:
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_DILATE, np.ones((2, 2), np.uint8), 1)
        cfg = r'--psm 7 --oem 1 -c tessedit_char_whitelist=DEAdeead'
        raw = pytesseract.image_to_string(th, config=cfg).strip()
        outs.append(raw)
        if "DEAD" in raw.upper():
            return True, raw
    joined = " ".join(outs).upper()
    if re.search(r'D[E3][A4]D', joined):
        return True, joined
    return False, joined
def is_hp_badge_red(ply_roi_bgr, frac_thresh=0.22):
    """
    检测玩家HP牌匾是否为“红底”状态（1血闪烁）。
    返回 True/False。
    frac_thresh: 红色像素占比阈值（可在 0.18~0.30 间微调）
    """
    hsv = cv2.cvtColor(ply_roi_bgr, cv2.COLOR_BGR2HSV)
    # 红色两段（高饱和高明度，避免误检粉/褐）
    mask1 = cv2.inRange(hsv, (0,   120, 120), (10,  255, 255))
    mask2 = cv2.inRange(hsv, (170, 120, 120), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    # 形态学清噪
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    red_ratio = (mask > 0).mean()
    return red_ratio >= frac_thresh
# -------------------- 主流程 --------------------
def main():
    if not os.path.exists(CONF):
        raise FileNotFoundError(f"未找到 {CONF}，请先运行 calibrate_roi.py")
    roi = json.load(open(CONF, "r", encoding="utf-8"))
    assert "boss_hp_roi" in roi and "player_hp_roi" in roi, "ROI 配置缺少必要字段"

    cam = dxcam.create(output_color="BGR")
    cam.start(target_fps=60, video_mode=True)
    time.sleep(0.2)

    HOLD_MS = 400
    last_boss = {"cur": None, "max": None, "ts": 0.0}
    last_pcur, last_p_ts = None, 0.0
    dead_confirm_cnt = 0

    try:
        while True:
            frame = grab(cam)
            now_ms = time.time() * 1000

            # ----- Boss HP -----
            bx, by, bw, bh = roi["boss_hp_roi"]
            boss_roi = frame[by:by+bh, bx:bx+bw]
            cur, mx = None, None
            try:
                raw_boss = ocr_number_block(boss_roi)
                c, m = parse_boss_hp(raw_boss)
                if c is not None and m is not None and 0 <= c <= m:
                    cur, mx = c, m
            except Exception:
                cur, mx, raw_boss = None, None, ""
            if cur is None or mx is None:
                if last_boss["cur"] is not None and (now_ms - last_boss["ts"]) <= HOLD_MS:
                    cur, mx = last_boss["cur"], last_boss["max"]
            else:
                last_boss.update(cur=cur, max=mx, ts=now_ms)

            # ----- Player HP -----
            # ----- Player HP（先判红底=1；否则再 OCR/形状/模板；DEAD 最后确认） -----
            px, py, pw, ph = roi["player_hp_roi"]
            ply_roi = frame[py:py+ph, px:px+pw]

            # 0) DEAD 快速判定优先（你原逻辑不变）
            dead_fast = is_dead_fast(ply_roi, thresh=0.72)
            if dead_fast is True:
                pcur, raw_p = 0, "DEAD_fast"
                dead_confirm_cnt = 2
            else:
                # 1) 先看是否红底（只有1血才会红白闪烁）
                if is_hp_badge_red(ply_roi, frac_thresh=0.22):
                    pcur, raw_p = 1, "red_bg"
                else:
                    # 2) 非红底 → 正常流程：OCR → 形状启发式(判1) → 模板兜底
                    try:
                        pcur, raw_p = read_player_digit_only(ply_roi)
                    except Exception:
                        pcur, raw_p = None, "exception"

                    digit_bin = extract_digit_bin(ply_roi, right_ratio=0.50)
                    if pcur is None and is_digit_one_by_shape(digit_bin):
                        pcur, raw_p = 1, "shape_1"
                    if pcur is None:
                        p_tmpl = digit_bank.match(digit_bin, thresh=0.62)
                        if p_tmpl is not None:
                            pcur, raw_p = p_tmpl, "tmpl_match"

                # 3) DEAD 慢路径（首次OCR确认 + 自举模板），避免红底误判覆盖
                if pcur is None or pcur == 0:
                    is_dead_slow, raw_dead = detect_dead_text(ply_roi)
                    if is_dead_slow:
                        pcur = 0
                        raw_p = "DEAD_ocr"
                        if globals().get("DEAD_TEMPLATE", None) is None:
                            globals()["DEAD_TEMPLATE"] = make_dead_template(ply_roi)
                        dead_confirm_cnt = min(dead_confirm_cnt + 1, 3)
                    else:
                        dead_confirm_cnt = max(dead_confirm_cnt - 1, 0)
                else:
                    dead_confirm_cnt = 0

            # 4) 模板库扩充（有稳定数字时）
            if isinstance(pcur, int) and pcur >= 0:
                # 注意：此处也可用非红底时的 digit_bin 以免把红底形态存模板
                try:
                    digit_bin = extract_digit_bin(ply_roi, right_ratio=0.50)
                    digit_bank.add(digit_bin, pcur)
                except Exception:
                    pass

            # 5) 稳态输出
            pshow = hp_filter.update(pcur)

            # ----- 输出 -----
            print(f"[BossHP] {cur}/{mx} (raw='{raw_boss}')    [PlayerHP] {pcur} (raw='{raw_p}')    dead_cnt={dead_confirm_cnt}")

            # ----- 可视化 -----
            vis = frame.copy()
            cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (0, 0, 255), 2)
            cv2.rectangle(vis, (px, py), (px+pw, py+ph), (0, 255, 0), 2)
            cv2.imshow("debug", cv2.resize(vis, (960, 540)))

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):   # q / ESC
                break
            if k == ord('r'):
                try:
                    roi = json.load(open(CONF, "r", encoding="utf-8"))
                    print("[INFO] 已重载 ROI 配置")
                except Exception as e:
                    print(f"[WARN] 重载 ROI 失败: {e}")

    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
