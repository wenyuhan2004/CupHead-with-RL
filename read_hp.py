# read_hp.py
import json, time, re, os, collections, warnings
import cv2, numpy as np, dxcam, pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
CONF = "cuphead_roi.json"

# ====== Parry/X 的 OCR 频率与缓存（全局） ======
PAR_OCR_EVERY = 2
X_OCR_EVERY   = 2

_parry_prev_sig = None
_parry_prev_val = 0
_parry_frame_idx = 0


_x_prev_sig = None
_x_prev_val = None
_x_frame_idx = 0
# -------------------- 通用抓帧 --------------------
def grab(cam):
    frame = cam.get_latest_frame()
    while frame is None:
        frame = cam.get_latest_frame()
        time.sleep(0.005)
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame

# -------------------- Boss HP OCR --------------------
def ocr_number_block(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cfg = "--psm 6 -c tessedit_char_whitelist=0123456789./"
    return pytesseract.image_to_string(th, config=cfg).strip()

def parse_boss_hp(text):
    text = text.strip()
    text = re.sub(r"^[^\d]+", "", text)
    m = re.search(r"(\d+(?:\.\d+)?)[^\d]+(\d+)", text)
    if not m:
        return None, None
    cur = float(m.group(1)); mx = float(m.group(2))
    if mx <= 0:
        return None, None
    cur = min(cur, mx)
    return cur, mx

# -------------------- 玩家 HP 识别（仅 0-4 单位数） --------------------
def read_player_digit_only(img_bgr):
    h, w = img_bgr.shape[:2]
    right_ratio = 0.50
    rw = max(12, int(w * right_ratio))
    digit_roi = img_bgr[:, w - rw : w]

    digit_roi = cv2.resize(digit_roi, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(digit_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    def try_ocr(bin_img):
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, np.ones((2, 2), np.uint8), 1)
        cfg = r'--psm 7 --oem 1 -c tessedit_char_whitelist=01234'
        raw = pytesseract.image_to_string(bin_img, config=cfg).strip()
        m = re.findall(r'[0-4]', raw)
        if m:
            val = int(m[-1])
            return val, raw
        return None, raw

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
    img = cv2.resize(bin_img, (56, 96), interpolation=cv2.INTER_AREA)
    img = cv2.medianBlur(img, 3)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    aspect = h / (w + 1e-6)
    area_ratio = area / (img.shape[0] * img.shape[1])
    fg_ratio = (img > 0).mean()
    return (aspect >= 2.4) and (0.03 <= area_ratio <= 0.35) and (fg_ratio <= 0.28)

class DigitTemplateBank:
    def __init__(self):
        self.templates = {}
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

# -------------------- DEAD 判定 --------------------
DEAD_TEMPLATE = None
DEAD_TSIZE = (180, 60)

def make_dead_template(badge_bgr):
    roi = cv2.resize(badge_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, blockSize=17, C=5)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1)
    return cv2.resize(th, DEAD_TSIZE, interpolation=cv2.INTER_AREA)

def is_dead_fast(badge_bgr, thresh=0.72):
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
    hsv = cv2.cvtColor(ply_roi_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0,   120, 120), (10,  255, 255))
    mask2 = cv2.inRange(hsv, (170, 120, 120), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    red_ratio = (mask > 0).mean()
    return red_ratio >= frac_thresh

# -------------------- Parry（仅数字 ROI） --------------------
def read_parry_count_only(img_bgr):
    global _parry_prev_sig, _parry_prev_val, _parry_frame_idx
    if img_bgr is None or img_bgr.size == 0:
        return None, "parry:empty"

    _parry_frame_idx += 1
    roi = cv2.resize(img_bgr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.bitwise_not(th)
    sig = int(th.sum())

    if _parry_prev_sig is not None and sig == _parry_prev_sig:
        return _parry_prev_val, "parry:cache-hit"

    if (_parry_frame_idx % PAR_OCR_EVERY) != 0:
        _parry_prev_sig = sig
        return _parry_prev_val, "parry:rate-limit"

    th = cv2.morphologyEx(th, cv2.MORPH_DILATE, np.ones((2,2), np.uint8), 1)
    cfg = r'--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789'
    raw = pytesseract.image_to_string(th, config=cfg).strip()
    nums = re.findall(r'(\d+)', raw)
    if not nums:
        _parry_prev_sig = sig
        return _parry_prev_val, "parry:none"
    val = int(nums[-1])
    if val + 5 < _parry_prev_val:
        _parry_prev_sig = sig
        return _parry_prev_val, f"parry:drop-ignored({val}->{_parry_prev_val})"
    val = max(0, min(val, 9999))
    _parry_prev_val = val
    _parry_prev_sig = sig
    return val, f"parry:'{raw}'"

# -------------------- X 坐标（仅数字 ROI，支持负号） --------------------
def read_xcoord_only(img_bgr):
    """
    读取 X 坐标（浮点/负号）。
    规则：
      - 若首个非空字符为 '-' → 取第一个带负号的数字（如 '-370-1' 取 -370）
      - 否则把 '-' 当分隔符 → 取减号之前的数字（如 '655-18' 取 655）
      - 若都不匹配 → 取全串中最长的数字段
    """
    global _x_prev_sig, _x_prev_val, _x_frame_idx
    if img_bgr is None or img_bgr.size == 0:
        return (_x_prev_val if _x_prev_val is not None else None), "x:empty"

    _x_frame_idx += 1

    roi = cv2.resize(img_bgr, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.bitwise_not(th)

    sig = int(th.sum())
    if _x_prev_sig is not None and sig == _x_prev_sig:
        return (_x_prev_val if _x_prev_val is not None else None), "x:cache-hit"

    if (_x_frame_idx % X_OCR_EVERY) != 0:
        _x_prev_sig = sig
        return (_x_prev_val if _x_prev_val is not None else None), "x:rate-limit"

    th = cv2.morphologyEx(th, cv2.MORPH_DILATE, np.ones((2,2), np.uint8), 1)
    cfg = r'--psm 7 --oem 1 -c tessedit_char_whitelist=-0123456789.'
    raw = pytesseract.image_to_string(th, config=cfg).strip()

    # 全部数字片段（含负号/小数）
    matches = re.findall(r'-?\d+(?:[.,]\d+)?', raw)
    if not matches:
        _x_prev_sig = sig
        return (_x_prev_val if _x_prev_val is not None else None), "x:none"

    s = raw.lstrip()  # 忽略前导空白
    if s.startswith('-'):
        # 情况A：首个非空字符为 '-' → 取第一个带负号的数字
        m = re.search(r'^-\d+(?:[.,]\d+)?', s)
        txt = m.group(0) if m else max(matches, key=len)
    else:
        # 情况B：内部 '-' 作为分隔符 → 取减号前面的数字
        parts = re.split(r'-', raw, maxsplit=1)
        pre_nums = re.findall(r'\d+(?:[.,]\d+)?', parts[0]) if parts else []
        txt = max(pre_nums, key=len) if pre_nums else max(matches, key=len)

    txt = txt.replace(',', '.')
    try:
        val = float(txt)
    except Exception:
        _x_prev_sig = sig
        return (_x_prev_val if _x_prev_val is not None else None), f"x:parse-fail('{raw}')"

    _x_prev_val = val
    _x_prev_sig = sig
    return val, f"x:'{raw}'"


# -------------------- 主流程 --------------------
def main():
    if not os.path.exists(CONF):
        raise FileNotFoundError(f"未找到 {CONF}，请先运行 calibrate_roi.py")
    roi = json.load(open(CONF, "r", encoding="utf-8"))
    if "boss_hp_roi" not in roi or "player_hp_roi" not in roi:
        raise KeyError("ROI 配置缺少 boss_hp_roi 或 player_hp_roi")

    if "parry_roi" not in roi:
        warnings.warn("ROI 配置缺少 parry_roi（格挡次数数字）。")
    if "xcoord_roi" not in roi:
        warnings.warn("ROI 配置缺少 xcoord_roi（X 坐标数字）。")

    cam = dxcam.create(output_color="BGR")
    cam.start(target_fps=60, video_mode=True)
    time.sleep(0.2)

    HOLD_MS = 400
    last_boss = {"cur": None, "max": None, "ts": 0.0}
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
            px, py, pw, ph = roi["player_hp_roi"]
            ply_roi = frame[py:py+ph, px:px+pw]

            dead_fast = is_dead_fast(ply_roi, thresh=0.72)
            if dead_fast is True:
                pcur, raw_p = 0, "DEAD_fast"
                dead_confirm_cnt = 2
            else:
                if is_hp_badge_red(ply_roi, frac_thresh=0.22):
                    pcur, raw_p = 1, "red_bg"
                else:
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

            # 模板库扩充
            if isinstance(pcur, int) and pcur >= 0:
                try:
                    digit_bin = extract_digit_bin(ply_roi, right_ratio=0.50)
                    digit_bank.add(digit_bin, pcur)
                except Exception:
                    pass

            pshow = hp_filter.update(pcur)

            # ----- Parry（若 ROI 存在） -----
            parry_info = "parry_na"
            if "parry_roi" in roi:
                sx, sy, sw, sh = roi["parry_roi"]
                parry_roi = frame[sy:sy+sh, sx:sx+sw]
                pval_raw, ptag = read_parry_count_only(parry_roi)
                parry_info = f"{pval_raw} (raw={pval_raw}, {ptag})"

            # ----- X 坐标（若 ROI 存在） -----
            x_info = "x_na"
            if "xcoord_roi" in roi:
                xx, xy, xw, xh = roi["xcoord_roi"]
                x_roi = frame[xy:xy+xh, xx:xx+xw]
                xval_raw, xtag = read_xcoord_only(x_roi)
                x_info = f"{xval_raw} (raw={xval_raw}, {xtag})"

            # ----- 输出 -----
            print(
                f"[BossHP] {cur}/{mx} (raw='{raw_boss}')    "
                f"[PlayerHP] {pcur} (raw='{raw_p}')    "
                f"[Parry] {parry_info}    "
                f"[X] {x_info}    "
                f"dead_cnt={dead_confirm_cnt}"
            )

            # 可视化（按需）
            SHOW_DEBUG = False
            if SHOW_DEBUG:
                vis = frame.copy()
                cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (0, 0, 255), 2)
                cv2.rectangle(vis, (px, py), (px+pw, py+ph), (0, 255, 0), 2)
                if "parry_roi" in roi:
                    sx, sy, sw, sh = roi["parry_roi"]
                    cv2.rectangle(vis, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
                if "xcoord_roi" in roi:
                    xx, xy, xw, xh = roi["xcoord_roi"]
                    cv2.rectangle(vis, (xx, xy), (xx+xw, xy+xh), (0, 255, 255), 2)
                cv2.imshow("debug", cv2.resize(vis, (960, 540)))
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27):
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
