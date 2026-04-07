# -*- coding: utf-8 -*-
"""
License Plate Reader - FastAPI Backend
Detects license plates using a YOLO model, crops & preprocesses the plate,
then extracts text using EasyOCR with Indian plate format post-processing.
Uses multi-variant preprocessing with voting to select the best OCR result.
"""

import uuid
import re
import traceback
from pathlib import Path

import math
import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="License Plate Reader")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# Lazy-load YOLO model
_model = None

def _get_model(path):
    global _model
    if _model is None:
        _model = YOLO(str(path))
    return _model


# Lazy-load EasyOCR reader (heavy init, only once)
_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=True)
    return _ocr_reader


# ── PIL-based annotation (ported from deep-license-plate-recognition) ──────────

_FONT_PATH = BASE_DIR / "assets" / "DejaVuSansMono.ttf"


def _annotate_image(img_bgr: np.ndarray, plates: list) -> np.ndarray:
    """
    Draw bounding boxes + plate-text labels onto *img_bgr* using PIL.
    Returns the annotated image as a BGR NumPy array.

    Styling mirrors deep-license-plate-recognition/plate_recognition.py
    draw_bb():
      - 3-pixel green (#00FF00) rectangle around each plate
      - White filled label box drawn above the bounding box
      - Plate text in black using DejaVuSansMono.ttf
    """
    # Convert BGR → RGB for PIL
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Try loading the TTF font; fall back to PIL default if missing
    try:
        font_size = max(14, img_bgr.shape[0] // 40)
        font = ImageFont.truetype(str(_FONT_PATH), font_size)
    except Exception:
        font = ImageFont.load_default()

    rect_color = (0, 255, 0)  # bright green

    for plate in plates:
        x1, y1, x2, y2 = plate["bbox"]
        label = plate.get("plate_text", "") or "PLATE"
        # Strip the [LOW CONF] suffix added by _clean_plate_text
        label = label.replace(" [LOW CONF]", "")
        is_reliable = plate.get("is_reliable", True)
        box_color = rect_color if is_reliable else (255, 215, 0)  # gold if low-conf

        # Draw 3-pixel-wide bounding box (three concentric rects)
        for pad in range(3):
            draw.rectangle(
                [(x1 - pad, y1 - pad), (x2 + pad, y2 + pad)],
                outline=box_color,
            )

        # Measure text size
        try:
            bbox_txt = font.getbbox(label)
            text_w = bbox_txt[2] - bbox_txt[0]
            text_h = bbox_txt[3] - bbox_txt[1]
        except AttributeError:
            # older Pillow without getbbox
            text_w, text_h = font.getsize(label)

        margin = math.ceil(0.05 * text_h)
        label_x0 = x1 - margin
        label_y0 = y1 - text_h - 2 * margin
        label_x1 = x1 + text_w + 2 * margin
        label_y1 = y1

        # White background for the label
        draw.rectangle([(label_x0, label_y0), (label_x1, label_y1)], fill="white")
        # Black text
        draw.text(
            (x1 + margin, y1 - text_h - margin),
            label,
            fill="black",
            font=font,
        )

    # Convert back to BGR NumPy array
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)



# ═══════════════════════════════════════════════════════════════════════════════
#  PERSPECTIVE CORRECTION — straighten slanted plates
# ═══════════════════════════════════════════════════════════════════════════════

def _order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]    # bottom-right has largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right has smallest diff
    rect[3] = pts[np.argmax(diff)] # bottom-left has largest diff
    return rect


def _deskew_plate(plate_img):
    """
    Straighten a slanted license plate using perspective transform.
    Falls back to original image if anything goes wrong.
    """
    try:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blurred, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return plate_img

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        img_area = plate_img.shape[0] * plate_img.shape[1]

        plate_cnt = None
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # Must be 4-sided AND cover at least 30% of image
            if len(approx) == 4 and cv2.contourArea(cnt) > 0.3 * img_area:
                plate_cnt = approx
                break

        if plate_cnt is None:
            return plate_img

        pts = plate_cnt.reshape(4, 2).astype("float32")
        rect = _order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        if maxWidth < 20 or maxHeight < 10:
            return plate_img

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(plate_img, M, (maxWidth, maxHeight))

        # Safety: if warped is blank or too small, fall back
        if warped is None or warped.size == 0:
            return plate_img
        if warped.shape[0] < 10 or warped.shape[1] < 20:
            return plate_img

        # Validate aspect ratio
        aspect = warped.shape[1] / warped.shape[0]
        if not (2.0 < aspect < 8.0):
            return plate_img

        return warped
    except Exception:
        return plate_img


def _remove_ind_strip(plate_img):
    """
    Detect and remove IND strip using white gap detection.
    Falls back to original if no clear gap found.
    """
    h, w = plate_img.shape[:2]
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Only look at leftmost 20%
    search_width = int(w * 0.20)
    left_region = gray[:, :search_width]
    col_means = np.mean(left_region, axis=0)
    
    # Find white columns (background) in left region
    white_cols = np.where(col_means > 210)[0]
    
    if len(white_cols) < 3:
        # No clear white gap found — don't crop
        return plate_img
    
    # Only crop if white gap is clearly after the IND box
    # IND box is typically in first 10% of plate width
    valid_white = white_cols[white_cols > int(search_width * 0.4)]
    
    if len(valid_white) == 0:
        return plate_img
    
    crop_x = int(valid_white[-1])
    cropped = plate_img[:, crop_x:]
    
    # Safety check — don't crop more than 18% of plate
    if crop_x > int(w * 0.18):
        return plate_img
        
    if cropped.shape[1] > cropped.shape[0] * 2:
        return cropped
    
    return plate_img


# ===============================================================================
#  PREPROCESSING VARIANTS
# ===============================================================================

def _prep_simple(plate_img):
    """
    SIMPLE preprocessing from reference code.
    Grayscale + basic threshold (BINARY_INV at 64).
    Works great for clean, high-contrast plates.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
    # Invert back to black text on white background for OCR
    thresh = cv2.bitwise_not(thresh)
    padded = cv2.copyMakeBorder(thresh, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    return padded


def _prep_light(plate_img):
    """
    LIGHT preprocessing for clean plates.
    Just grayscale + resize x2 + bilateral filter. NO threshold.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    padded = cv2.copyMakeBorder(gray, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    return padded


def _prep_thresh(plate_img):
    """
    Adaptive threshold for low-contrast/blurry plates.
    Includes fallback if threshold produces blank image.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(blur, -1, kernel)

    thresh = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15, 3
    )

    # Fallback: if threshold produced blank/mostly-white image
    if thresh is None or thresh.sum() == 0 or np.mean(thresh > 127) > 0.95:
        return cv2.copyMakeBorder(sharp, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)

    # Ensure black text on white background
    if np.mean(thresh > 127) < 0.5:
        thresh = cv2.bitwise_not(thresh)

    padded = cv2.copyMakeBorder(thresh, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    return padded


def _prep_sharp_gray(plate_img):
    """Sharpened grayscale, no binarization."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(blur, -1, kernel)
    return cv2.copyMakeBorder(sharp, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)


def _prep_contrast(plate_img):
    """CLAHE contrast-boosted grayscale."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.copyMakeBorder(enhanced, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)


def _prep_color(plate_img):
    """Original color just resized."""
    resized = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(resized, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(255, 255, 255))


# ===============================================================================
#  FORMAT VALIDATION (from reference code)
# ===============================================================================

# Character confusion maps (from reference util.py)
CHAR_TO_INT = {'O': '0', 'Q': '0', 'D': '0',
               'I': '1', 'L': '1',
               'Z': '2', 'E': '3',
               'A': '4', 'S': '5',
               'G': '6', 'T': '7', 'B': '8'}

INT_TO_CHAR = {'0': 'O', '1': 'I', '2': 'Z',
               '3': 'B', '4': 'A', '5': 'S',
               '6': 'G', '7': 'T', '8': 'B', '9': 'G'}

# All valid Indian state codes
VALID_STATE_CODES = {
    'AP', 'AR', 'AS', 'BR', 'CH', 'DL', 'GA', 'GJ', 'HR', 'HP',
    'JK', 'JH', 'KA', 'KL', 'LD', 'MP', 'MH', 'MN', 'ML', 'MZ',
    'NL', 'OD', 'OR', 'PY', 'PB', 'RJ', 'SK', 'TN', 'TR', 'UP', 'WB',
    'TS', 'UK', 'CT', 'DN', 'DD', 'AN'
}

# Odisha district codes (01-40) — extend with other states as needed
VALID_DISTRICT_CODES = {
    'OD': set(f"{i:02d}" for i in range(1, 41)),   # OD01–OD40
}


def _looks_like_plate(text):
    """
    Check if text could be an Indian license plate.
    Indian format: AA00AA0000 (10 chars) or AA00A0000 (9 chars)
    At minimum: starts with ~2 letters, has digits, etc.
    Returns True if it looks plausible.
    """
    clean = re.sub(r"[^A-Za-z0-9]", "", text.upper())
    if len(clean) < 8 or len(clean) > 12:
        return False

    # Check first 2 chars can be letters (allow common digit->letter confusions)
    for c in clean[:2]:
        if not (c.isalpha() or c in INT_TO_CHAR):
            return False

    # Check positions 2,3 can be digits (allow common letter->digit confusions)
    for c in clean[2:4]:
        if not (c.isdigit() or c in CHAR_TO_INT):
            return False

    return True


# ===============================================================================
#  OCR (EasyOCR with format-aware scoring)
# ===============================================================================

def _prep_carbon_fiber(plate_img):
    """
    Special preprocessing for carbon fiber texture HSRP plates.
    Uses morphological operations to separate text from texture.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # Morphological tophat to separate text from background texture
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    # Boost contrast
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(tophat)
    
    # Threshold
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_small)
    
    return cv2.copyMakeBorder(thresh, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)


def _run_ocr(reader, img):
    """
    Run EasyOCR on one image.
    """
    try:
        if img is None or (hasattr(img, 'size') and img.size == 0):
            return "", 0.0
        results = reader.readtext(
            img,
            detail=1,
            paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            width_ths=0.9,      # merge horizontally close boxes
            ycenter_ths=0.5,    # merge boxes on same line
            min_size=10,
        )
    except Exception:
        return "", 0.0

    if not results or len(results) == 0:
        return "", 0.0

    # ✅ Sort detections left-to-right by x-coordinate of bounding box
    results = sorted(results, key=lambda r: r[0][0][0])

    # Combine detections
    all_texts = []
    all_confs = []
    for item in results:
        _, text, conf = item
        text = text.strip().upper().replace(" ", "")
        if text:
            all_texts.append(text)
            all_confs.append(float(conf))
            
    combined = "".join(all_texts)
    combined_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
    return combined, combined_conf


def _run_ocr_direct(reader, img):
    """
    Bypass EasyOCR's text detector and run recognizer directly on full image.
    Use when the plate is already cropped and clean.
    """
    try:
        # EasyOCR's recognize method skips detection
        result = reader.recognize(
            img,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            detail=1,
        )
        if not result:
            return "", 0.0
        text = result[0][1].upper().replace(" ", "")
        conf = float(result[0][2])
        return text, conf
    except Exception as e:
        print(f"Direct OCR error: {e}")
        return "", 0.0


# ===============================================================================
#  MULTI-OCR VOTING - run on 6 variants, pick best
# ===============================================================================

def _run_multi_ocr(reader, plate_crop):
    
    # Step 1: Get EasyOCR results
    variant_funcs = [
        ("light",        _prep_light),
        ("thresh",       _prep_thresh),
        ("contrast",     _prep_contrast),
        ("carbon_fiber", _prep_carbon_fiber),  # ✅ NEW
    ]
    
    variants = []
    for name, func in variant_funcs:
        try:
            img = func(plate_crop)
            if img is not None and img.size > 0:
                variants.append((name, img))
        except Exception:
            continue

    if not variants:
        variants = [("raw", plate_crop)]
    
    all_results = []
    for name, img in variants:
        text, conf = _run_ocr(reader, img)
        clean = re.sub(r"[^A-Z0-9]", "", text)
        if 8 <= len(clean) <= 12:
            _, state_ok = _correct_state_code(clean[:2])
            all_results.append((text, conf, img, state_ok))

    # ✅ Also try direct recognition on preprocessed images
    for name, img in variants:
        text, conf = _run_ocr_direct(reader, img)
        clean = re.sub(r"[^A-Z0-9]", "", text)
        if 8 <= len(clean) <= 12:
            _, state_ok = _correct_state_code(clean[:2])
            all_results.append((text, conf, img, state_ok))
    
    if not all_results:
        return "", 0.0, variants[0][1]
    
    all_results.sort(key=lambda x: (x[3], x[1]), reverse=True)
    best_text, best_conf, best_img = all_results[0][0], all_results[0][1], all_results[0][2]

    return best_text, best_conf, best_img


# ===============================================================================
#  POST-PROCESSING - Indian plate position correction
# ===============================================================================

def _correct_state_code(raw_two_chars):
    """
    Given the first 2 chars from OCR, return the closest valid state code.
    Returns (corrected_code, is_valid).
    """
    candidate = raw_two_chars.upper()
    if candidate in VALID_STATE_CODES:
        return candidate, True
        
    fixed = ""
    for c in candidate:
        fixed += INT_TO_CHAR.get(c, c)
    if fixed in VALID_STATE_CODES:
        return fixed, True
        
    for code in VALID_STATE_CODES:
        mismatches = sum(1 for a, b in zip(candidate, code) if a != b)
        if mismatches <= 1:
            return code, True
            
    return candidate, False

def _correct_district_code(raw_two_chars, state_code):
    """
    Given chars at positions 2-3 from OCR, return valid district number.
    """
    candidate = raw_two_chars.upper()
    fixed = ""
    for c in candidate:
        fixed += CHAR_TO_INT.get(c, c)
        
    if not fixed.isdigit():
        return candidate, False
        
    known_districts = VALID_DISTRICT_CODES.get(state_code)
    if known_districts is None:
        return fixed, fixed.isdigit()
        
    if fixed in known_districts:
        return fixed, True
        
    num = int(fixed)
    valid_nums = [int(d) for d in known_districts]
    closest = min(valid_nums, key=lambda x: abs(x - num))
    return f"{closest:02d}", False

def _find_best_window(text):
    best_window = text[:10]
    best_score = -1
    for start in range(len(text) - 9):
        for length in [9, 10]:
            if start + length > len(text):
                continue
            window = text[start:start + length]
            score = 0
            if len(window) >= 2:
                score += sum(1 for c in window[:2] if c.isalpha())
            if len(window) >= 4:
                score += sum(1 for c in window[2:4] if c.isdigit())
            if length == 10 and len(window) >= 6:
                score += sum(1 for c in window[4:6] if c.isalpha())
            if len(window) >= length:
                score += sum(1 for c in window[-4:] if c.isdigit())
            score += 0.5 if length == 10 else 0
            if score > best_score:
                best_score = score
                best_window = window
    return best_window

def _fix_indian_plate(text):
    """
    Fix Indian plate by CHARACTER POSITION and KNOWN CODES.
    """
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    
    if len(text) > 10:
        text = _find_best_window(text)
    if len(text) < 9:
        return text, False, False

    if len(text) == 10:
        raw_state    = text[:2]
        raw_district = text[2:4]
        raw_series   = text[4:6]
        raw_number   = text[6:10]
    else:  # length 9
        raw_state    = text[:2]
        raw_district = text[2:4]
        raw_series   = text[4:5]
        raw_number   = text[5:9]

    def fix_letters(s):
        result = ""
        for c in s:
            if c == "0": result += "O"
            elif c == "1": result += "I"
            elif c == "2": result += "Z"
            elif c == "3": result += "B"
            elif c == "4": result += "A"
            elif c == "5": result += "S"
            elif c == "6": result += "G"
            elif c == "7": result += "T"
            elif c == "8": result += "B"
            elif c == "9": result += "G"
            else: result += c
        return result

    def fix_digits(s):
        result = ""
        for c in s:
            if c in ("O", "Q", "D"): result += "0"
            elif c in ("I", "L"): result += "1"
            elif c == "Z": result += "2"
            elif c == "E": result += "3"
            elif c == "A": result += "4"
            elif c == "S": result += "5"
            elif c == "G": result += "6"
            elif c == "T": result += "7"
            elif c == "B": result += "8"
            else: result += c
        return result

    state_code, state_valid = _correct_state_code(raw_state)
    district_code, district_valid = _correct_district_code(raw_district, state_code)

    series = fix_letters(raw_series)
    number = fix_digits(raw_number)

    return state_code + district_code + series + number, state_valid, district_valid


def _clean_plate_text(raw_text, confidence):
    """
    Master cleaning pipeline:
    """
    text = raw_text.upper().replace(" ", "")

    for old, new in {"|": "I", "!": "1", "@": "A", "$": "S", "&": "8"}.items():
        text = text.replace(old, new)

    text = re.sub(r"[^A-Z0-9]", "", text)

    if not text:
        return "", False, False

    corrected, s_valid, d_valid = _fix_indian_plate(text)

    if confidence < 0.5:
        corrected = corrected + " [LOW CONF]"

    return corrected, s_valid, d_valid


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/detect")
async def detect_plate(
    image: UploadFile = File(...),
    model_path: str = Form("models/license_best.pt"),
    conf: float = Form(0.25),
):
    try:
        """
        1. Run YOLO detection on the uploaded image.
        2. For each plate: crop → deskew → preprocess → OCR → format correct.
        3. Return plate texts + annotated image URL + cropped plate URLs.
        """
        img_bytes = await image.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        mp = Path(model_path)
        if not mp.is_absolute():
            mp = PARENT_DIR / mp
        if not mp.exists():
            raise HTTPException(status_code=400, detail=f"Model not found: {mp}")

        model = _get_model(mp)
        results = model.predict(source=img, conf=conf, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            raise HTTPException(status_code=404, detail="No license plate detected in the image")

        job_id = str(uuid.uuid4())[:8]
        job_dir = DATA_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        reader = _get_ocr_reader()
        plates = []

        for idx, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_confidence = float(box.conf.item())

            # Crop with dynamic margin
            h_img, w_img = img.shape[:2]
            margin = int(0.02 * max(w_img, h_img))
            cx1 = max(0, x1 - margin)
            cy1 = max(0, y1 - margin)
            cx2 = min(w_img, x2 + margin)
            cy2 = min(h_img, y2 + margin)
            plate_crop = img[cy1:cy2, cx1:cx2]

            if plate_crop.size == 0:
                continue

            # Step 1: Perspective correction (straighten slanted plate)
            deskewed = _deskew_plate(plate_crop)

            # 🔥 SUPER RESOLUTION (CRITICAL)
            deskewed = cv2.resize(deskewed, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

            # ✅ NEW: Remove IND strip from HSRP plates
            deskewed = _remove_ind_strip(deskewed)

            # Step 2: Multi-OCR voting (3 variants)
            raw_text, ocr_conf, best_processed = _run_multi_ocr(reader, deskewed)

            # Step 3: Post-process — clean + Indian format + confidence filter
            plate_text, state_valid, district_valid = _clean_plate_text(raw_text, ocr_conf)
            is_reliable = ocr_conf >= 0.5

            # Save crop + processed images
            crop_path = job_dir / f"plate_{idx}_crop.jpg"
            proc_path = job_dir / f"plate_{idx}_processed.jpg"
            cv2.imwrite(str(crop_path), plate_crop)
            cv2.imwrite(str(proc_path), best_processed)

            plates.append({
                "index": idx,
                "bbox": [x1, y1, x2, y2],
                "detection_confidence": round(det_confidence, 3),
                "ocr_confidence": round(ocr_conf, 3),
                "is_reliable": bool(is_reliable),
                "state_valid": bool(state_valid),
                "district_valid": bool(district_valid),
                "raw_ocr_text": raw_text,
                "plate_text": plate_text,
                "crop_url": f"/data/{job_id}/plate_{idx}_crop.jpg",
                "processed_url": f"/data/{job_id}/plate_{idx}_processed.jpg",
            })

        # Build annotated image with PIL (sharp TTF labels, green boxes)
        annotated_bgr = _annotate_image(img, plates)

        annotated_path = job_dir / "annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated_bgr)
        original_path = job_dir / "original.jpg"
        cv2.imwrite(str(original_path), img)

        return JSONResponse({
            "job_id": job_id,
            "total_plates": len(plates),
            "plates": plates,
            "annotated_url": f"/data/{job_id}/annotated.jpg",
            "original_url": f"/data/{job_id}/original.jpg",
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

