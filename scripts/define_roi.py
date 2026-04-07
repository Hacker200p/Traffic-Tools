import cv2
import json
import argparse
import numpy as np
import ctypes
import tkinter as tk

try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

points_orig = []
base_img = None
disp_img = None
scale = 1.0

def get_screen_size():
    root = tk.Tk()
    root.withdraw()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return w, h

def redraw():
    global disp_img
    h, w = base_img.shape[:2]
    show_w, show_h = int(w * scale), int(h * scale)
    canvas = cv2.resize(base_img, (show_w, show_h), interpolation=cv2.INTER_AREA)

    pts_disp = [(int(x * scale), int(y * scale)) for x, y in points_orig]
    for i, p in enumerate(pts_disp):
        cv2.circle(canvas, p, 4, (0, 255, 0), -1)
        if i > 0:
            cv2.line(canvas, pts_disp[i - 1], p, (0, 255, 0), 2)

    if len(pts_disp) >= 3:
        cv2.polylines(canvas, [np.array(pts_disp, dtype=np.int32)], True, (255, 255, 0), 2)

    disp_img = canvas

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ox = int(round(x / scale))
        oy = int(round(y / scale))
        ox = max(0, min(base_img.shape[1] - 1, ox))
        oy = max(0, min(base_img.shape[0] - 1, oy))
        points_orig.append([ox, oy])
        redraw()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", required=True, help="Path to output roi.json")
    args = parser.parse_args()

    global base_img, scale
    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read first frame from video.")

    base_img = frame
    h, w = base_img.shape[:2]
    sw, sh = get_screen_size()
    scale = min((sw - 120) / w, (sh - 120) / h, 1.0)

    redraw()
    cv2.namedWindow("Define ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Define ROI", disp_img.shape[1], disp_img.shape[0])
    cv2.setMouseCallback("Define ROI", on_mouse)

    print("Left click: add point | U: undo | S: save | ESC/Q: quit")
    while True:
        cv2.imshow("Define ROI", disp_img)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("u") and points_orig:
            points_orig.pop()
            redraw()
        elif key == ord("s"):
            if len(points_orig) < 3:
                print("Need at least 3 points.")
                continue
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"roi": points_orig}, f, indent=2)
            print(f"Saved ROI: {args.out}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()