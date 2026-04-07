import os
import cv2
import json
import csv
import argparse
import numpy as np
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import ctypes
import tkinter as tk


# Windows DPI fix (prevents mouse/display mismatch on some systems)
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_set(arg: str):
    if not arg or not arg.strip():
        return None
    return {x.strip().lower() for x in arg.split(",") if x.strip()}


def in_roi(cx, cy, roi_poly):
    return cv2.pointPolygonTest(roi_poly, (float(cx), float(cy)), False) >= 0


def get_screen_size():
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h


def fit_for_display(frame, margin=120):
    fh, fw = frame.shape[:2]
    sw, sh = get_screen_size()
    scale = min((sw - margin) / fw, (sh - margin) / fh, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt model")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--roi", required=True, help="Path to roi.json (format: {\"roi\": [[x,y], ...]})")
    parser.add_argument("--segment-m", type=float, required=True, help="Real road segment length in meters")
    parser.add_argument("--lanes", type=int, default=1, help="Number of lanes in ROI")
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    parser.add_argument("--device", default="", help="cuda:0 or cpu (empty=auto)")
    parser.add_argument("--smooth", type=int, default=30, help="Smoothing window (frames)")
    parser.add_argument("--vehicle-names", default="", help='Optional filter, e.g. "car,truck,bus,motorcycle". Empty = keep all classes')
    parser.add_argument("--csv-out", default=r"C:\Users\Asus\Desktop\traffic-tools\outputs\results.csv", help="CSV output path")
    parser.add_argument("--video-out", default=r"C:\Users\Asus\Desktop\traffic-tools\outputs\result.mp4", help="Annotated output video")
    parser.add_argument("--out-fps", type=float, default=0.0, help="Force output video fps. 0 = use input fps")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    parser.add_argument("--debug", action="store_true", help="Print class map and early frame detection stats")
    args = parser.parse_args()

    # Load ROI
    with open(args.roi, "r", encoding="utf-8") as f:
        roi_data = json.load(f)
    if "roi" not in roi_data:
        raise ValueError("ROI JSON must contain key 'roi'")
    roi_poly = np.array(roi_data["roi"], dtype=np.int32)
    if roi_poly.shape[0] < 3:
        raise ValueError("ROI needs at least 3 points")

    # Model and tracker
    model = YOLO(args.model)
    if args.debug:
        print("Model classes:", model.names)

    tracker = DeepSort(max_age=30, n_init=2, max_cosine_distance=0.35)
    keep_names = parse_set(args.vehicle_names)

    # Video input
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.out_fps if args.out_fps > 0 else in_fps

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame from video")

    h, w = frame.shape[:2]

    # Outputs
    ensure_parent_dir(args.csv_out)
    ensure_parent_dir(args.video_out)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(args.video_out, fourcc, fps, (w, h))
    if not out_vid.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {args.video_out}")

    if args.show:
        cv2.namedWindow("Vehicle Density", cv2.WINDOW_NORMAL)

    frame_idx = 0
    history = deque(maxlen=max(1, args.smooth))

    with open(args.csv_out, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["frame", "time_sec", "vehicles_in_roi", "density_veh_per_km", "density_veh_per_km_lane_smoothed"])

        while ok:
            frame_idx += 1

            pred = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device if args.device else None,
                verbose=False
            )[0]

            dets = []
            raw_count = 0

            if pred.boxes is not None and len(pred.boxes) > 0:
                for b in pred.boxes:
                    raw_count += 1
                    cls_id = int(b.cls.item())
                    cls_name = str(model.names.get(cls_id, cls_id)).lower()
                    conf = float(b.conf.item())

                    if keep_names is not None and cls_name not in keep_names:
                        continue

                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    bw, bh = x2 - x1, y2 - y1
                    if bw <= 2 or bh <= 2:
                        continue

                    dets.append(([x1, y1, bw, bh], conf, cls_name))

            if args.debug and frame_idx <= 10:
                print(f"frame={frame_idx}, raw={raw_count}, kept={len(dets)}")

            tracks = tracker.update_tracks(dets, frame=frame)

            active_ids = set()
            for tr in tracks:
                if not tr.is_confirmed():
                    continue
                l, t, r, b = tr.to_ltrb()
                cx, cy = (l + r) / 2.0, (t + b) / 2.0

                if in_roi(cx, cy, roi_poly):
                    active_ids.add(tr.track_id)
                    cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {tr.track_id}", (int(l), int(t) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Density
            n = len(active_ids)
            density = n / (args.segment_m / 1000.0)  # veh/km
            density_lane = density / max(args.lanes, 1)

            history.append(density_lane)
            smooth_density_lane = sum(history) / len(history)

            # Draw overlays
            cv2.polylines(frame, [roi_poly], True, (255, 255, 0), 2)
            cv2.putText(frame, f"N={n}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Density={density:.1f} veh/km", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Density/lane(smoothed)={smooth_density_lane:.1f}", (25, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            t_sec = frame_idx / in_fps
            writer.writerow([frame_idx, f"{t_sec:.3f}", n, f"{density:.3f}", f"{smooth_density_lane:.3f}"])

            out_vid.write(frame)

            if args.show:
                preview = fit_for_display(frame)
                cv2.imshow("Vehicle Density", preview)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

            ok, frame = cap.read()

    cap.release()
    out_vid.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Done. CSV saved: {args.csv_out}")
    print(f"Done. Video saved: {args.video_out}")


if __name__ == "__main__":
    main()