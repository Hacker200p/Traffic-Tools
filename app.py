import uuid
import threading
import concurrent.futures
import importlib.util
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Traffic Density Comparator")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


def _load_license_plate_app():
    lpr_app_file = BASE_DIR / "license-plate-reader" / "app.py"
    if not lpr_app_file.exists():
        raise RuntimeError(f"License plate app not found: {lpr_app_file}")

    spec = importlib.util.spec_from_file_location("license_plate_reader_app", str(lpr_app_file))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load license plate app module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    lpr_app = getattr(module, "app", None)
    if lpr_app is None:
        raise RuntimeError("License plate app module has no FastAPI 'app' object")
    return lpr_app


app.mount("/lpr", _load_license_plate_app())

JOBS = {}  # job_id -> {"status","message","result"}


class RunRequest(BaseModel):
    roi1: list[list[int]]
    roi2: list[list[int]]
    model_path: str = "models/vehicle_best.pt"
    segment_m: float = 120.0
    lanes1: int = 1
    lanes2: int = 1
    conf: float = 0.15
    imgsz: int = 640
    device: str = ""
    smooth: int = 30
    vehicle_names: str = ""


def _save_bytes(data: bytes, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _extract_first_frame(video_path: Path, out_img: Path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {video_path}")
    cv2.imwrite(str(out_img), frame)
    h, w = frame.shape[:2]
    return w, h


def _parse_set(text: str):
    if not text or not text.strip():
        return None
    return {x.strip().lower() for x in text.split(",") if x.strip()}


def _inside_roi(cx, cy, poly):
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


def _run_density(job_id: str, video_path: Path, model: YOLO, roi_points, req: RunRequest, lanes: int, video_label: str):
    roi_poly = np.array(roi_points, dtype=np.int32)
    tracker = DeepSort(max_age=30, n_init=2, max_cosine_distance=0.35)
    keep_names = _parse_set(req.vehicle_names)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1
    hist = deque(maxlen=max(1, req.smooth))
    rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % 5 == 0 or frame_idx == total_frames:
            if "progress" not in JOBS[job_id]:
                JOBS[job_id]["progress"] = {}
            JOBS[job_id]["progress"][video_label] = f"{int((frame_idx/total_frames)*100)}% ({frame_idx}/{total_frames})"
            p1 = JOBS[job_id]["progress"].get("Video 1", "0%")
            p2 = JOBS[job_id]["progress"].get("Video 2", "0%")
            JOBS[job_id]["message"] = f"V1: {p1} | V2: {p2}"

        if frame_idx % 5 != 0:
            continue

        pred = model.predict(
            source=frame,
            conf=req.conf,
            imgsz=req.imgsz,
            device=req.device if req.device else None,
            verbose=False
        )[0]

        dets = []
        if pred.boxes is not None and len(pred.boxes) > 0:
            for b in pred.boxes:
                cls_id = int(b.cls.item())
                cls_name = str(model.names.get(cls_id, cls_id)).lower()
                if keep_names is not None and cls_name not in keep_names:
                    continue

                x1, y1, x2, y2 = b.xyxy[0].tolist()
                w, h = x2 - x1, y2 - y1
                if w <= 2 or h <= 2:
                    continue

                dets.append(([x1, y1, w, h], float(b.conf.item()), cls_name))

        tracks = tracker.update_tracks(dets, frame=frame)

        ids = set()
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            l, t, r, b = tr.to_ltrb()
            cx, cy = (l + r) / 2.0, (t + b) / 2.0
            if _inside_roi(cx, cy, roi_poly):
                ids.add(tr.track_id)

        n = len(ids)
        density = n / (req.segment_m / 1000.0)
        density_lane = density / max(1, lanes)
        hist.append(density_lane)
        smooth_density = float(sum(hist) / len(hist))

        rows.append({
            "frame": frame_idx,
            "time_sec": frame_idx / fps,
            "density": smooth_density
        })

    cap.release()
    return pd.DataFrame(rows)


def _worker(job_id: str, req: RunRequest):
    try:
        JOBS[job_id]["status"] = "running"
        job_dir = DATA_DIR / job_id
        v1 = job_dir / "video1.mp4"
        v2 = job_dir / "video2.mp4"

        model_path = Path(req.model_path)
        if not model_path.is_absolute():
            model_path = BASE_DIR / model_path
        if not model_path.exists():
            raise RuntimeError(f"Model not found: {model_path}")

        model_1 = YOLO(str(model_path))
        model_2 = YOLO(str(model_path))

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(_run_density, job_id, v1, model_1, req.roi1, req, req.lanes1, "Video 1")
            f2 = executor.submit(_run_density, job_id, v2, model_2, req.roi2, req, req.lanes2, "Video 2")
            d1 = f1.result()
            d2 = f2.result()

        JOBS[job_id]["message"] = "Merging results..."

        merged = pd.merge_asof(
            d1.sort_values("time_sec"),
            d2.sort_values("time_sec"),
            on="time_sec",
            direction="nearest",
            tolerance=0.1,
            suffixes=("_1", "_2")
        ).dropna(subset=["density_1", "density_2"])

        # Stateful Phase Logic: higher density -> GREEN, but with max GREEN time
        MAX_GREEN_SEC = 10.0
        MIN_GREEN_SEC = 2.0
        
        if len(merged) > 0:
            current_green = 1 if merged.iloc[0]["density_1"] >= merged.iloc[0]["density_2"] else 2
            time_in_phase = 0.0
            last_time = merged.iloc[0]["time_sec"]
            
            sig1, sig2 = [], []
            for _, row in merged.iterrows():
                t = row["time_sec"]
                d1 = row["density_1"]
                d2 = row["density_2"]
                
                time_in_phase += (t - last_time)
                last_time = t
                
                if current_green == 1:
                    if time_in_phase > MAX_GREEN_SEC:
                        current_green = 2
                        time_in_phase = 0.0
                    elif time_in_phase > MIN_GREEN_SEC and d2 > d1:
                        current_green = 2
                        time_in_phase = 0.0
                else:
                    if time_in_phase > MAX_GREEN_SEC:
                        current_green = 1
                        time_in_phase = 0.0
                    elif time_in_phase > MIN_GREEN_SEC and d1 > d2:
                        current_green = 1
                        time_in_phase = 0.0
                        
                if current_green == 1:
                    sig1.append("GREEN")
                    sig2.append("RED")
                else:
                    sig1.append("RED")
                    sig2.append("GREEN")
                    
            merged["signal_1"] = sig1
            merged["signal_2"] = sig2
        else:
            merged["signal_1"] = []
            merged["signal_2"] = []

        out_csv = job_dir / "signals_comparison.csv"
        merged.to_csv(out_csv, index=False)

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {
            "csv_url": f"/data/{job_id}/signals_comparison.csv",
            "timeline": merged[["time_sec", "density_1", "density_2", "signal_1", "signal_2"]].to_dict(orient="records")
        }
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["message"] = str(e)


@app.get("/")
def root():
    return RedirectResponse(url="/static/home.html")


@app.get("/traffic")
def traffic_home():
    return RedirectResponse(url="/static/index.html")


@app.get("/license")
def license_home():
    return RedirectResponse(url="/lpr/static/index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/upload")
async def upload(video1: UploadFile = File(...), video2: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    v1 = job_dir / "video1.mp4"
    v2 = job_dir / "video2.mp4"

    _save_bytes(await video1.read(), v1)
    _save_bytes(await video2.read(), v2)

    w1, h1 = _extract_first_frame(v1, job_dir / "frame1.jpg")
    w2, h2 = _extract_first_frame(v2, job_dir / "frame2.jpg")

    JOBS[job_id] = {"status": "uploaded", "message": "", "result": None}
    return {
        "job_id": job_id,
        "frame1_url": f"/data/{job_id}/frame1.jpg",
        "frame2_url": f"/data/{job_id}/frame2.jpg",
        "frame1_size": [w1, h1],
        "frame2_size": [w2, h2],
    }


@app.post("/api/run/{job_id}")
def run(job_id: str, req: RunRequest):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    if JOBS[job_id]["status"] == "running":
        raise HTTPException(status_code=400, detail="Job already running")

    t = threading.Thread(target=_worker, args=(job_id, req), daemon=True)
    t.start()
    return {"ok": True, "status": "running"}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]


@app.get("/api/result/{job_id}")
def result(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    if JOBS[job_id]["status"] != "done":
        raise HTTPException(status_code=400, detail="Result not ready")
    return JOBS[job_id]["result"]