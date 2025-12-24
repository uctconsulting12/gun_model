"""
================================================================================
File Name: inference.py
================================================================================

Gun Detection System
- Invisible wrist-box filtering
- DeepSORT person tracking
- Gun → person association
- State-based alerts
- Single-frame (JSON stdin) + Video runner

This version preserves the original script and adds a new field `gun_holders`
(in the JSON return) that lists only the person track IDs that are holding guns.
It also writes an additional add-only JSONL file that contains only frames where
there are gun holders (person(s) with guns).
================================================================================
"""

"""
================================================================================
Gun Detection System – Optimized Box Accuracy Version
================================================================================
- Reduced overlapping person boxes
- More accurate gun detection
- Stricter wrist–gun association
- Same API + JSON contract
================================================================================
"""

"""
================================================================================
Gun Detection System – Optimized Box Accuracy + Person ID Version
================================================================================
- Stable person_id using DeepSort
- Correct pose → track association
- Reduced overlapping person boxes
- Accurate wrist–gun ownership
================================================================================
"""

"""
================================================================================
Gun Detection System – Optimized Box Accuracy Version
================================================================================
- Reduced overlapping person boxes
- More accurate gun detection
- Stricter wrist–gun association
- Same API + JSON contract
================================================================================
"""

# ===================== WARNING SUPPRESSION =====================
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ===================== STANDARD IMPORTS =====================

import base64
import sys
import threading
from datetime import datetime, timedelta
from typing import Dict, List
import time
import math

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ===================== LOGGER =====================
def log(msg: str):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)

# ===================== CONFIG =====================
DEFAULTS = {
    # Models
    "GUN_MODEL_PATH": r"gun_dd_f_best.pt",
    "POSE_MODEL_PATH": r"yolov8x-pose.pt",

    # Detection thresholds
    "CONF_THR_GUN": 0.35,
    "CONF_THR_POSE": 0.10,
    "CONF_THR_WRIST": 0.30,

    "IMG_SIZE": 1280,
    "NMS_IOU_GUN": 0.30,

    # Wrist association
    "WRIST_HALF": 22,
    "MIN_INTERSECTION_FRAC": 0.15,

    # Area filters
    "USE_ABSOLUTE_AREA": False,
    "GUN_MIN_AREA": 800,
    "GUN_MAX_AREA": 40000,

    # Image-relative filters
    "USE_IMAGE_RELATIVE": True,
    "GUN_MIN_FRAC": 0.0015,
    "GUN_MAX_FRAC": 0.15,

    # Wrist-relative filters
    "USE_RELATIVE_TO_WRIST": True,
    "GUN_TO_WRIST_MIN_RATIO": 0.5,
    "GUN_TO_WRIST_MAX_RATIO": 3.0,

    # Alerts
    "ENABLE_ALERTS": True,
    "ALERT_LEVELS": {
        "CRITICAL": 0.85,
        "HIGH": 0.65,
        "MEDIUM": 0.45,
        "LOW": 0.30,
    },
}

LEFT_WRI, RIGHT_WRI = 9, 10
VIDEO_PATH = r"E:\All_models\gun_detection\HORRIFIC_Body_Cam_Shows_Police_View_of_Uvalde_School_Shooting_480P.mp4"

# ===================== UTILITIES =====================
def _decode_b64_image(b64):
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _encode_b64_image(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()

def area(b): return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

def intersect(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * \
           max(0, min(a[3], b[3]) - max(a[1], b[1]))

def iou(a, b):
    inter = intersect(a, b)
    union = area(a) + area(b) - inter
    return 0 if union <= 0 else inter / union

def nms(boxes, scores, thr):
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        ious = np.array([iou(boxes[i], boxes[j]) for j in rest])
        order = rest[ious < thr]
    return keep

def compute_alert(score, cfg):
    if not cfg["ENABLE_ALERTS"]:
        return False, None
    for lvl, thr in sorted(cfg["ALERT_LEVELS"].items(), key=lambda x: -x[1]):
        if score >= thr:
            return True, lvl
    return False, None

# ===================== MODEL INIT =====================
def model_fn():
    log("Loading models...")
    gun = YOLO(DEFAULTS["GUN_MODEL_PATH"])
    pose = YOLO(DEFAULTS["POSE_MODEL_PATH"])

    tracker = DeepSort(
        max_age=15,
        n_init=5,
        max_iou_distance=0.6,
        nms_max_overlap=0.7
    )

    return {"gun": gun, "pose": pose, "tracker": tracker, "cfg": DEFAULTS}

# ===================== CORE INFERENCE =====================
def predict_frame(frame, meta, model):
    cfg = model["cfg"]
    tracker = model["tracker"]

    # -------- Pose Detection --------
    pose_res = model["pose"].predict(frame, conf=cfg["CONF_THR_POSE"], verbose=False)[0]
    boxes = pose_res.boxes.xyxy.cpu().numpy()
    scores = pose_res.boxes.conf.cpu().numpy()
    kpts = pose_res.keypoints.data.cpu().numpy()

    # Pose NMS (fix overlapping green boxes)
    keep = nms(boxes, scores, 0.5)
    boxes, scores, kpts = boxes[keep], scores[keep], kpts[keep]

    detections = []
    for b, s in zip(boxes, scores):
        w, h = b[2]-b[0], b[3]-b[1]
        if w < 40 or h < 80:
            continue
        detections.append(([b[0], b[1], w, h], float(s), "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

    persons = []
    for t in tracks:
        if not t.is_confirmed():
            continue
        l, t_, r, b = map(int, t.to_ltrb())
        persons.append([l, t_, r, b, t.track_id])

    # -------- Gun Detection --------
    gun_res = model["gun"].predict(frame, conf=cfg["CONF_THR_GUN"],
                                   imgsz=cfg["IMG_SIZE"], verbose=False)[0]

    gboxes = gun_res.boxes.xyxy.cpu().numpy()
    gscores = gun_res.boxes.conf.cpu().numpy()

    gkeep = nms(gboxes, gscores, cfg["NMS_IOU_GUN"])

    accepted = []
    out = frame.copy()

    for i in gkeep:
        gb = list(map(int, gboxes[i]))
        score = float(gscores[i])
        ga = area(gb)

        if cfg["USE_IMAGE_RELATIVE"]:
            frac = ga / (frame.shape[0] * frame.shape[1])
            if not (cfg["GUN_MIN_FRAC"] <= frac <= cfg["GUN_MAX_FRAC"]):
                continue

        holder_id = None
        min_dist = float("inf")

        for (l, t_, r, b, pid) in persons:
            for kp in kpts:
                for wi in (LEFT_WRI, RIGHT_WRI):
                    if kp[wi][2] < cfg["CONF_THR_WRIST"]:
                        continue
                    wx, wy = kp[wi][:2]
                    wb = [wx-cfg["WRIST_HALF"], wy-cfg["WRIST_HALF"],
                          wx+cfg["WRIST_HALF"], wy+cfg["WRIST_HALF"]]

                    if intersect(gb, wb) / max(1, ga) < cfg["MIN_INTERSECTION_FRAC"]:
                        continue

                    cx, cy = (l+r)/2, (t_+b)/2
                    dist = math.hypot(cx-wx, cy-wy)
                    if dist < min_dist:
                        min_dist = dist
                        holder_id = pid

        if holder_id is None:
            continue

        alert, level = compute_alert(score, cfg)

        accepted.append({
            "track_id": holder_id,
            "bbox": gb,
            "score": score,
            "alert": alert,
            "alert_level": level or ""
        })

        cv2.rectangle(out, (gb[0], gb[1]), (gb[2], gb[3]), (0, 0, 0), 2)

    # -------- Draw Persons --------
    for l, t_, r, b, pid in persons:
        color = (128, 0, 128) if any(a["track_id"] == pid for a in accepted) else (0, 200, 0)
        thickness = 3 if color[0] == 128 else 1
        cv2.rectangle(out, (l, t_), (r, b), color, thickness)

    holder_ids = sorted({a["track_id"] for a in accepted})

    return {
        "cam_id": meta.get("cam_id"),
        "org_id": meta.get("org_id"),
        "user_id": meta.get("user_id"),
        "guns": accepted,
        "persons_present": [p[4] for p in persons],
        "gun_holders": holder_ids,
        "annotated_frame": _encode_b64_image(out),
        "status": 0,
        "time_stamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }

# ===================== ENTRY =====================
_MODEL = model_fn()

def run_inference(payload):
    frame = _decode_b64_image(payload["encoding"])
    return predict_frame(frame, payload, _MODEL)

def live_inference(video):
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, buf = cv2.imencode(".jpg", frame)
        payload = {
            "cam_id": 1,
            "org_id": 1,
            "user_id": 1,
            "encoding": base64.b64encode(buf).decode()
        }
        res = run_inference(payload)
        img = cv2.imdecode(np.frombuffer(base64.b64decode(res["annotated_frame"]), np.uint8),
                           cv2.IMREAD_COLOR)
        cv2.imshow("Gun Detection", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_inference(VIDEO_PATH)

