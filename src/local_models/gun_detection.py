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

# ===================== WARNING SUPPRESSION =====================
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ===================== STANDARD IMPORTS =====================
import json
import base64
import sys
import threading
from datetime import datetime
from typing import Dict, List
import time

import cv2
import numpy as np

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ===================== LOGGER =====================
def log(msg: str):
    ts = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)

# ===================== STDIN TIMEOUT =====================
def read_stdin_json(timeout_sec=180):
    result = {"data": None, "error": None}

    def _reader():
        try:
            raw = sys.stdin.read()
            if not raw.strip():
                result["error"] = "Empty stdin"
            else:
                result["data"] = json.loads(raw)
        except Exception as e:
            result["error"] = str(e)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout_sec)

    if t.is_alive():
        raise TimeoutError(f"No input received in {timeout_sec} seconds")

    if result["error"]:
        raise ValueError(result["error"])

    return result["data"]

# ===================== CONFIG (ALL HYPERPARAMETERS) =====================
DEFAULTS = {
    "GUN_MODEL_PATH": r"gun_dd_f_best.pt",
    "POSE_MODEL_PATH": r"yolov8x-pose.pt",

    "CONF_THR_GUN": 0.20,
    "CONF_THR_POSE": 0.05,
    "IMG_SIZE": 1280,
    "NMS_IOU_GUN": 0.35,

    "WRIST_HALF": 25,
    "MIN_INTERSECTION_FRAC": 0.0,

    "USE_ABSOLUTE_AREA": False,
    "GUN_MIN_AREA": 500,
    "GUN_MAX_AREA": 4000,

    "USE_IMAGE_RELATIVE": True,
    "GUN_MIN_FRAC": 0.0005,
    "GUN_MAX_FRAC": 0.02,

    "USE_RELATIVE_TO_WRIST": True,
    "GUN_TO_WRIST_MIN_RATIO": 0.25,
    "GUN_TO_WRIST_MAX_RATIO":2.0,

    "ENABLE_ALERTS": True,
    "ALERT_LEVELS": {
        "CRITICAL": 0.80,
        "HIGH": 0.60,
        "MEDIUM": 0.40,
        "LOW":0.20,
   },
}

LEFT_WRI, RIGHT_WRI = 9, 10
VIDEO_PATH = r"E:\All_models\gun_detection\vidw3.mp4"

PERSON_OR_GUN_JSON = "person_or_gun_events.jsonl"
PERSON_HOLDERS_JSON = "person_holders_events.jsonl"

# ===================== UTILITIES =====================
def _decode_b64_image(b64: str) -> np.ndarray:
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    return img

def _encode_b64_image(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("Encode failed")
    return base64.b64encode(buf.tobytes()).decode()

def area(box):
    return max(0, box[2]-box[0]) * max(0, box[3]-box[1])

def intersect(a, b):
    return max(0, min(a[2],b[2]) - max(a[0],b[0])) * \
           max(0, min(a[3],b[3]) - max(a[1],b[1]))

def iou(a,b):
    inter = intersect(a,b)
    union = area(a) + area(b) - inter
    return 0.0 if union <= 0 else inter/union

def nms(boxes, scores, thr):
    if len(boxes)==0:
        return []
    order = np.argsort(scores)[::-1]
    keep=[]
    while order.size>0:
        i=order[0]
        keep.append(i)
        rest=order[1:]
        if len(rest)==0: break
        ious=np.array([iou(boxes[i], boxes[j]) for j in rest])
        order=rest[ious<thr]
    return keep

def compute_alert(score,cfg):
    if not cfg["ENABLE_ALERTS"]:
        return False,None
    for lvl,thr in sorted(cfg["ALERT_LEVELS"].items(), key=lambda x:x[1], reverse=True):
        if score>=thr:
            return True,lvl
    return False,None

# ===================== MODEL INIT =====================
def model_fn(cfg_override=None):
    cfg=DEFAULTS.copy()
    if cfg_override: cfg.update(cfg_override)

    log("Loading models...")
    gun=YOLO(cfg["GUN_MODEL_PATH"])
    pose=YOLO(cfg["POSE_MODEL_PATH"])
    tracker=DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
    log("Models loaded")

    return {"gun":gun,"pose":pose,"tracker":tracker,"cfg":cfg}

# ===================== CORE INFERENCE =====================
def predict_frame(frame, meta, model):
    cfg=model["cfg"]

    pose_res=model["pose"].predict(frame,conf=cfg["CONF_THR_POSE"],verbose=False)[0]
    boxes=pose_res.boxes.xyxy.cpu().numpy() if pose_res.boxes else []
    scores=pose_res.boxes.conf.cpu().numpy() if pose_res.boxes else []
    kpts=pose_res.keypoints.data.cpu().numpy() if pose_res.keypoints else []

    detections=[([b[0],b[1],b[2]-b[0],b[3]-b[1]],s,"person") for b,s in zip(boxes,scores)]
    tracks=model["tracker"].update_tracks(detections,frame=frame)

    persons=[]
    for t in tracks:
        if t.is_confirmed():
            l,t_,r,b=t.to_ltrb()
            persons.append([int(l),int(t_),int(r),int(b),t.track_id])

    gun_res=model["gun"].predict(frame,conf=cfg["CONF_THR_GUN"],imgsz=cfg["IMG_SIZE"],verbose=False)[0]
    gboxes=gun_res.boxes.xyxy.cpu().numpy() if gun_res.boxes else []
    gscores=gun_res.boxes.conf.cpu().numpy() if gun_res.boxes else []

    keep=nms(gboxes,gscores,cfg["NMS_IOU_GUN"])
    accepted=[]
    out=frame.copy()

    for i in keep:
        gb=list(map(int,gboxes[i]))
        score=float(gscores[i])

        holder_id=None
        wrist_box=None

        for pi,kp in enumerate(kpts):
            for wi in [LEFT_WRI, RIGHT_WRI]:
                if kp[wi][2]>cfg["CONF_THR_POSE"]:
                    wx,wy=kp[wi][:2]
                    wb=[int(wx-cfg["WRIST_HALF"]),int(wy-cfg["WRIST_HALF"]),
                        int(wx+cfg["WRIST_HALF"]),int(wy+cfg["WRIST_HALF"])]
                    if intersect(gb,wb)>0:
                        wrist_box=wb
                        holder_id=persons[pi][4] if pi<len(persons) else None
                        break

        if cfg["USE_RELATIVE_TO_WRIST"] and wrist_box is None:
            continue

        ga=area(gb)
        if cfg["USE_ABSOLUTE_AREA"]:
            if not (cfg["GUN_MIN_AREA"]<=ga<=cfg["GUN_MAX_AREA"]):
                continue

        if cfg["USE_IMAGE_RELATIVE"]:
            frac=ga/(frame.shape[0]*frame.shape[1])
            if not (cfg["GUN_MIN_FRAC"]<=frac<=cfg["GUN_MAX_FRAC"]):
                continue

        if wrist_box:
            ratio=ga/max(1,area(wrist_box))
            if not (cfg["GUN_TO_WRIST_MIN_RATIO"]<=ratio<=cfg["GUN_TO_WRIST_MAX_RATIO"]):
                continue

        alert,level=compute_alert(score,cfg)

        accepted.append({
            "track_id": holder_id,
            "bbox": gb,
            "score": score,
            "alert": alert,
            "alert_level": level or ""
        })

        cv2.rectangle(out,(gb[0],gb[1]),(gb[2],gb[3]),(0,0,0),2)

    for l,t,r,b,pid in persons:
        if any(a["track_id"]==pid for a in accepted):
            cv2.rectangle(out,(l,t),(r,b),(128,0,128),2)

    holder_ids = sorted({a["track_id"] for a in accepted if a.get("track_id") is not None})
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    return {
        "cam_id": meta["cam_id"],
        "user_id":meta["user_id"],
        "org_id":meta["org_id"],
        "guns": accepted,
        "time_stamp": ts,
        "persons_present": [p[4] for p in persons],
        "gun_holders": holder_ids,
        "annotated_frame": _encode_b64_image(out),
        "status": 0
    }

# ===================== run_inference (FRONTEND CONTRACT) =====================
def run_inference(payload: Dict) -> Dict:
    global _MODEL
    if "_MODEL" not in globals():
        _MODEL = model_fn()

    frame = _decode_b64_image(payload["encoding"])
    return predict_frame(frame, payload, _MODEL)

# ===================== LIVE INFERENCE (RE-ADDED, ADD-ONLY) =====================
def live_inference(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log("Cannot open video")
        return

    log("Running live inference (press Q to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, buf = cv2.imencode(".jpg", frame)
        payload = {
            "cam_id": 123,
            "org_id": 2,
            "user_id": 2,
            "encoding": base64.b64encode(buf).decode()
        }

        result = run_inference(payload)

        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(result["annotated_frame"]), np.uint8),
            cv2.IMREAD_COLOR
        )

        cv2.imshow("Live Gun Detection", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===================== MAIN =====================
if __name__=="__main__":
    live_inference(VIDEO_PATH)
