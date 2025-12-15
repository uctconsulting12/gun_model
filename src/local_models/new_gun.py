"""
inference_single_frame.py

Single-frame pipeline for gun + wrist detection.

Provides:
 - helper utils (geometry, NMS, keypoint helpers)
 - model_fn(model_dir_or_config)
 - input_frame_fn(request_body, content_type="application/json")
 - predict_frame_fn(input_data, model)
 - output_frame_fn(prediction)

Input JSON example:
{
  "cam_id": 123,
  "org_id": 2,
  "user_id": 2,
  "encoding": "<base64_jpeg_data>"
}

Output example (from output_frame_fn):
{
  "cam_id": 123,
  "guns": [ { "bbox": [...], "score":..., "matched_wrist": {...}, "holder": {"person_idx":..., "bbox": [...] } }, ... ],
  "annotated_frame": "<base64_jpeg>",
  "status": 0
}
"""

import os
import json
import base64
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Install ultralytics: pip install ultralytics") from e

# ---------------- Defaults (editable) ----------------
DEFAULTS = {
    "GUN_MODEL_PATH": "/content/gun_dd_f_best.pt",
    "POSE_MODEL_PATH": "/content/yolov8x-pose.pt",
    "IMG_SIZE": 1280,
    "CONF_THR_GUN": 0.05,
    "CONF_THR_POSE": 0.05,
    "MIN_INTERSECTION_FRAC": 0.0,
    "WRIST_HALF": 25,
    "WRIST_BOX_FRACTION_OF_TORSO": 0.18,
    "WRIST_BOX_MIN_HALF": 15,
    "GUN_SCALE": 1.10,
    "GUN_PAD_PIXELS": 0,
    "GUN_MIN_AREA": 500,
    "GUN_MAX_AREA": 4000,
    "GUN_MIN_FRAC": 0.0005,
    "GUN_MAX_FRAC": 0.02,
    "USE_ABSOLUTE_AREA": False,
    "USE_IMAGE_RELATIVE": True,
    "USE_RELATIVE_TO_WRIST": False,
    "GUN_TO_WRIST_MIN_RATIO": 0.25,
    "GUN_TO_WRIST_MAX_RATIO": 2.0,
    "NMS_IOU_GUN": 0.35,
    "DRAW_WRISTS": True,
    "VERBOSE": True,
}

# keypoint indices (yolov8 pose format typical indices)
LEFT_SHO, RIGHT_SHO = 5, 6
LEFT_WRI, RIGHT_WRI = 9, 10

# ---------------- Geometry helpers ----------------
def area_of_box(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return int(w * h)

def intersect_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    return int(iw * ih)

def scale_box(
    box: Tuple[int, int, int, int],
    scale_x: float = 1.0,
    scale_y: Optional[float] = None,
    pad_pixels: int = 0,
    clip_to: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, int, int]:
    if scale_y is None:
        scale_y = scale_x
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale_x
    h = (y2 - y1) * scale_y
    nx1 = int(round(cx - w / 2.0)) - pad_pixels
    ny1 = int(round(cy - h / 2.0)) - pad_pixels
    nx2 = int(round(cx + w / 2.0)) + pad_pixels
    ny2 = int(round(cy + h / 2.0)) + pad_pixels
    if clip_to is not None:
        W, H = clip_to
        nx1 = max(0, min(W - 1, nx1))
        ny1 = max(0, min(H - 1, ny1))
        nx2 = max(0, min(W - 1, nx2))
        ny2 = max(0, min(H - 1, ny2))
    return (nx1, ny1, nx2, ny2)

# ---------------- NMS helpers ----------------
def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = areaA + areaB - inter
    return 0.0 if union <= 0 else inter / union

def nms_numpy(boxes: List[Tuple[int,int,int,int]], scores: List[float], iou_thresh: float = 0.5) -> List[int]:
    if len(boxes) == 0:
        return []
    boxes_np = np.array(boxes, dtype=float)
    scores_np = np.array(scores, dtype=float)
    order = scores_np.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([iou(boxes_np[i], boxes_np[j]) for j in rest])
        inds = np.where(ious < iou_thresh)[0]
        order = rest[inds]
    return keep

# ---------------- Keypoint helpers ----------------
def valid_kp(kp_row, idx, conf_thr):
    try:
        x = float(kp_row[idx][0])
        y = float(kp_row[idx][1])
        c = float(kp_row[idx][2])
    except Exception:
        return None
    if c is None or np.isnan(c) or c <= conf_thr:
        return None
    return (x, y, c)

def wrist_box_from_kp(
    kp,
    img_w: int,
    img_h: int,
    hand: str = "L",
    conf_thr: float = DEFAULTS["CONF_THR_POSE"],
    WRIST_HALF: int = DEFAULTS["WRIST_HALF"],
    WRIST_BOX_FRACTION_OF_TORSO: float = DEFAULTS["WRIST_BOX_FRACTION_OF_TORSO"],
    WRIST_BOX_MIN_HALF: int = DEFAULTS["WRIST_BOX_MIN_HALF"],
) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    wrist_idx = LEFT_WRI if hand == "L" else RIGHT_WRI
    v = valid_kp(kp, wrist_idx, conf_thr)
    if v is None:
        return None, 0.0
    wx, wy, wc = v

    sho_l = valid_kp(kp, LEFT_SHO, conf_thr)
    sho_r = valid_kp(kp, RIGHT_SHO, conf_thr)
    if sho_l and sho_r:
        torso_width = abs(sho_l[0] - sho_r[0])
        half = max(WRIST_BOX_MIN_HALF, int(torso_width * WRIST_BOX_FRACTION_OF_TORSO))
    else:
        half = WRIST_HALF

    x1 = int(max(0, round(wx - half)))
    y1 = int(max(0, round(wy - half)))
    x2 = int(min(img_w - 1, round(wx + half)))
    y2 = int(min(img_h - 1, round(wy + half)))
    return (x1, y1, x2, y2), float(wc)

def extract_gun_boxes_from_res(res, conf_thr: float = DEFAULTS["CONF_THR_GUN"]):
    out: List[Dict[str, Any]] = []
    try:
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [int(round(v)) for v in boxes[i].tolist()]
            s = float(scores[i])
            cls = int(classes[i])
            if s >= conf_thr:
                out.append({"bbox": [x1, y1, x2, y2], "score": s, "class": cls})
    except Exception:
        try:
            for b in getattr(res, "boxes", []):
                xy = b.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(round(v)) for v in xy]
                s = float(b.conf[0].cpu().numpy().tolist())
                cls = int(b.cls[0].cpu().numpy().tolist())
                if s >= conf_thr:
                    out.append({"bbox": [x1, y1, x2, y2], "score": s, "class": cls})
        except Exception:
            return []
    return out

def person_box_from_kp(kp, img_w:int, img_h:int, pad:int=20) -> Optional[Tuple[int,int,int,int]]:
    try:
        xs = [float(p[0]) for p in kp if p[0] is not None and not np.isnan(p[0])]
        ys = [float(p[1]) for p in kp if p[1] is not None and not np.isnan(p[1])]
        if len(xs) == 0 or len(ys) == 0:
            return None
        x1 = int(max(0, min(xs) - pad))
        y1 = int(max(0, min(ys) - pad))
        x2 = int(min(img_w - 1, max(xs) + pad))
        y2 = int(min(img_h - 1, max(ys) + pad))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1,y1,x2,y2)
    except Exception:
        return None

# ---------------- Model loader ----------------
def model_fn(model_dir_or_config: Optional[Dict[str, Any]] = None):
    """
    Load and return a model dict: {"gun_model", "pose_model", "config"}.
    Accepts a dict of overrides or None to use DEFAULTS.
    """
    cfg = DEFAULTS.copy()
    if model_dir_or_config:
        if isinstance(model_dir_or_config, dict):
            cfg.update(model_dir_or_config)
        else:
            model_dir = str(model_dir_or_config)
            cfg["GUN_MODEL_PATH"] = os.path.join(model_dir, os.path.basename(cfg["GUN_MODEL_PATH"]))
            cfg["POSE_MODEL_PATH"] = os.path.join(model_dir, os.path.basename(cfg["POSE_MODEL_PATH"]))

    if cfg.get("VERBOSE"):
        print(f"[info] Loading gun model from: {cfg.get('GUN_MODEL_PATH')}")
        print(f"[info] Loading pose model from: {cfg.get('POSE_MODEL_PATH')}")

    gun_model = YOLO(str(cfg.get("GUN_MODEL_PATH")))
    pose_model = YOLO(str(cfg.get("POSE_MODEL_PATH")))

    return {"gun_model": gun_model, "pose_model": pose_model, "config": cfg}

# ---------------- Single-frame handlers ----------------
def input_frame_fn(request_body: Any, content_type: str = "application/json") -> Dict[str, Any]:
    """
    Accepts JSON payload with base64 JPEG (encoding) and optional metadata keys.
    Returns dict: { cam_id, org_id, user_id, frame (BGR numpy), raw_payload }
    """
    if content_type != "application/json":
        raise ValueError("frame handler expects application/json with base64 JPEG in 'encoding' field")

    if isinstance(request_body, str):
        payload = json.loads(request_body)
    else:
        payload = request_body

    cam_id = payload.get("cam_id", -1)
    org_id = payload.get("org_id", None)
    user_id = payload.get("user_id", None)
    b64 = payload.get("encoding") or payload.get("image")
    if not b64:
        raise ValueError("Payload must include 'encoding' with base64 jpeg data")

    if isinstance(b64, str) and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]

    try:
        decoded = base64.b64decode(b64)
        arr = np.frombuffer(decoded, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}")

    return {"cam_id": cam_id, "org_id": org_id, "user_id": user_id, "frame": img, "raw_payload": payload}

def predict_frame_fn(input_data: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run single-frame inference. Returns dict with:
      - cam_id
      - guns: list of accepted guns with holder info
      - annotated_frame: base64 jpeg
      - status: 0 success, 1 error
      - debug: optional internals
    """
    gun_model = model.get("gun_model")
    pose_model = model.get("pose_model")
    cfg = model.get("config", DEFAULTS).copy()

    frame = input_data["frame"]
    cam_id = input_data.get("cam_id", -1)

    try:
        h, w = frame.shape[:2]

        # Pose inference
        pres = pose_model.predict(source=frame, conf=cfg.get("CONF_THR_POSE"), imgsz=cfg.get("IMG_SIZE"), verbose=False)[0]
        kpts_all = None
        if getattr(pres, "keypoints", None) is not None and len(pres.keypoints) > 0:
            try:
                kpts_all = pres.keypoints.data.cpu().numpy()
            except Exception:
                kpts_all = pres.keypoints.cpu().numpy()

        # Build wrist boxes
        wrist_boxes: List[Dict[str, Any]] = []
        if kpts_all is not None:
            for pi, kp in enumerate(kpts_all):
                lbox, lscore = wrist_box_from_kp(kp, w, h, hand="L", conf_thr=cfg.get("CONF_THR_POSE"))
                rbox, rscore = wrist_box_from_kp(kp, w, h, hand="R", conf_thr=cfg.get("CONF_THR_POSE"))
                if lbox is not None:
                    wrist_boxes.append({"person_idx": int(pi), "hand": "L", "box": tuple(lbox), "score": float(lscore), "kp": kp})
                if rbox is not None:
                    wrist_boxes.append({"person_idx": int(pi), "hand": "R", "box": tuple(rbox), "score": float(rscore), "kp": kp})

        # Gun inference
        gres = gun_model.predict(source=frame, conf=cfg.get("CONF_THR_GUN"), imgsz=cfg.get("IMG_SIZE"), verbose=False)[0]
        gun_boxes = extract_gun_boxes_from_res(gres, conf_thr=cfg.get("CONF_THR_GUN"))

        # Pre-filter & NMS
        img_area = max(1, w * h)
        min_img_area = cfg.get("GUN_MIN_FRAC") * img_area
        max_img_area = cfg.get("GUN_MAX_FRAC") * img_area

        candidate_guns = []
        for g in gun_boxes:
            raw_bbox = tuple(g["bbox"])
            raw_area = area_of_box(raw_bbox)
            if cfg.get("USE_ABSOLUTE_AREA"):
                if raw_area < cfg.get("GUN_MIN_AREA") or raw_area > cfg.get("GUN_MAX_AREA"):
                    continue
            if cfg.get("USE_IMAGE_RELATIVE"):
                if raw_area < min_img_area or raw_area > max_img_area:
                    continue
            candidate_guns.append(g)

        gb_list = [tuple(g["bbox"]) for g in candidate_guns]
        gs_list = [g["score"] for g in candidate_guns]
        keep_idx = nms_numpy(gb_list, gs_list, iou_thresh=cfg.get("NMS_IOU_GUN")) if len(gb_list) > 0 else []
        candidate_guns = [candidate_guns[i] for i in keep_idx]

        # Matching + holder (single-frame fallbacks)
        accepted: List[Dict[str, Any]] = []
        for g in candidate_guns:
            raw_bbox = tuple(g["bbox"])
            gbox_for_match = scale_box(raw_bbox, scale_x=cfg.get("GUN_SCALE"), scale_y=cfg.get("GUN_SCALE"),
                                       pad_pixels=cfg.get("GUN_PAD_PIXELS"), clip_to=(w, h))
            g_area = area_of_box(gbox_for_match)
            if g_area == 0:
                continue

            matched_info = None
            keep_flag = False

            # match to wrist boxes first
            for wb in wrist_boxes:
                wbox = wb["box"]
                ia = intersect_area(gbox_for_match, wbox)
                frac = ia / float(g_area) if g_area > 0 else 0.0
                if ia > 0 and frac >= cfg.get("MIN_INTERSECTION_FRAC"):
                    if cfg.get("USE_RELATIVE_TO_WRIST"):
                        wrist_area = area_of_box(wbox)
                        if wrist_area == 0:
                            continue
                        raw_g_area = area_of_box(raw_bbox)
                        ratio = float(raw_g_area) / float(wrist_area)
                        if ratio < cfg.get("GUN_TO_WRIST_MIN_RATIO") or ratio > cfg.get("GUN_TO_WRIST_MAX_RATIO"):
                            continue
                        matched_info = {"person_idx": wb["person_idx"], "hand": wb["hand"], "wrist_box": list(wbox), "wrist_score": wb["score"], "inter_area": int(ia), "inter_frac": float(frac), "size_ratio": float(ratio)}
                    else:
                        matched_info = {"person_idx": wb["person_idx"], "hand": wb["hand"], "wrist_box": list(wbox), "wrist_score": wb["score"], "inter_area": int(ia), "inter_frac": float(frac)}
                    keep_flag = True
                    break

            holder_person_box = None
            holder_person_idx = None

            if keep_flag and matched_info is not None:
                # derive person box from pose keypoints
                if kpts_all is not None:
                    pidx = matched_info.get("person_idx")
                    try:
                        kp_for_person = kpts_all[int(pidx)]
                        pb = person_box_from_kp(kp_for_person, w, h, pad=20)
                        if pb is not None:
                            holder_person_box = list(pb)
                            holder_person_idx = int(pidx)
                    except Exception:
                        holder_person_box = None
            else:
                # fallback: nearest person keypoints bbox
                if kpts_all is not None and len(kpts_all) > 0:
                    gx1, gy1, gx2, gy2 = gbox_for_match
                    gcx = (gx1 + gx2) / 2.0
                    gcy = (gy1 + gy2) / 2.0
                    best_dist = float("inf")
                    best_pb = None
                    best_pi = None
                    for pi, kp in enumerate(kpts_all):
                        pb = person_box_from_kp(kp, w, h, pad=20)
                        if pb is None:
                            continue
                        px1, py1, px2, py2 = pb
                        pcx = (px1 + px2) / 2.0
                        pcy = (py1 + py2) / 2.0
                        dist = (pcx - gcx) * 2 + (pcy - gcy) * 2
                        if dist < best_dist:
                            best_dist = dist
                            best_pb = pb
                            best_pi = pi
                    if best_pb is not None:
                        holder_person_box = list(best_pb)
                        holder_person_idx = best_pi
                        keep_flag = True
                        matched_info = {"person_idx": None, "hand": None, "wrist_box": None, "wrist_score": 0.0, "inter_area": 0, "inter_frac": 0.0}

            if keep_flag:
                rec = {
                    "bbox": list(raw_bbox),
                    "bbox_used_for_matching": list(gbox_for_match),
                    "score": g["score"],
                    "class": g["class"],
                    "matched_wrist": matched_info,
                    "holder_person_idx": holder_person_idx,
                    "holder_person_box": holder_person_box,
                }
                accepted.append(rec)

        # Draw annotations on copy of frame
        out_frame = frame.copy()
        if cfg.get("DRAW_WRISTS"):
            for wb in wrist_boxes:
                x1, y1, x2, y2 = wb["box"]
                color = (0, 255, 0) if wb["hand"] == "L" else (255, 0, 0)
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out_frame, f"{wb['hand']}:{wb['score']:.2f}", (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        drawn_holders = set()
        for ag in accepted:
            x1, y1, x2, y2 = ag["bbox"]
            s = ag["score"]
            cv2.rectangle(out_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)
            label = f"GUN:{s:.2f}"
            if ag.get("matched_wrist"):
                mw = ag["matched_wrist"]
                if mw.get("hand") is not None:
                    label += f" {mw['hand']}@P{mw.get('person_idx')}"
            cv2.putText(out_frame, label, (int(x1), max(10, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            holder_box = ag.get("holder_person_box")
            holder_idx = ag.get("holder_person_idx")
            if holder_box is not None:
                key = holder_idx if holder_idx is not None else tuple(holder_box)
                if key not in drawn_holders:
                    hx1, hy1, hx2, hy2 = holder_box
                    purple = (128, 0, 128)
                    cv2.rectangle(out_frame, (int(hx1), int(hy1)), (int(hx2), int(hy2)), purple, 3)
                    if holder_idx is not None:
                        cv2.putText(out_frame, f"HOLDER P{holder_idx}", (int(hx1), max(20, int(hy1) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, purple, 2)
                    else:
                        cv2.putText(out_frame, "HOLDER", (int(hx1), max(20, int(hy1) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, purple, 2)
                    drawn_holders.add(key)

        # encode annotated frame to base64 JPEG
        ok, jpg = cv2.imencode('.jpg', out_frame)
        if not ok:
            raise RuntimeError("Failed to encode annotated frame to JPEG")
        b64_out = base64.b64encode(jpg.tobytes()).decode('utf-8')

        # simplified guns output
        guns_out = []
        for ag in accepted:
            guns_out.append({
                "bbox": ag["bbox"],
                "score": ag["score"],
                "matched_wrist": ag.get("matched_wrist"),
                "holder": {
                    "person_idx": ag.get("holder_person_idx"),
                    "bbox": ag.get("holder_person_box")
                }
            })

        return {
            "cam_id": cam_id,
            "guns": guns_out,
            "annotated_frame": b64_out,
            "status": 0,
            "debug": {"num_wrist_boxes": len(wrist_boxes), "num_gun_candidates": len(candidate_guns), "num_accepted": len(accepted)}
        }

    except Exception as e:
        return {"cam_id": cam_id, "guns": [], "annotated_frame": "", "status": 1, "error": str(e)}

def output_frame_fn(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final formatting for client response (strips debug by default).
    """
    return {
        "cam_id": prediction.get("cam_id", -1),
        "guns": prediction.get("guns", []),
        "annotated_frame": prediction.get("annotated_frame", ""),
        "status": prediction.get("status", 0)
    }

# ---------------- Minimal _main_ ----------------
if _name_ == "_main_":
    # No auto-run — loading the module prints a message.
    print("inference_single_frame.py loaded. Call model_fn(), input_frame_fn(), predict_frame_fn(), output_frame_fn() programmatically.")