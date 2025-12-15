"""
inference.py - Production Inference Module for Gun + Wrist Detection v1.0
=========================================================================
Single-frame inference pipeline for API deployment with a clean I/O format.

Expected input JSON:
{
    "cam_id": 123,
    "org_id": 2,
    "user_id": 2,
    "encoding": "<base64_jpeg_data>"
}

Output JSON:
{
    "cam_id": int,
    "org_id": int,
    "user_id": int,
    "frame_id": str,
    "timestamp": str,
    "guns": List[Dict],
    "message": str,
    "annotated_frame": str,   # base64 JPEG
    "status": int
}
"""

import base64
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics YOLO not installed. Run: pip install ultralytics") from e


# =============================================================================
# CONFIG CONSTANTS (SIMILAR TO YOUR ORIGINAL SCRIPT)
# =============================================================================

GUN_MODEL_PATH = r"gun_dd_f_best.pt"
POSE_MODEL_PATH = r"yolov8x-pose.pt"

IMG_SIZE = 1280
CONF_THR_GUN = 0.05
CONF_THR_POSE = 0.05

DRAW_WRISTS = True
MIN_INTERSECTION_FRAC = 0.0

WRIST_HALF = 25
WRIST_BOX_FRACTION_OF_TORSO = 0.18
WRIST_BOX_MIN_HALF = 15

GUN_SCALE = 1.10
GUN_PAD_PIXELS = 0

GUN_MIN_FRAC = 0.0005
GUN_MAX_FRAC = 0.02

USE_ABSOLUTE_AREA = False     # if True, can add GUN_MIN_AREA/GUN_MAX_AREA
USE_IMAGE_RELATIVE = True
USE_RELATIVE_TO_WRIST = False
GUN_TO_WRIST_MIN_RATIO = 0.25
GUN_TO_WRIST_MAX_RATIO = 2.0

# COCO keypoints used (0-based)
LEFT_SHO, RIGHT_SHO = 5, 6
LEFT_WRI, RIGHT_WRI = 9, 10


# =============================================================================
# MODEL INITIALIZATION (model_fn)
# =============================================================================

class GunWristModel:
    """Gun + wrist detection model wrapper (YOLO pose + YOLO gun detector)."""

    def __init__(
        self,
        pose_model_path: Optional[str] = None,
        gun_model_path: Optional[str] = None,
        imgsz: int = IMG_SIZE,
        conf_thr_pose: float = CONF_THR_POSE,
        conf_thr_gun: float = CONF_THR_GUN,
    ):
        self.pose_model_path = pose_model_path or POSE_MODEL_PATH
        self.gun_model_path = gun_model_path or GUN_MODEL_PATH
        self.imgsz = imgsz
        self.conf_thr_pose = conf_thr_pose
        self.conf_thr_gun = conf_thr_gun

        # Load YOLO models
        self.pose_model = YOLO(str(self.pose_model_path))
        self.gun_model = YOLO(str(self.gun_model_path))

        # Performance stats
        self.performance_stats = {
            "total_frames_processed": 0,
            "avg_inference_time": 0.0,
            "total_guns_detected": 0,
        }

    def update_perf(self, inference_time: float, num_guns: int):
        self.performance_stats["total_frames_processed"] += 1
        prev_avg = self.performance_stats["avg_inference_time"]
        self.performance_stats["avg_inference_time"] = 0.9 * prev_avg + 0.1 * inference_time
        self.performance_stats["total_guns_detected"] += num_guns


def model_fn(model_dir: Optional[str] = None) -> GunWristModel:
    """
    Load and initialize the model (SageMaker-style).
    model_dir can be used to resolve relative model paths if you package models with the code.
    """
    try:
        # You could add extra dependency checks here if you want (like in theft module)
        pose_path = f"{model_dir}/yolov8x-pose.pt" if model_dir else POSE_MODEL_PATH
        gun_path = f"{model_dir}/gun_dd_f_best.pt" if model_dir else GUN_MODEL_PATH

        model = GunWristModel(
            pose_model_path=pose_path,
            gun_model_path=gun_path,
            imgsz=IMG_SIZE,
            conf_thr_pose=CONF_THR_POSE,
            conf_thr_gun=CONF_THR_GUN,
        )
        return model

    except Exception as e:
        # In your theft module you use logger; here we'll just raise.
        raise RuntimeError(f"Failed to load GunWristModel: {e}") from e


# =============================================================================
# INPUT PROCESSING (input_fn)
# =============================================================================

def input_fn(request_body: str, content_type: str = "application/json") -> Dict[str, Any]:
    """
    Parse and validate input request.
    Expected format (same style as theft inference):
      {
        "cam_id": 123,
        "org_id": 2,
        "user_id": 2,
        "encoding": "base64_jpeg_data"
      }
    """
    try:
        if content_type != "application/json":
            raise ValueError(f"Unsupported content type: {content_type}")

        data = json.loads(request_body)

        # Validate required fields
        required_fields = ["cam_id", "org_id", "user_id", "encoding"]
        for f in required_fields:
            if f not in data:
                raise ValueError(f"Missing required field: {f}")

        if not isinstance(data["cam_id"], int):
            raise ValueError("cam_id must be an integer")
        if not isinstance(data["org_id"], int):
            raise ValueError("org_id must be an integer")
        if not isinstance(data["user_id"], int):
            raise ValueError("user_id must be an integer")

        # Decode base64 image -> OpenCV frame
        encoded = data["encoding"]
        if isinstance(encoded, str) and "," in encoded:
            encoded = encoded.split(",", 1)[1]
        frame_bytes = base64.b64decode(encoded)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode frame image")

        data["frame"] = frame
        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}") from e
    except Exception as e:
        raise ValueError(f"Input validation failed: {e}") from e


# =============================================================================
# PREDICTION (predict_fn)
# =============================================================================

def predict_fn(input_data: Dict[str, Any], model: GunWristModel) -> Dict[str, Any]:
    """
    Run gun + wrist inference on a single RGB frame.
    Mirrors the structure of the theft predict_fn (build results dict, timing, etc.).
    """
    t0 = time.time()

    try:
        cam_id = input_data["cam_id"]
        org_id = input_data["org_id"]
        user_id = input_data["user_id"]
        frame = input_data["frame"]
        h, w = frame.shape[:2]

        # Build frame_id similar style: {CAMID}{USERID}{ORGID}{DDMMYYYY}{HHMMSS}
        now = datetime.now()
        frame_id = f"{cam_id}{user_id}{org_id}{now.strftime('%d%m%Y%H%M%S')}"

        results: Dict[str, Any] = {
            "cam_id": cam_id,
            "org_id": org_id,
            "user_id": user_id,
            "frame_id": frame_id,
            "timestamp": now.isoformat() + "Z",
            "guns": [],
            "annotated_frame": None,
            "status": 1,
        }

        # ---------------- STEP 1: Pose inference -> wrist boxes ----------------
        pres = model.pose_model.predict(
            source=frame,
            conf=model.conf_thr_pose,
            imgsz=model.imgsz,
            verbose=False,
        )[0]

        kpts_all = None
        if getattr(pres, "keypoints", None) is not None and len(pres.keypoints) > 0:
            try:
                kpts_all = pres.keypoints.data.cpu().numpy()
            except Exception:
                kpts_all = pres.keypoints.cpu().numpy()

        wrist_boxes: List[Dict[str, Any]] = []
        if kpts_all is not None:
            for pi, kp in enumerate(kpts_all):
                lbox, lscore = wrist_box_from_kp(kp, w, h, hand="L", conf_thr=model.conf_thr_pose)
                rbox, rscore = wrist_box_from_kp(kp, w, h, hand="R", conf_thr=model.conf_thr_pose)
                if lbox is not None:
                    wrist_boxes.append(
                        {"person_idx": int(pi), "hand": "L", "box": tuple(lbox), "score": float(lscore)}
                    )
                if rbox is not None:
                    wrist_boxes.append(
                        {"person_idx": int(pi), "hand": "R", "box": tuple(rbox), "score": float(rscore)}
                    )

        # ---------------- STEP 2: Gun detector ----------------
        gres = model.gun_model.predict(
            source=frame,
            conf=model.conf_thr_gun,
            imgsz=model.imgsz,
            verbose=False,
        )[0]
        gun_boxes = extract_gun_boxes_from_res(gres, conf_thr=model.conf_thr_gun)

        # ---------------- STEP 3: Area filtering ----------------
        img_area = max(1, w * h)
        min_img_area = GUN_MIN_FRAC * img_area
        max_img_area = GUN_MAX_FRAC * img_area

        candidate_guns: List[Dict[str, Any]] = []
        for g in gun_boxes:
            raw_bbox = tuple(g["bbox"])
            raw_area = area_of_box(raw_bbox)

            if USE_IMAGE_RELATIVE:
                if raw_area < min_img_area or raw_area > max_img_area:
                    continue

            candidate_guns.append(g)

        # ---------------- STEP 4: Intersection with wrist boxes ----------------
        accepted: List[Dict[str, Any]] = []
        for g in candidate_guns:
            raw_bbox = tuple(g["bbox"])
            gbox_for_match = scale_box(
                raw_bbox,
                scale_x=GUN_SCALE,
                scale_y=GUN_SCALE,
                pad_pixels=GUN_PAD_PIXELS,
                clip_to=(w, h),
            )
            g_area = area_of_box(gbox_for_match)
            if g_area == 0:
                continue

            matched_info = None
            keep = False
            for wb in wrist_boxes:
                wbox = wb["box"]
                ia = intersect_area(gbox_for_match, wbox)
                frac = ia / float(g_area) if g_area > 0 else 0.0
                if ia > 0 and frac >= MIN_INTERSECTION_FRAC:
                    if USE_RELATIVE_TO_WRIST:
                        wrist_area = area_of_box(wbox)
                        if wrist_area == 0:
                            continue
                        raw_g_area = area_of_box(raw_bbox)
                        ratio = float(raw_g_area) / float(wrist_area)
                        if ratio < GUN_TO_WRIST_MIN_RATIO or ratio > GUN_TO_WRIST_MAX_RATIO:
                            continue
                        matched_info = {
                            "person_idx": wb["person_idx"],
                            "hand": wb["hand"],
                            "wrist_box": list(wbox),
                            "wrist_score": wb["score"],
                            "inter_area": int(ia),
                            "inter_frac": float(frac),
                            "size_ratio": float(ratio),
                        }
                    else:
                        matched_info = {
                            "person_idx": wb["person_idx"],
                            "hand": wb["hand"],
                            "wrist_box": list(wbox),
                            "wrist_score": wb["score"],
                            "inter_area": int(ia),
                            "inter_frac": float(frac),
                        }
                    keep = True
                    break

            if keep:
                accepted.append(
                    {
                        "bbox": list(raw_bbox),
                        "bbox_used_for_matching": list(gbox_for_match),
                        "score": g["score"],
                        "class": g["class"],
                        "matched_wrist": matched_info,
                    }
                )

        results["guns"] = accepted

        # ---------------- STEP 5: Visualization ----------------
        annotated = _visualize_frame(frame, wrist_boxes, accepted)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        results["annotated_frame"] = base64.b64encode(buf).decode("utf-8")

        # ---------------- STEP 6: Perf stats ----------------
        dt = time.time() - t0
        model.update_perf(dt, len(accepted))

        return results

    except Exception as e:
        # Mirror theft style: return error structure with status=0
        return {
            "cam_id": input_data.get("cam_id", -1),
            "org_id": input_data.get("org_id", -1),
            "user_id": input_data.get("user_id", -1),
            "frame_id": "",
            "timestamp": datetime.now().isoformat() + "Z",
            "guns": [],
            "annotated_frame": None,
            "status": 0,
            "error": str(e),
        }


# =============================================================================
# OUTPUT FORMATTING (output_fn)
# =============================================================================

def output_fn(prediction: Dict[str, Any], accept: str = "application/json") -> str:
    """
    Format prediction output for API response.
    Same pattern as theft module: pick only certain keys and serialize to JSON.
    """
    try:
        if accept != "application/json":
            raise ValueError(f"Unsupported accept type: {accept}")

        out = {
            "cam_id": prediction.get("cam_id", -1),
            "org_id": prediction.get("org_id", -1),
            "user_id": prediction.get("user_id", -1),
            "frame_id": prediction.get("frame_id", ""),
            "timestamp": prediction.get("timestamp", datetime.now().isoformat() + "Z"),
            "guns": prediction.get("guns", []),
            "message": prediction.get("message", ""),
            "annotated_frame": prediction.get("annotated_frame", ""),
            "status": prediction.get("status", 0),
        }

        if "error" in prediction:
            out["error"] = prediction["error"]

        return json.dumps(out, indent=2)

    except Exception as e:
        error_response = {
            "status": 0,
            "error": str(e),
        }
        return json.dumps(error_response)


# =============================================================================
# HELPER FUNCTIONS (GEOMETRY + VISUALIZATION)
# =============================================================================

def area_of_box(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return w * h


def intersect_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    return iw * ih


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


def valid_kp(kp_row, idx, conf_thr):
    try:
        x, y, c = float(kp_row[idx][0]), float(kp_row[idx][1]), float(kp_row[idx][2])
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
    conf_thr: float = CONF_THR_POSE,
    WRIST_HALF: int = WRIST_HALF,
    WRIST_BOX_FRACTION_OF_TORSO: float = WRIST_BOX_FRACTION_OF_TORSO,
    WRIST_BOX_MIN_HALF: int = WRIST_BOX_MIN_HALF,
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


def extract_gun_boxes_from_res(res, conf_thr: float = CONF_THR_GUN):
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


def _visualize_frame(
    frame: np.ndarray,
    wrist_boxes: List[Dict[str, Any]],
    guns: List[Dict[str, Any]],
) -> np.ndarray:
    """Draw wrist boxes and accepted gun boxes on the frame."""
    out = frame.copy()

    # Draw wrists
    if DRAW_WRISTS:
        for wb in wrist_boxes:
            x1, y1, x2, y2 = wb["box"]
            color = (0, 255, 0) if wb["hand"] == "L" else (255, 0, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                f"{wb['hand']}:{wb['score']:.2f}",
                (x1, max(10, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

    # Draw guns
    for g in guns:
        x1, y1, x2, y2 = g["bbox"]
        s = g["score"]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)
        label = f"GUN:{s:.2f}"
        if g.get("matched_wrist"):
            mw = g["matched_wrist"]
            label += f" {mw['hand']}@P{mw['person_idx']}"
        cv2.putText(
            out,
            label,
            (int(x1), max(10, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return out


# =============================================================================
# MAIN INFERENCE ENTRY POINT (run_inference)
# =============================================================================

def run_inference(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for inference.

    Follows the same pattern as theft run_inference:
      - Uses global _inference_model
      - Goes through input_fn -> predict_fn -> output_fn
      - Adds a human-readable message
    """
    global _inference_model

    try:
        # Lazy init model
        if "_inference_model" not in globals():
            _inference_model = model_fn()

        # Normalize input via input_fn
        processed_input = input_fn(json.dumps(input_data), "application/json")

        # Run prediction
        prediction = predict_fn(processed_input, _inference_model)

        # Build message field
        guns = prediction.get("guns", [])
        if guns:
            count = len(guns)
            # Simple message: number of guns + info about first one
            first = guns[0]
            msg = f"🔴 {count} gun detection(s) near wrist region"
            if first.get("matched_wrist"):
                mw = first["matched_wrist"]
                msg += f" (hand {mw['hand']} for person {mw['person_idx']})"
            prediction["message"] = msg
        else:
            prediction["message"] = "✅ No guns detected near wrist regions"

        # Format with output_fn and return dict (like theft code)
        out_str = output_fn(prediction, "application/json")
        return json.loads(out_str)

    except Exception as e:
        return {
            "cam_id": input_data.get("cam_id", -1),
            "org_id": input_data.get("org_id", -1),
            "user_id": input_data.get("user_id", -1),
            "frame_id": "",
            "timestamp": datetime.now().isoformat() + "Z",
            "guns": [],
            "message": f"❌ Error: {str(e)}",
            "annotated_frame": "",
            "status": 0,
            "error": str(e),
        }


# =============================================================================
# EXAMPLE USAGE AND TESTING (__main__)
# =============================================================================



def live_inference(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[error] Cannot open video")
        return

    print("[info] Running live gun+wrist inference... Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Encode to base64
        _, buffer = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(buffer).decode("utf-8")

        # Create payload
        input_payload = {
            "cam_id": 123,
            "org_id": 2,
            "user_id": 2,
            "encoding": b64,
        }

        # Run inference
        result = run_inference(input_payload)

        # Decode annotated frame
        if result.get("annotated_frame"):
            ann_bytes = base64.b64decode(result["annotated_frame"])
            arr = np.frombuffer(ann_bytes, dtype=np.uint8)
            ann_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            ann_frame = frame  # fallback

        # Show on screen
        cv2.imshow("Live Inference", ann_frame)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    """
    Example usage of the Gun + Wrist inference system
    """

    live_inference(r"E:\All_models\gun_detection\vidw3.mp4")

    # def test_with_image(image_path: str):
    #     """Test with an actual image file."""
    #     try:
    #         frame = cv2.imread(image_path)
    #         if frame is None:
    #             raise ValueError(f"Could not read image: {image_path}")

    #         # Encode to base64
    #         _, buffer = cv2.imencode(".jpg", frame)
    #         b64 = base64.b64encode(buffer).decode("utf-8")

    #         input_payload = {
    #             "cam_id": 123,
    #             "org_id": 2,
    #             "user_id": 2,
    #             "encoding": b64,
    #         }

    #         print("[info] Running gun+wrist inference on real image...")
    #         result = run_inference(input_payload)

    #         print("\n" + "=" * 80)
    #         print("INFERENCE RESULT")
    #         print("=" * 80)
    #         print(json.dumps(result, indent=2))
    #         print("=" * 80 + "\n")

    #         # Save annotated frame
    #         if result.get("annotated_frame"):
    #             ann_bytes = base64.b64decode(result["annotated_frame"])
    #             arr = np.frombuffer(ann_bytes, dtype=np.uint8)
    #             ann_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    #             out_path = "gun_wrist_annotated.jpg"
    #             cv2.imwrite(out_path, ann_frame)
    #             print(f"[info] Saved annotated frame to: {out_path}")

    #         return result

    #     except Exception as e:
    #         print(f"[error] Test failed: {e}")
    #         return None

    
