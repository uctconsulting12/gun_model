# Gun Detection Service

Real-time multi-class weapon detection service built for security and surveillance. Streams video from any camera source, runs parallel TensorRT inference to detect weapons and identify the person holding them, raises graded alerts, and persists annotated evidence frames to AWS S3 with metadata in PostgreSQL. Exposed over a FastAPI WebSocket so clients can start/stop streams and receive live detection payloads.

---

## Use Case

- Public spaces — schools, malls, banks, transit stations
- Private premises and facility monitoring
- Live CCTV / IP camera streams via AWS Kinesis Video Streams (KVS), RTSP, or HLS
- Local video files for testing and development

Key capabilities:

- Detects 7 weapon classes and identifies the **person holding** the weapon via pose estimation and wrist-proximity association.
- Maintains **persistent per-camera tracking** — once a person is marked as armed their bounding box stays purple for the entire session and never reverts to green.
- **Threat-only ReID** — OSNet appearance embeddings run exclusively for confirmed armed persons, not for every person in the scene.
- **ReID cooldown** — after each gun detection event the re-identification is locked for 2 seconds per-detection to prevent ID swaps while the tracker stabilises.
- Sends graded alerts (`HIGH`, `CRITICAL`) based on detection confidence.
- Stores **only frames containing weapons** to S3 to minimise storage cost.
- Supports multiple concurrent cameras, each with isolated model state.
- Live **OpenCV viewer** available for monitoring any stream locally.

---

## Architecture

```
                ┌──────────────────────────┐
   Client ───►  │  /ws/gundetection/       │  WebSocket (FastAPI)
                │      {client_id}         │
                └────────────┬─────────────┘
                             │  {"action":"start_stream", ...}
                             ▼
                ┌──────────────────────────┐
                │  gun_handler.py          │  Validates stream_name:
                │                          │  • local file / rtsp:// / http(s)://
                │                          │    → passed directly to VideoCapture
                │                          │  • bare KVS stream name
                │                          │    → resolved via get_kvs_hls_url()
                └────────────┬─────────────┘
                             │  loop.run_in_executor(ThreadPoolExecutor)
                             ▼
            ┌────────────────────────────────────┐
            │  gun_detection_websocket.py        │
            │  cv2.VideoCapture(url)             │
            │  for each frame:                   │
            │    run_inference_raw(frame, cam_id)│──────────────────────┐
            └──────────────────┬─────────────────┘                      │
                               │                                         ▼
                               │                    ┌────────────────────────────────┐
                               │                    │  gun_detection.py              │
                               │                    │  CameraTracker (per-camera)    │
                               │                    │    model_fn() on first call    │
                               │                    │    predict_frame_fn()          │
                               │                    │    output_frame_fn()           │
                               │                    └────────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                      ▼
   ws.send_text(detections JSON)       Storage thread (daemon)
   every frame                         only when guns detected:
                                         upload_to_s3()
                                         insert_data() → Postgres
```

### Inference pipeline (per frame)

```
Raw frame (BGR numpy)
        │
        ▼
Letterbox resize to 640×640
(eliminates internal TRT resize overhead)
        │
        ├───────────────────────────────────────────┐
        ▼                                           ▼
yolo11m-pose.engine (TRT, FP16)          best.engine (TRT, FP16)
Pose + person detection                  Weapon detection (YOLO26x, 7-class)
stream=True, parallel thread             stream=True, parallel thread
        │                                           │
        ▼                                           ▼
Unscale boxes to original coords         Unscale boxes to original coords
                                         Filter IGNORED_CLASS_IDS
                                         (Person / Tool / Fire_Smoke stripped)
                                         NMS + confidence filter
        │                                           │
        └───────────────┬───────────────────────────┘
                        ▼
           DeepSort person tracker
           Threat-only OSNet ReID:
             • armed persons past their lock → embed (gallery refresh)
             • armed persons within lock window → reuse gallery (no OSNet)
             • unarmed persons → unit-vector placeholder (IoU-only)
                        │
                        ▼
           Wrist-proximity association
           (gun bbox ↔ person wrist keypoint → 3 fallback strategies)
                        │
                        ▼
           GunTracker (IoU-based, stable G-IDs across frames)
                        │
                        ▼
           AlertManager (per-person, configurable cooldown)
                        │
                        ▼
           Annotated frame + detection JSON
```

### Bounding box colour scheme

| Colour | Meaning |
|---|---|
| Green | Tracked person — never held a weapon. No ID label shown. |
| Purple | Confirmed weapon holder. Permanent — never reverts to green. Shows `ID:{n} [weapon] (conf)`. |
| Red box | Detected firearm / explosive |
| Orange box | Detected melee / blunt weapon |

### ReID pipeline — threat-only with per-detection cooldown

OSNet appearance embeddings are computed **only for confirmed armed persons** (persons in `armed_ids`). Unarmed persons use IoU-only tracking.

When a gun is detected on a person:
1. The person is added to `armed_ids` permanently.
2. A **ReID lock** is set for that specific detection event: `unlock_frame = current_frame + REID_LOCK_FRAMES` (default 60 frames = 2 s @ 30 fps).
3. During the lock window, the existing gallery embedding is reused — no OSNet call, no risk of ID swap.
4. The lock resets on every new detection event (not just the first), so if the person lowers and raises the weapon again, the cooldown slides forward each time.
5. After the lock expires, OSNet re-embeds the person every frame to keep the gallery fresh for occlusion recovery.

### Performance (benchmarked on local GPU)

| Metric | Value |
|---|---|
| Steady-state median latency | ~46 ms |
| Throughput median | **21.7 fps** |
| Throughput avg | 19.9 fps |
| p95 latency | ~71 ms |
| Target | 15 fps |

---

## Project Structure

```
gun_model/
├── app.py                               # FastAPI entry point — WebSocket endpoint
├── Dockerfile                           # CUDA 12.1 + PyTorch + TensorRT base image
├── requirements.txt                     # Python dependencies
├── convert_to_tensorrt.py               # One-time .pt → .engine conversion script
├── test_websocket.py                    # End-to-end WebSocket test client (--show flag)
├── view_stream.py                       # Standalone OpenCV live viewer for any stream
├── best.pt                              # YOLO26x fine-tuned weapon detector weights
├── best.engine                          # TensorRT FP16 engine (generated by convert script)
├── yolo11m-pose.pt                      # YOLO11m pose model weights (auto-downloaded)
├── yolo11m-pose.engine                  # TensorRT FP16 engine (generated by convert script)
└── src/
    ├── websocket/
    │   └── gun_detection_websocket.py   # Frame loop, run_inference_raw, storage queue
    ├── handlers/
    │   └── gun_handler.py               # WebSocket lifecycle, stream URL resolution
    ├── local_models/
    │   ├── gun_detection.py             # CameraTracker, run_inference / run_inference_raw
    │   └── inference_gun_detection_reid.py  # Model load, parallel predict, annotate
    ├── store_s3/
    │   └── gun_store.py                 # S3 upload helpers
    ├── database/
    │   └── gun_query.py                 # Postgres insert for detection metadata
    └── utils/
        └── kvs_stream.py                # Resolve AWS KVS stream name → HLS URL
```

---

## Models

| File | Description | Size |
|---|---|---|
| `best.pt` | YOLO26x fine-tuned weapon detector (7 classes) | ~118 MB |
| `best.engine` | TensorRT FP16 engine built from `best.pt` | ~121 MB |
| `yolo11m-pose.pt` | YOLO11m pose estimation (auto-downloaded if missing) | ~52 MB |
| `yolo11m-pose.engine` | TensorRT FP16 engine built from pose model | ~50 MB |

### Weapon detection classes (`best.pt` / `best.engine`)

| Class ID | Name | Trigger | Box colour |
|---|---|---|---|
| 0 | Blunt_Weapon | Melee alert | Orange |
| 1 | Explosive | Armed alert | Red |
| 2 | Fire_Smoke | **Ignored** — environmental | — |
| 3 | Firearm | Armed alert | Red |
| 4 | Melee_Weapon | Melee alert | Orange |
| 5 | Person | **Ignored** — handled by pose model | — |
| 6 | Tool | **Ignored** — not a threat | — |

Classes 2, 5, and 6 are silently filtered at inference via `IGNORED_CLASS_IDS`.

`FIREARM_CLASS_IDS = {1, 3}` — these classes permanently mark the holder in `armed_ids` and trigger the purple box.  
`MELEE_CLASS_IDS = {0, 4}` — these classes get an orange weapon box and a purple holder box this frame, but do **not** permanently mark the holder.

Both `.engine` files are **device-specific** — regenerate them if you move to a different GPU (see [TensorRT Conversion](#tensorrt-conversion)).

The `non_weapons.pt` cross-verification model has been removed. Size filtering, confidence thresholds, and wrist-proximity association provide sufficient false-positive suppression without the latency cost.

---

## Installation

### Option A — Docker (recommended for production)

The image is built on `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`. TensorRT is installed on top.

```bash
# Build (requires NVIDIA GPU at build time for TRT conversion)
docker build -t gun-detection:latest .

# Run
docker run --gpus all -p 8004:8004 \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e AWS_DEFAULT_REGION=ap-south-1 \
  -e DB_HOST=... -e DB_NAME=... -e DB_USER=... -e DB_PASSWORD=... \
  gun-detection:latest
```

### Option B — Local Python (development)

Requires Python 3.10+, CUDA 12.1 drivers, and NVIDIA TensorRT.

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Install PyTorch with CUDA (not in requirements.txt — provided by Docker base image)
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install TensorRT
pip install tensorrt

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Convert models to TensorRT (one-time, requires GPU)
python convert_to_tensorrt.py
```

### Environment variables

| Variable | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | S3 upload + KVS stream resolution |
| `AWS_DEFAULT_REGION` | Default AWS region |
| `S3_BUCKET` | Bucket where annotated frames are stored |
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | Postgres connection |

---

## TensorRT Conversion

Run once before starting the server (or after changing GPU):

```bash
python convert_to_tensorrt.py
```

This exports:
- `best.pt` → `best.engine` (FP16, dynamic shapes, workspace ≤ 2 GB)
- `yolo11m-pose.pt` → `yolo11m-pose.engine` (FP16, dynamic shapes)

Dynamic shapes allow `GUN_INFER_IMGSZ` to be tuned at runtime without re-exporting. The server automatically uses `.engine` files when present and falls back to `.pt` if they are missing.

> **6 GB VRAM note:** the build workspace is capped at 2 GB to avoid OOM. Models with attention-based architectures (e.g. YOLO12m) may fail TRT conversion on 6 GB GPUs — use the `.pt` fallback for those or switch to a smaller variant.

---

## Running the Service

```bash
# Start the server
.venv\Scripts\uvicorn app:app --host 0.0.0.0 --port 8004

# With auto-reload during development
.venv\Scripts\uvicorn app:app --host 0.0.0.0 --port 8004 --reload
```

The server listens on `ws://localhost:8004/ws/gundetection/{client_id}`.

---

## WebSocket API

### Connect

```
ws://localhost:8004/ws/gundetection/<client_id>
```

### Start a stream

```json
{
    "action": "start_stream",
    "stream_name": "https://example.com/live.m3u8",
    "camera_id": 10,
    "user_id": 10,
    "org_id": 10,
    "region": "ap-south-1"
}
```

`stream_name` routing:

| Value | Behaviour |
|---|---|
| Local file path (`video.mp4`, `C:\videos\test.mp4`) | Passed directly to `cv2.VideoCapture` |
| `rtsp://...` or `rtsps://...` | Passed directly to `cv2.VideoCapture` |
| `http://...` or `https://...` (HLS, HTTP video) | Passed directly to `cv2.VideoCapture` |
| Bare stream name (e.g. `Cam424`) | Resolved via AWS KVS → HLS URL |

### Stop a stream

```json
{ "action": "stop_stream" }
```

### Detection payload (received per frame)

```json
{
  "detections": {
    "cam_id": 10,
    "org_id": 10,
    "user_id": 10,
    "frame_number": 142,
    "timestamp": "2026-07-16T12:34:56.789+00:00",
    "guns": [
      {
        "gun_id": 1,
        "bbox": [x1, y1, x2, y2],
        "score": 0.87,
        "holder_id": 5,
        "class_id": 3,
        "weapon_type": "Firearm"
      }
    ],
    "gun_holders": [
      {
        "track_id": 5,
        "confidence": 0.87,
        "weapon_type": "Firearm",
        "class_id": 3
      }
    ],
    "persons_present": [5, 7],
    "alerts": [
      {
        "track_id": 5,
        "confidence": 0.91,
        "level": "CRITICAL",
        "weapon_type": "Firearm",
        "class_id": 3,
        "timestamp": "2026-07-16T12:34:56.789+00:00"
      }
    ],
    "annotated_frame": "<base64-encoded JPEG>",
    "stats": {
      "raw_preds": 2,
      "verified_guns": 1,
      "guns_drawn": 1,
      "holders_drawn": 1,
      "persons_tracked": 3
    },
    "status": 0
  }
}
```

`status: 0` = success, `status: 1` = inference error (includes `"error"` key).

---

## Testing

### Standalone inference test (no server required)

Runs the full inference pipeline against a local video file and prints a benchmark report:

```bash
python -m src.local_models.gun_detection
```

Edit `VIDEO_PATH` at the bottom of `gun_detection.py` to point at your test file. Output includes per-frame latency (min / avg / median / max / p95) and a pass/fail against the 15 fps target. An OpenCV window shows the annotated output in real time.

### End-to-end WebSocket test

```bash
# Terminal 1 — start server
.venv\Scripts\uvicorn app:app --host 0.0.0.0 --port 8004

# Terminal 2 — run test (prints stats, no window)
python test_websocket.py

# With live OpenCV display window
python test_websocket.py --show

# Common options
python test_websocket.py --max-frames 100                        # quick smoke test
python test_websocket.py --url rtsp://192.168.1.10:554/stream    # RTSP camera
python test_websocket.py --url https://example.com/live.m3u8     # HLS stream
python test_websocket.py --url https://example.com/live.m3u8 --show  # HLS + display
python test_websocket.py --host 10.0.0.5 --port 8004 --camera-id 3
```

### Live stream viewer

`view_stream.py` is a dedicated viewer that connects to any stream URL through the inference server and shows the annotated output in an OpenCV window. Use this whenever you want to watch inference output from an HLS or RTSP source.

```bash
# HLS stream
python view_stream.py --url https://example.com/live.m3u8

# RTSP camera
python view_stream.py --url rtsp://192.168.1.100:554/stream

# Local video file
python view_stream.py --url C:/path/to/video.mp4

# Custom server
python view_stream.py --url <stream_url> --host localhost --port 8004
```

Window controls:

| Key | Action |
|---|---|
| `q` or `Esc` | Quit |
| `p` | Pause / resume |
| `s` | Save current frame as `screenshot_N.jpg` |
| Close button | Quit |

---

## Configuration

All tuning knobs live in two places:

- **`DEFAULTS`** in `src/local_models/inference_gun_detection_reid.py` — base values
- **`CONFIG_OVERRIDES`** in `src/local_models/gun_detection.py` — runtime overrides (takes precedence)

Edit `CONFIG_OVERRIDES` for day-to-day tuning. Only touch `DEFAULTS` when adding new keys.

| Key | Default | Override | Description |
|---|---|---|---|
| `FINAL_CONFIDENCE_THRESHOLD` | `0.50` | `0.40` | Min score for a gun detection to be reported |
| `CONF_THR_POSE` | `0.40` | `0.20` | Min person detection confidence |
| `CONF_THR_WRIST` | `0.20` | `0.15` | Min wrist keypoint confidence for holder association |
| `ALERT_ON_FIRST_DETECTION_ONLY` | `True` | `True` | Alert once per person; set `False` for repeat alerts |
| `ALERT_COOLDOWN_FRAMES` | `90` | `90` | Frames between repeat alerts (~3 s @ 30 fps) |
| `ALERT_THRESHOLD` | `0.45` | — | Min confidence to trigger an alert |
| `GUN_SKIP_FRAMES` | `1` | `1` | Run gun model every N frames (1 = every frame) |
| `GUN_INFER_IMGSZ` | `480` | `480` | Gun model input size (smaller = faster, requires dynamic engine) |
| `INFER_IMGSZ` | `640` | — | Letterbox target before both models — must match engine build size |
| `USE_OSNET` | `True` | `True` | Enable OSNet ReID for armed-person tracking |
| `REID_LOCK_FRAMES` | `60` | — | Frames to freeze ReID after each detection event (~2 s @ 30 fps) |
| `INFERENCE_WORKERS` | `3` | `3` | Thread pool size (pose + gun + OSNet run in parallel) |
| `TRACKER_MAX_AGE` | `90` | — | Frames to hold a lost track before dropping it (~3 s @ 30 fps) |
| `TRACKER_MAX_COSINE_DISTANCE` | `0.20` | — | Tighter = fewer appearance-based ID swaps |

### Tuning the ReID cooldown

```python
# gun_detection.py — CONFIG_OVERRIDES
"REID_LOCK_FRAMES": 90,   # 3 s cooldown — use if ID swaps still occur
"REID_LOCK_FRAMES": 30,   # 1 s cooldown — faster re-ID recovery after occlusion
```

---

## Storage Behaviour

- Annotated frames are queued for S3 upload **only when guns are detected** (`result["guns"]` is non-empty).
- Storage runs in a **daemon thread** per camera session — S3/DB I/O never blocks inference.
- The storage queue is bounded at 500 items; frames are dropped with a warning if the queue fills (slow S3 connection).
- Each camera session gets its own storage thread, started when the stream begins and shut down cleanly when it ends.

---

## Notes

- Each `cam_id` gets its own `model_fn()` instance, frame counter, `armed_ids` set, and `armed_id_locked_until` dict — safe for multi-camera concurrent streaming.
- The first frame of each session is slow (~2–3 s) due to TRT execution context initialisation. Subsequent frames run at full speed.
- `run_inference_raw(frame, cam_id)` accepts a raw BGR numpy array and skips the base64 encode/decode round-trip — use it for any caller that already has a decoded frame (the WebSocket loop uses this path).
- `run_inference(payload)` accepts a base64-encoded JSON payload — use this for HTTP or legacy callers.
- Once a person is added to `armed_ids` their bounding box is permanently purple. This state persists for the lifetime of the camera session and is cleared only on an explicit `reset_camera(cam_id)` call or server restart.
- S3 / Postgres credential errors during local testing are expected and do not affect inference.
