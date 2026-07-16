"""
view_stream.py — Live OpenCV viewer for the gun detection server.

Connects to the WebSocket inference server, sends any stream URL
(HLS, RTSP, HTTP video, local file), and displays the annotated
output in a real-time OpenCV window.

Usage:
    # HLS stream
    .venv\\Scripts\\python view_stream.py --url https://example.com/stream.m3u8

    # RTSP camera
    .venv\\Scripts\\python view_stream.py --url rtsp://192.168.1.100:554/stream

    # Local video file
    .venv\\Scripts\\python view_stream.py --url C:/path/to/video.mp4

    # Custom server host / port
    .venv\\Scripts\\python view_stream.py --url <stream_url> --host localhost --port 8004

Controls:
    q / Esc  — quit
    s        — save current frame as screenshot (screenshot_<N>.jpg)
    p        — pause / resume
"""

import asyncio
import json
import time
import argparse
import base64
import os
import sys
from datetime import datetime

try:
    import websockets
except ImportError:
    print("[ERROR] websockets not installed.  Run: pip install websockets")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("[ERROR] opencv-python not installed.  Run: pip install opencv-python")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_HOST      = "localhost"
DEFAULT_PORT      = 8004
DEFAULT_CLIENT_ID = "viewer-001"
DEFAULT_CAMERA_ID = 1
DEFAULT_ORG_ID    = 1
DEFAULT_USER_ID   = 1
WINDOW_NAME       = "Gun Detection — Live View  |  q=quit  s=save  p=pause"


# ─────────────────────────────────────────────────────────────────────────────
# Shared pause flag (set from keyboard handler in main thread)
# ─────────────────────────────────────────────────────────────────────────────
_paused       = False
_screenshot_n = 0
_quit         = False


def _handle_key(key: int, frame: "np.ndarray | None"):
    """Process a single keypress from cv2.waitKey()."""
    global _paused, _screenshot_n, _quit
    if key == ord("q") or key == 27:   # q or Esc
        _quit = True
    elif key == ord("p"):
        _paused = not _paused
        status = "PAUSED" if _paused else "RESUMED"
        print(f"\n  [{status}]  press 'p' again to toggle")
    elif key == ord("s") and frame is not None:
        fname = f"screenshot_{_screenshot_n}.jpg"
        cv2.imwrite(fname, frame)
        print(f"\n  [SAVED] {fname}")
        _screenshot_n += 1


# ─────────────────────────────────────────────────────────────────────────────
# Viewer
# ─────────────────────────────────────────────────────────────────────────────
async def view(host: str, port: int, stream_url: str,
               client_id: str, camera_id: int,
               org_id: int, user_id: int) -> None:
    global _quit, _paused

    uri = f"ws://{host}:{port}/ws/gundetection/{client_id}"

    print("=" * 70)
    print("  GUN DETECTION — LIVE VIEWER")
    print("=" * 70)
    print(f"  Server  : {uri}")
    print(f"  Stream  : {stream_url}")
    print(f"  Controls: q/Esc=quit   s=screenshot   p=pause")
    print()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    last_frame: "np.ndarray | None" = None
    frame_count = 0
    t_start     = time.perf_counter()

    try:
        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=30,
            max_size=10 * 1024 * 1024,  # 10 MB — room for annotated frames
        ) as ws:
            print(f"  [OK] Connected to {uri}\n")

            # ── Start the stream ──────────────────────────────────────────
            start_msg = {
                "action":      "start_stream",
                "stream_name": stream_url,
                "camera_id":   camera_id,
                "user_id":     user_id,
                "org_id":      org_id,
                "region":      "ap-south-1",
            }
            await ws.send(json.dumps(start_msg))

            try:
                async for raw in ws:
                    # ── Keyboard events ───────────────────────────────────
                    key = cv2.waitKey(1) & 0xFF
                    _handle_key(key, last_frame)
                    if _quit:
                        print("\n  [INFO] Quit by user.")
                        break

                    # ── Parse message ─────────────────────────────────────
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if msg.get("status") == "error":
                        print(f"\n  [ERROR] Server: {msg.get('message', msg)}")
                        break

                    det = msg.get("detections", {})
                    if not det:
                        continue

                    if det.get("status", 0) != 0:
                        print(f"\n  [ERROR] Inference: {det.get('error', 'unknown')}")
                        continue

                    # ── Decode annotated frame ────────────────────────────
                    b64 = det.get("annotated_frame", "")
                    if b64:
                        try:
                            arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if img is not None:
                                last_frame = img
                        except Exception:
                            pass

                    # ── Show frame (skip when paused) ─────────────────────
                    if last_frame is not None and not _paused:
                        cv2.imshow(WINDOW_NAME, last_frame)

                    # ── Stats overlay in console ──────────────────────────
                    frame_count += 1
                    live_fps     = frame_count / (time.perf_counter() - t_start)
                    guns         = det.get("guns", [])
                    holders      = det.get("gun_holders", [])
                    persons      = det.get("stats", {}).get("persons_tracked", 0)
                    alerts       = det.get("alerts", [])

                    if alerts:
                        print()
                        for a in alerts:
                            lvl = a.get("level", "HIGH")
                            tag = "[CRITICAL]" if lvl == "CRITICAL" else "[HIGH]   "
                            print(f"  {tag} ALERT  ID:{a.get('track_id')}  "
                                  f"conf:{a.get('confidence', 0):.2f}  "
                                  f"weapon:{a.get('weapon_type', '?')}")

                    print(f"\r  [F{frame_count:5d}] "
                          f"{live_fps:5.1f}fps  "
                          f"guns={len(guns)}  "
                          f"armed={len(holders)}  "
                          f"persons={persons}  "
                          f"{'[PAUSED]' if _paused else '        '}",
                          end="", flush=True)

                    # Check window close button
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        print("\n  [INFO] Window closed.")
                        break

            except websockets.exceptions.ConnectionClosedOK:
                print("\n  [INFO] Stream ended — server closed connection.")
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"\n  [ERROR] Connection dropped: {e}")

            # ── Stop stream ───────────────────────────────────────────────
            try:
                await ws.send(json.dumps({"action": "stop_stream"}))
            except Exception:
                pass

    except ConnectionRefusedError:
        print(f"\n  [ERROR] Could not connect to {uri}")
        print(f"          Start the server with:")
        print(f"          .venv\\Scripts\\uvicorn app:app --host 0.0.0.0 --port {port}")
    except Exception as e:
        print(f"\n  [ERROR] {type(e).__name__}: {e}")
    finally:
        cv2.destroyAllWindows()
        wall = time.perf_counter() - t_start
        avg  = frame_count / wall if wall > 0 else 0
        print(f"\n\n  Frames shown : {frame_count}")
        print(f"  Wall time    : {wall:.1f}s")
        print(f"  Average FPS  : {avg:.1f}")
        print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Live OpenCV viewer for the gun detection WebSocket server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url",       required=True,
                        help="Stream URL: HLS (https://…m3u8), RTSP, local video file, etc.")
    parser.add_argument("--host",      default=DEFAULT_HOST,      help="Server host (default: localhost)")
    parser.add_argument("--port",      default=DEFAULT_PORT,      type=int, help="Server port (default: 8004)")
    parser.add_argument("--client-id", default=DEFAULT_CLIENT_ID, help="WebSocket client ID")
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID, type=int)
    parser.add_argument("--org-id",    default=DEFAULT_ORG_ID,    type=int)
    parser.add_argument("--user-id",   default=DEFAULT_USER_ID,   type=int)
    args = parser.parse_args()

    asyncio.run(view(
        host       = args.host,
        port       = args.port,
        stream_url = args.url,
        client_id  = args.client_id,
        camera_id  = args.camera_id,
        org_id     = args.org_id,
        user_id    = args.user_id,
    ))


if __name__ == "__main__":
    main()
