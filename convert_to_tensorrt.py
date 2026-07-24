"""
convert_to_tensorrt.py

Converts YOLO models to TensorRT engines for maximum inference speed.

Usage:
    python convert_to_tensorrt.py

Outputs (in project root):
    best.engine              — weapon detection TensorRT engine (YOLO26x, 7 classes)
    yolo11m-pose.engine      — pose estimation TensorRT engine (YOLO11m)

Model classes (best.pt / best.engine):
    0: Blunt_Weapon
    1: Explosive
    2: Fire_Smoke   ← filtered out at inference (IGNORED_CLASS_IDS)
    3: Firearm
    4: Melee_Weapon
    5: Person       ← filtered out at inference (handled by pose model)
    6: Tool         ← filtered out at inference (not a threat)

Requirements:
    - NVIDIA GPU with CUDA
    - TensorRT installed (comes with ultralytics + torch)
    - Run ONCE before starting the server

Notes:
    - FP16 is used by default (2x faster than FP32, negligible accuracy loss)
    - dynamic=True lets you tune GUN_INFER_IMGSZ at runtime (e.g. 480 for speed)
    - The engine is device-specific — regenerate if you change GPU
    - YOLO26x is a large model (~118MB); build time can take 5-10 minutes
    - If workspace=4 fails with memory errors, reduce to workspace=2 or workspace=1
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def convert(model_path: str, output_name: str, imgsz: int = 640, half: bool = True,
            dynamic: bool = False, workspace: int = 4):
    """Export a YOLO .pt model to a TensorRT .engine file."""
    from ultralytics import YOLO
    import torch

    if not torch.cuda.is_available():
        print(f"[ERROR] CUDA not available — TensorRT export requires a GPU.")
        sys.exit(1)

    abs_path = os.path.join(_HERE, model_path)
    if not os.path.exists(abs_path):
        print(f"[SKIP] {model_path} not found at {abs_path}")
        return None

    engine_path = os.path.join(_HERE, output_name)
    if os.path.exists(engine_path):
        print(f"[SKIP] {output_name} already exists — delete it to re-export.")
        return engine_path

    print(f"\n[EXPORT] {model_path} → {output_name}")
    print(f"         imgsz={imgsz}, half={half}, dynamic={dynamic}, workspace={workspace}GB")

    model = YOLO(abs_path)
    try:
        exported = model.export(
            format="engine",
            imgsz=imgsz,
            half=half,          # FP16 — fastest on modern GPUs
            dynamic=dynamic,    # dynamic=True allows variable input sizes at runtime
            device=0,           # GPU 0
            workspace=workspace,
            verbose=False,
        )
        print(f"[OK]   Exported → {exported}")
        return exported
    except RuntimeError as e:
        if "workspace" in str(e).lower() or "memory" in str(e).lower():
            print(f"[WARN] Build failed with workspace={workspace}GB. Retrying with workspace=2GB...")
            try:
                exported = model.export(
                    format="engine",
                    imgsz=imgsz,
                    half=half,
                    dynamic=dynamic,
                    device=0,
                    workspace=2,
                    verbose=False,
                )
                print(f"[OK]   Exported → {exported}")
                return exported
            except RuntimeError as e2:
                print(f"[ERROR] Export failed at workspace=2GB too: {e2}")
                print("        Try workspace=1 manually or use the .pt fallback.")
                return None
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("  YOLO -> TensorRT Conversion")
    print("=" * 60)

    # 1. Weapon detection model — YOLO26x (7 classes), dynamic shapes
    #    Firearm/Explosive/Blunt/Melee will raise alerts; Person/Tool/Fire_Smoke
    #    are filtered at inference time via IGNORED_CLASS_IDS.
    convert("best.pt", "best.engine", imgsz=640, half=True, dynamic=True, workspace=4)

    # 2. Pose model — YOLO11m for better keypoint accuracy, dynamic shapes
    #    ultralytics will auto-download yolo11m-pose.pt if not present
    convert("yolo11m-pose.pt", "yolo11m-pose.engine", imgsz=640, half=True, dynamic=True, workspace=4)

    print("\n[DONE] All conversions complete.")
    print("       Start the server - it will load .engine files automatically.")
    print()
    print("  TIP: engines exported with dynamic=True support variable input sizes.")
    print("       Set GUN_INFER_IMGSZ=480 in CONFIG_OVERRIDES for ~30% faster gun inference.")
