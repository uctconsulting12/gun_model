import cv2
import numpy as np
import base64
import time
import boto3
import logging
from PIL import Image

S3_BUCKET = "gun-detections"
s3 = boto3.client("s3")

logger = logging.getLogger("s3_utils_gundetections")


def upload_to_s3(frame, frame_num):
    """Upload annotated frame to S3 and return its URL."""

    # ---------------- Convert to NumPy array if needed ----------------
    if frame is None:
        raise ValueError(f"Frame {frame_num} is None, cannot upload")

    # PIL Image -> NumPy array
    if isinstance(frame, Image.Image):
        frame = np.array(frame)

    # Base64 string -> NumPy array
    elif isinstance(frame, str):
        try:
            img_bytes = base64.b64decode(frame)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"Frame {frame_num} is not a valid base64 image -> {e}")

    # Check final type
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Frame {frame_num} is not a valid NumPy array, got {type(frame)}")

    # ---------------- Encode and upload ----------------
    try:
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError(f"Frame {frame_num}: cv2.imencode failed")

        key = f"shoplifting-results/frame_{frame_num}_{int(time.time())}.jpg"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=buffer.tobytes(),
            ContentType="image/jpeg"
        )

        url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
        logger.info(f"Frame {frame_num}: uploaded to S3 at {url}")
        return url

    except Exception as e:
        logger.error(f"Frame {frame_num}: failed to upload to S3 -> {e}")
        raise
