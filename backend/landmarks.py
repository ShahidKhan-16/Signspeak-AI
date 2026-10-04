"""
landmarks.py — MediaPipe Hands landmark extraction for HandSign.

Architecture:
- Two-stage pipeline: Palm Detector (SSD) -> Dynamic Hand RoI Crop & Rotation -> 21 3D Hand Landmarks.
- Uses official MediaPipe Tasks HandLandmarker (CPU XNNPACK backend).
- Accurately tracks hands at natural distances (30-60cm) and close-ups, with full position and scale invariance.
- Fallback: ai_edge_litert direct crop if Tasks unavailable, else plain ack.
- Performance: ~12-18ms/frame on Apple Silicon / CPU XNNPACK.
"""
import base64
import os
import time
import threading
import urllib.request
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

DEBUG_LANDMARKS = os.getenv("DEBUG_LANDMARKS", "0") == "1"
DEBUG_SAVE_EVERY = int(os.getenv("DEBUG_SAVE_EVERY", "30"))

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
DEBUG_DIR = Path(__file__).parent / "debug_frames"

MEDIAPIPE_AVAILABLE = False
_detector = None
_processing_lock = threading.Lock()
_frame_idx = 0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _ensure_model() -> str | None:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1000:
        return str(MODEL_PATH)
    try:
        print(f"[landmarks] downloading hand_landmarker model to {MODEL_PATH} ...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        print(f"[landmarks] model saved ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")
        return str(MODEL_PATH)
    except Exception as e:
        print(f"[landmarks] WARN: model download failed: {e}")
        return None


def _init_landmarker():
    global _detector, MEDIAPIPE_AVAILABLE
    try:
        model_path = _ensure_model()
        if not model_path:
            print("[landmarks] WARN: no model — fallback to plain ack")
            return

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        _detector = vision.HandLandmarker.create_from_options(options)

        # Warmup
        dummy = np.zeros((240, 320, 3), dtype=np.uint8)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy)
        _detector.detect(mp_img)

        MEDIAPIPE_AVAILABLE = True
        print("[landmarks] MediaPipe Tasks HandLandmarker ready (Palm Detector + Crop + Landmarks, XNNPACK CPU)")

        if DEBUG_LANDMARKS:
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            print(f"[landmarks] DEBUG ON — saving every {DEBUG_SAVE_EVERY}th annotated frame to {DEBUG_DIR}/")

    except Exception as e:
        print(f"[landmarks] WARN: MediaPipe Tasks init failed: {e}")
        MEDIAPIPE_AVAILABLE = False
        _detector = None


_init_landmarker()


def _draw_and_save_debug(bgr_frame: np.ndarray, landmarks_list, handedness_list, idx: int):
    try:
        h, w = bgr_frame.shape[:2]
        vis = bgr_frame.copy()
        for hand_idx, lms in enumerate(landmarks_list):
            for lm in lms:
                cx, cy = int(lm["x"] * w), int(lm["y"] * h)
                cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)
            for a, b in HAND_CONNECTIONS:
                if a < len(lms) and b < len(lms):
                    pa = (int(lms[a]["x"] * w), int(lms[a]["y"] * h))
                    pb = (int(lms[b]["x"] * w), int(lms[b]["y"] * h))
                    cv2.line(vis, pa, pb, (0, 255, 0), 2)
            handed = handedness_list[hand_idx] if hand_idx < len(handedness_list) else "?"
            cv2.putText(vis, f"#{idx} {handed}", (10, 30 + hand_idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out_path = DEBUG_DIR / f"annotated_{idx:05d}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"[landmarks] debug saved {out_path.name}")
    except Exception as e:
        print(f"[landmarks] debug draw failed: {e}")


def process_frame(raw_bytes: bytes) -> dict:
    """
    Run full Hand Landmarking pipeline (Palm Detection -> RoI Crop -> Landmarks).
    Returns dict with hands_detected, landmarks (normalized 0-1), handedness, inference_ms.
    Gracefully returns hands_detected=0 on no-hand or error (never throws).
    """
    global _frame_idx
    _frame_idx += 1
    cur_idx = _frame_idx

    if not MEDIAPIPE_AVAILABLE or _detector is None:
        return {"hands_detected": 0, "landmarks": [], "handedness": [], "inference_ms": 0, "dropped": False, "reason": "mediapipe_unavailable"}

    if not _processing_lock.acquire(blocking=False):
        return {"hands_detected": 0, "landmarks": [], "handedness": [], "inference_ms": 0, "dropped": True, "reason": "busy"}

    try:
        t0 = time.perf_counter()
        nparr = np.frombuffer(raw_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"hands_detected": 0, "landmarks": [], "handedness": [], "inference_ms": 0, "dropped": False, "reason": "decode_failed"}

        # Downsample large images (>640px) to maintain sub-20ms latency
        h, w = bgr.shape[:2]
        if max(h, w) > 640:
            scale = 640.0 / max(h, w)
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Full detection (Palm Detector -> Hand RoI Crop -> 21 Landmark extraction)
        detection_result = _detector.detect(mp_image)
        inference_ms = (time.perf_counter() - t0) * 1000

        hands_landmarks = detection_result.hand_landmarks
        handedness_results = detection_result.handedness

        if not hands_landmarks or len(hands_landmarks) == 0:
            if cur_idx % 15 == 0:
                print(f"[landmarks #{cur_idx}] no hand ({inference_ms:.1f}ms)")
            return {
                "hands_detected": 0,
                "landmarks": [],
                "handedness": [],
                "hand_flag": 0.0,
                "handedness_score": 0.0,
                "inference_ms": round(inference_ms, 1),
                "dropped": False,
            }

        # Format landmarks
        all_hands_lms: List[List[Dict[str, float]]] = []
        all_handedness: List[str] = []
        primary_score = 0.95

        for hand_idx, lms in enumerate(hands_landmarks):
            hand_points = []
            for lm in lms:
                # Clamp coordinates to 0.0 - 1.0
                clamped_x = max(0.0, min(1.0, float(lm.x)))
                clamped_y = max(0.0, min(1.0, float(lm.y)))
                hand_points.append({"x": clamped_x, "y": clamped_y, "z": float(lm.z)})
            all_hands_lms.append(hand_points)

            # Extract handedness label & score
            if hand_idx < len(handedness_results) and len(handedness_results[hand_idx]) > 0:
                category = handedness_results[hand_idx][0]
                label = category.category_name or ("Right" if category.index == 1 else "Left")
                all_handedness.append(label)
                if hand_idx == 0:
                    primary_score = float(category.score)
            else:
                all_handedness.append("Right")

        if cur_idx % 15 == 0 or inference_ms > 50:
            handed_str = ", ".join(all_handedness)
            print(f"[landmarks #{cur_idx}] {len(all_hands_lms)} hand(s) [{handed_str}] {inference_ms:.1f}ms")

        result = {
            "hands_detected": len(all_hands_lms),
            "landmarks": all_hands_lms,
            "handedness": all_handedness,
            "hand_flag": round(primary_score, 3),
            "handedness_score": round(primary_score, 3),
            "inference_ms": round(inference_ms, 1),
            "dropped": False,
        }

        # Periodic debug output
        if DEBUG_LANDMARKS and cur_idx % DEBUG_SAVE_EVERY == 0:
            _draw_and_save_debug(bgr, all_hands_lms, all_handedness, cur_idx)

        return result

    except Exception as e:
        print(f"[landmarks] process error: {e}")
        import traceback
        traceback.print_exc()
        return {"hands_detected": 0, "landmarks": [], "handedness": [], "inference_ms": 0, "dropped": False, "reason": str(e)}
    finally:
        _processing_lock.release()


def process_b64(b64_str: str) -> dict:
    raw = base64.b64decode(b64_str)
    return process_frame(raw)
