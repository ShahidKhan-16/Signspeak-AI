"""
classifier.py — ISL Alphabet Classifier with 90-Dimensional Geometric Invariants.

Feature Representation (90 Dimensions):
1. Scale-Normalized 3D Coordinates (63 dims):
   - Wrist (0) translated to (0,0,0)
   - Scaled by robust max Euclidean distance from wrist (max_dist).
2. 15 Knuckle / Joint 3D Bending Angles (15 dims):
   - 3 angles per finger for all 5 fingers.
   - 100% position-, scale-, and 2D rotation-invariant.
3. Fingertip-to-Wrist Extension/Curl Distances (5 dims):
   - Normalized tip extension/curl indicators for Thumb, Index, Middle, Ring, Pinky.
4. Thumb-to-Fingertip Pinch Distances (4 dims):
   - Distance from Thumb Tip (4) to Index (8), Middle (12), Ring (16), Pinky (20).
5. Adjacent Fingertip Spread Distances (3 dims):
   - Distance between Index-Middle (8-12), Middle-Ring (12-16), Ring-Pinky (16-20).
"""
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "isl_alphabet_classifier.pkl"
_model = None
_labels = None
_feature_type = None
_load_error_logged = False

# Joint triplets for 15 knuckle angles: 3 angles per finger
JOINT_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),        # Thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8),        # Index
    (0, 9, 10), (9, 10, 11), (10, 11, 12),  # Middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16),# Ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20) # Pinky
]
TIPS = [4, 8, 12, 16, 20]


def landmarks_to_feature(landmarks_21, is_left: bool = False) -> np.ndarray:
    """
    Extract 90-dim geometric feature vector from 21 landmarks.
    landmarks_21: list of 21 dicts {x,y,z}, or (21,3) ndarray, or flat 63 array.
    """
    if isinstance(landmarks_21, np.ndarray):
        if landmarks_21.size == 63:
            arr = landmarks_21.reshape(21, 3).astype(np.float32).copy()
        elif landmarks_21.shape == (21, 3):
            arr = landmarks_21.astype(np.float32).copy()
        else:
            raise ValueError(f"Unexpected ndarray shape {landmarks_21.shape}")
    elif isinstance(landmarks_21, list):
        if len(landmarks_21) == 21 and isinstance(landmarks_21[0], dict):
            arr = np.array([[p["x"], p["y"], p.get("z", 0.0)] for p in landmarks_21], dtype=np.float32)
        elif len(landmarks_21) == 63:
            arr = np.array(landmarks_21, dtype=np.float32).reshape(21, 3)
        else:
            arr = np.array(landmarks_21, dtype=np.float32).reshape(21, 3)
    else:
        raise ValueError(f"Unsupported landmarks input type {type(landmarks_21)}")

    # 1. Wrist relative translation + Robust max-distance scale normalization
    wrist = arr[0].copy()
    rel = arr - wrist
    max_d = float(np.max(np.linalg.norm(rel, axis=1)))
    if max_d < 1e-6:
        max_d = 1.0

    norm_coords = (rel / max_d).flatten()  # 63 dims

    # 2. 15 Joint Bending Angles (naturally scale-invariant)
    angles = []
    for a, b, c in JOINT_TRIPLETS:
        v1 = rel[a] - rel[b]
        v2 = rel[c] - rel[b]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            angles.append(0.0)
        else:
            cos_a = np.dot(v1, v2) / (n1 * n2)
            angles.append(float(np.arccos(np.clip(cos_a, -1.0, 1.0)) / np.pi))
    angles = np.array(angles, dtype=np.float32)  # 15 dims

    # 3. Fingertip-to-Wrist Extension/Curl Distances
    tip_wrist_dists = np.linalg.norm(rel[TIPS], axis=1) / max_d  # 5 dims

    # 4. Thumb Tip to other fingertips Pinch Distances
    thumb_tip = rel[4]
    thumb_dists = np.linalg.norm(rel[[8, 12, 16, 20]] - thumb_tip, axis=1) / max_d  # 4 dims

    # 5. Adjacent Fingertip Spread Distances
    spread_dists = np.array([
        np.linalg.norm(rel[8] - rel[12]),
        np.linalg.norm(rel[12] - rel[16]),
        np.linalg.norm(rel[16] - rel[20])
    ], dtype=np.float32) / max_d  # 3 dims

    feat = np.concatenate([norm_coords, angles, tip_wrist_dists, thumb_dists, spread_dists]).astype(np.float32)
    return feat  # Exactly 90 dims


def _load_model() -> bool:
    global _model, _labels, _feature_type, _load_error_logged
    if _model is not None:
        return True
    try:
        if not MODEL_PATH.exists():
            if not _load_error_logged:
                print(f"[classifier] No model at {MODEL_PATH}, run python backend/train_alphabet.py")
                _load_error_logged = True
            return False
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "model" in payload:
            _model = payload["model"]
            _labels = payload.get("labels", getattr(_model, "classes_", None))
            _feature_type = payload.get("feature_type", "unknown")
        else:
            _model = payload
            _labels = getattr(_model, "classes_", None)
            _feature_type = "legacy"
        print(f"[classifier] Loaded {MODEL_PATH} labels={list(_labels) if _labels is not None else '?'} feature={_feature_type}")
        return True
    except Exception as e:
        if not _load_error_logged:
            print(f"[classifier] Failed to load {MODEL_PATH}: {e}")
            _load_error_logged = True
        _model = None
        return False


def classify_landmarks_proba(landmarks, handedness: Optional[str] = None) -> Optional[Tuple[List[str], np.ndarray]]:
    """
    Get full class probability distribution for EMA smoothing.
    Returns (labels, probabilities) or None if no hand/model.
    """
    if landmarks is None or not _load_model():
        return None

    # Extract 21 landmarks
    landmarks_21 = None
    if isinstance(landmarks, dict) and "hands_detected" in landmarks:
        if landmarks.get("hands_detected", 0) == 0:
            return None
        lms = landmarks.get("landmarks", [])
        if not lms or len(lms) == 0 or len(lms[0]) == 0:
            return None
        landmarks_21 = lms[0]
    elif isinstance(landmarks, list) and len(landmarks) > 0 and isinstance(landmarks[0], list):
        if len(landmarks[0]) == 0:
            return None
        landmarks_21 = landmarks[0]
    else:
        landmarks_21 = landmarks

    try:
        feat = landmarks_to_feature(landmarks_21, is_left=False)
        if hasattr(_model, "predict_proba"):
            probs = _model.predict_proba([feat])[0]
            labels = list(_labels) if _labels is not None else list(_model.classes_)
            return labels, probs
        else:
            pred = str(_model.predict([feat])[0])
            labels = list(_labels) if _labels is not None else [pred]
            probs = np.array([1.0 if l == pred else 0.0 for l in labels], dtype=np.float32)
            return labels, probs
    except Exception as e:
        print(f"[classifier] classify_proba error: {e}")
        return None


def classify_landmarks(landmarks, handedness: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Classify single frame directly (highest probability letter)."""
    res = classify_landmarks_proba(landmarks, handedness)
    if res is None:
        return None
    labels, probs = res
    idx = int(np.argmax(probs))
    return {"letter": str(labels[idx]), "confidence": float(probs[idx])}


def get_labels() -> Optional[List[str]]:
    if not _load_model():
        return None
    return list(_labels) if _labels is not None else list(getattr(_model, "classes_", []))
