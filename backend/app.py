import os
import base64
import csv
import threading
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from flask import Flask, request
from flask_socketio import SocketIO, emit

load_dotenv()

# landmarks import with graceful fallback
try:
    import landmarks  # backend/landmarks.py
    print(f"[app] landmarks module loaded (MEDIAPIPE_AVAILABLE={landmarks.MEDIAPIPE_AVAILABLE})")
except Exception as _e:
    print(f"[app] WARN: landmarks import failed — falling back to plain ack: {_e}")
    landmarks = None  # type: ignore

# classifier import — lightweight, gracefully disabled if model not trained yet
try:
    import classifier  # backend/classifier.py
    print(f"[app] classifier module loaded")
except Exception as _e:
    print(f"[app] WARN: classifier import failed — recognized_letter disabled: {_e}")
    classifier = None  # type: ignore

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
socketio = SocketIO(app, cors_allowed_origins="*")

# --- video_frame streaming state (per-client counters) ---
frame_counters: dict[str, int] = {}

# --- Step 3: Per-Client Real-Time Probability EMA Smoother (Continuous, 0 Lag) ---
client_ema_probs: dict[str, np.ndarray | None] = {}
EMA_ALPHA = 0.55  # Optimal responsiveness vs noise suppression for 5fps
MIN_EMIT_CONFIDENCE = 0.35  # Allows valid letters (C, E, B) to emit smoothly while filtering 15-25% flicker


# --- Data collection (my_own_data.csv) — 126-col format (left 63 + right 63) ---
MY_OWN_DATA_PATH = Path(__file__).parent / "data" / "isl_landmarks" / "my_own_data.csv"
MY_OWN_DATA_HEADER = (
    [f"left_lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
    + [f"right_lm{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
    + ["label"]
)
_collect_lock = threading.Lock()
ALLOWED_LABELS = set([chr(ord("A") + i) for i in range(26) if chr(ord("A") + i) not in ("H", "J", "Y")])


def _ensure_own_data_header():
    MY_OWN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MY_OWN_DATA_PATH.exists():
        with open(MY_OWN_DATA_PATH, "w", newline="") as f:
            csv.writer(f).writerow(MY_OWN_DATA_HEADER)


def _landmarks_to_126_row(landmarks_list, handedness_list):
    """
    Convert landmarks.py output (list of hands, each 21 dicts) + handedness
    into 126 floats: left 63 + right 63. Missing hand = 63 zeros.
    """
    left = [0.0] * 63
    right = [0.0] * 63

    if not landmarks_list:
        return left + right

    for idx, hand_lms in enumerate(landmarks_list[:2]):
        handed = handedness_list[idx] if idx < len(handedness_list) else None
        is_left = False
        if isinstance(handed, str):
            is_left = handed.lower().startswith("left")
        elif isinstance(handed, dict) and "category_name" in handed:
            is_left = str(handed["category_name"]).lower().startswith("left")
        else:
            is_left = False if len(landmarks_list) == 1 else (idx == 0)

        wrist_x = float(hand_lms[0].get("x", 0))
        wrist_y = float(hand_lms[0].get("y", 0))
        wrist_z = float(hand_lms[0].get("z", 0))
        flat = []
        for lm in hand_lms:
            flat.extend([
                float(lm.get("x", 0)) - wrist_x,
                float(lm.get("y", 0)) - wrist_y,
                float(lm.get("z", 0)) - wrist_z,
            ])

        flat = (flat + [0.0] * 63)[:63]

        if is_left:
            if all(v == 0.0 for v in left):
                left = flat
            elif all(v == 0.0 for v in right):
                right = flat
        else:
            if all(v == 0.0 for v in right):
                right = flat
            elif all(v == 0.0 for v in left):
                left = flat

    return left + right


def _get_collect_counts():
    if not MY_OWN_DATA_PATH.exists():
        return {lbl: 0 for lbl in sorted(ALLOWED_LABELS)}, 0
    try:
        with open(MY_OWN_DATA_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            counts = Counter()
            total = 0
            for row in reader:
                lbl = str(row.get("label", "")).strip().upper()
                if lbl in ALLOWED_LABELS:
                    counts[lbl] += 1
                    total += 1
            full = {lbl: counts.get(lbl, 0) for lbl in sorted(ALLOWED_LABELS)}
            return full, total
    except Exception as e:
        print(f"[collect] count read error: {e}")
        return {lbl: 0 for lbl in sorted(ALLOWED_LABELS)}, 0


@app.route('/')
def index():
    return "HandSign backend is running"


@socketio.on('connect')
def handle_connect():
    print(f"Phone connected (sid={getattr(request, 'sid', 'unknown')})")


@socketio.on('disconnect')
def handle_disconnect():
    sid = getattr(request, 'sid', None)
    if sid:
        frame_counters.pop(sid, None)
        client_ema_probs.pop(sid, None)
    print(f"Phone disconnected (sid={sid})")


@socketio.on('test_message')
def handle_test_message(data):
    print("Received from phone:", data)
    emit('recognized_word', {'word': 'hello'})


@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Streaming path: decodes JPEG, runs MediaPipe Hands, applies continuous Probability EMA,
    and emits frame_ack + recognized_letter.
    """
    try:
        sid = getattr(request, 'sid', 'default')

        count = frame_counters.get(sid, 0) + 1
        frame_counters[sid] = count

        b64_str: str | None = None
        if isinstance(data, dict):
            b64_str = data.get('image') or data.get('data') or data.get('frame')
        elif isinstance(data, str):
            b64_str = data

        if not b64_str:
            emit('frame_ack', {'status': 'error', 'message': 'no image', 'count': count})
            return

        if ',' in b64_str and b64_str.startswith('data:'):
            b64_str = b64_str.split(',', 1)[1]

        raw_bytes = base64.b64decode(b64_str)

        # ---- MediaPipe Landmark Processing ----
        lm_result = None
        if landmarks is not None and getattr(landmarks, "MEDIAPIPE_AVAILABLE", False):
            try:
                lm_result = landmarks.process_frame(raw_bytes)
            except Exception as lm_e:
                print(f"[video_frame #{count}] landmarks error: {lm_e}")
                lm_result = {"hands_detected": 0, "landmarks": [], "handedness": [], "inference_ms": 0, "reason": str(lm_e)}
        else:
            lm_result = None

        # ---- Emit Extended Frame Ack ----
        if lm_result is not None:
            emit('frame_ack', {
                'status': 'frame_received',
                'count': count,
                'size': len(raw_bytes),
                'hands_detected': lm_result.get('hands_detected', 0),
                'landmarks': lm_result.get('landmarks', []),
                'handedness': lm_result.get('handedness', []),
                'inference_ms': lm_result.get('inference_ms', 0),
                'dropped': lm_result.get('dropped', False),
                'reason': lm_result.get('reason'),
            })
        else:
            emit('frame_ack', {'status': 'frame_received', 'count': count, 'size': len(raw_bytes)})

        # ---- Step 3: Probability EMA Smoothing Classifier ----
        if classifier is not None and lm_result is not None:
            try:
                hands_detected = lm_result.get("hands_detected", 0)
                if hands_detected >= 1:
                    lms_list = lm_result.get("landmarks", [])
                    handed_list = lm_result.get("handedness", ["Right"])
                    if lms_list and len(lms_list[0]) == 21:
                        primary_lms = lms_list[0]
                        primary_handedness = handed_list[0] if handed_list else "Right"

                        # Classify current frame probability distribution
                        proba_res = classifier.classify_landmarks_proba(primary_lms, handedness=primary_handedness)
                        if proba_res is not None:
                            labels, curr_probs = proba_res
                            prev_ema = client_ema_probs.get(sid)

                            if prev_ema is None or len(prev_ema) != len(curr_probs):
                                smooth_probs = curr_probs.copy()
                            else:
                                smooth_probs = EMA_ALPHA * curr_probs + (1.0 - EMA_ALPHA) * prev_ema

                            client_ema_probs[sid] = smooth_probs

                            top_idx = int(np.argmax(smooth_probs))
                            rec_letter = str(labels[top_idx])
                            rec_conf = float(smooth_probs[top_idx])

                            if count % 10 == 0 or rec_conf >= 0.70:
                                print(f"[classify #{count}] {rec_letter} (conf={rec_conf:.2f}, hands={hands_detected})")

                            if rec_conf >= MIN_EMIT_CONFIDENCE:
                                emit('recognized_letter', {
                                    "letter": rec_letter,
                                    "confidence": round(rec_conf, 3)
                                })
                            else:
                                emit('recognized_letter', {"letter": None, "confidence": 0.0})
                else:
                    # No hand detected -> immediately clear EMA and emit null
                    if sid in client_ema_probs and client_ema_probs[sid] is not None:
                        client_ema_probs[sid] = None
                    emit('recognized_letter', {"letter": None, "confidence": 0.0})

            except Exception as ce:
                print(f"[classify] error: {ce}")

    except Exception as e:
        print(f"[video_frame] error decoding frame: {e}")
        import traceback
        traceback.print_exc()
        emit('frame_ack', {'status': 'error', 'message': str(e)})


@socketio.on('collect_sample')
def handle_collect_sample(data):
    """
    Data collection: {image: base64, label: "A"}
    """
    try:
        sid = getattr(request, 'sid', 'default')
        label = str((data or {}).get("label", "")).strip().upper() if isinstance(data, dict) else ""
        if label not in ALLOWED_LABELS:
            emit('collect_result', {'status': 'error', 'message': f'Invalid label {label}. Use one of {sorted(ALLOWED_LABELS)}'})
            return

        b64_str = None
        if isinstance(data, dict):
            b64_str = data.get("image") or data.get("data") or data.get("frame")
        elif isinstance(data, str):
            b64_str = data

        if not b64_str:
            emit('collect_result', {'status': 'error', 'message': 'No image'})
            return

        if ',' in b64_str and b64_str.startswith('data:'):
            b64_str = b64_str.split(',', 1)[1]

        raw_bytes = base64.b64decode(b64_str)

        if landmarks is None or not getattr(landmarks, "MEDIAPIPE_AVAILABLE", False):
            emit('collect_result', {'status': 'error', 'message': 'Landmarks not available'})
            return

        lm_result = landmarks.process_frame(raw_bytes)
        if lm_result.get("hands_detected", 0) == 0:
            emit('collect_result', {'status': 'no_hand', 'message': 'No hand detected — not saved', 'label': label})
            return

        landmarks_list = lm_result.get("landmarks", [])
        handedness_list = lm_result.get("handedness", [])
        row_126 = _landmarks_to_126_row(landmarks_list, handedness_list)

        _ensure_own_data_header()
        with _collect_lock:
            with open(MY_OWN_DATA_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row_126 + [label])

        counts, total = _get_collect_counts()
        print(f"[collect] saved {label} total={total} counts={counts.get(label)} sid={sid}")
        emit('collect_result', {
            'status': 'ok',
            'message': f'Saved {label}',
            'label': label,
            'count': counts.get(label, 0),
            'counts': counts,
            'total': total,
        })

    except Exception as e:
        print(f"[collect] error: {e}")
        import traceback
        traceback.print_exc()
        emit('collect_result', {'status': 'error', 'message': str(e)})


@socketio.on('get_collect_counts')
def handle_get_collect_counts(data=None):
    try:
        counts, total = _get_collect_counts()
        emit('collect_counts', {'counts': counts, 'total': total})
    except Exception as e:
        emit('collect_counts', {'counts': {}, 'total': 0, 'error': str(e)})


if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5001'))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    socketio.run(app, host=host, port=port, debug=debug)