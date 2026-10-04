#!/usr/bin/env python3
"""
train_alphabet.py — Train ISL Alphabet Classifier with 90-Dim Geometric Features & Zero-Leakage Augmentation.

Key Improvements:
1. 90-Dimensional Geometric Feature Vector (Palm-base normalized coords, 15 joint angles, curl & pinch distances).
2. Consistent Selfie Coordinates: Operates directly in the natural camera frame without erratic selfie handedness flips.
3. Strict Data Splitting: Dataset is split into train/test BEFORE any data augmentation.
4. Training-Only Data Augmentation: 3D random rotations, scaling, and Gaussian joint jitter expand training samples 15x.
5. Model Parity: Uses classifier.py:landmarks_to_feature() directly so training and live inference use identical math.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import classifier as clf

JOINT_TRIPLETS = clf.JOINT_TRIPLETS
TIPS = clf.TIPS
landmarks_to_feature = clf.landmarks_to_feature


def extract_raw_21_from_row_126(row_126: np.ndarray) -> np.ndarray:
    """Extract active hand's 21x3 raw coordinates from a 126-column row."""
    arr = row_126.reshape(2, 21, 3)
    left_sum = np.abs(arr[0]).sum()
    right_sum = np.abs(arr[1]).sum()
    # Pick whichever hand slot is active (non-zero)
    if left_sum > right_sum:
        return arr[0]
    else:
        return arr[1]


def rotate_landmarks_3d(lms: np.ndarray, rx: float, ry: float, rz: float) -> np.ndarray:
    """Rotate (21, 3) landmarks around origin by angles (in radians)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)

    R = Rz @ Ry @ Rx
    return (lms @ R.T).astype(np.float32)


def augment_single_hand(lms_21: np.ndarray, n_aug: int = 16, rng: np.random.Generator = None) -> list:
    """
    Generate n_aug realistic coordinate variations from 1 raw hand.
    Includes horizontal mirroring (X-flip) so the model is 100% invariant to:
    - Left Hand vs Right Hand signing
    - Front camera selfie mirroring vs unmirrored streams
    """
    if rng is None:
        rng = np.random.default_rng(42)

    augmented_features = []
    
    # 1. Base hand
    orig_feat = landmarks_to_feature(lms_21, is_left=False)
    augmented_features.append(orig_feat)

    wrist = lms_21[0].copy()
    rel = lms_21 - wrist

    # 2. Horizontally mirrored hand (X-flip)
    rel_mirrored = rel.copy()
    rel_mirrored[:, 0] = -rel_mirrored[:, 0]
    mirrored_feat = landmarks_to_feature(rel_mirrored, is_left=False)
    augmented_features.append(mirrored_feat)

    # 3. Generate 3D rotations, scaling, and Gaussian jitter for BOTH original and mirrored hands
    half_aug = max(1, (n_aug - 2) // 2)
    for hand_base in (rel, rel_mirrored):
        for _ in range(half_aug):
            # Random 3D rotations: Z (in-plane) ±15 deg, X/Y (tilt) ±12 deg
            rz = float(rng.uniform(-np.radians(15), np.radians(15)))
            rx = float(rng.uniform(-np.radians(12), np.radians(12)))
            ry = float(rng.uniform(-np.radians(12), np.radians(12)))
            rot = rotate_landmarks_3d(hand_base, rx, ry, rz)

            # Random scaling: ±12%
            scale = float(rng.uniform(0.88, 1.12))
            scaled = rot * scale

            # Gaussian joint jitter
            jitter = rng.normal(0.0, 0.008, scaled.shape).astype(np.float32)
            jitter[0] = [0, 0, 0]
            aug_lms = scaled + jitter

            aug_feat = landmarks_to_feature(aug_lms, is_left=False)
            augmented_features.append(aug_feat)

    return augmented_features


def prepare_dataset_from_csv(csv_path: Path):
    """Load CSV, parse raw 21x3 hands and labels."""
    df = pd.read_csv(csv_path)
    label_col = "label" if "label" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != label_col]

    raw_hands = []
    labels = []

    for _, row in df.iterrows():
        lbl = str(row[label_col]).strip().upper()
        vals = row[feature_cols].values.astype(np.float32)
        if len(feature_cols) == 126:
            hand_21 = extract_raw_21_from_row_126(vals)
        elif len(feature_cols) == 63:
            hand_21 = vals.reshape(21, 3)
        else:
            continue
        raw_hands.append(hand_21)
        labels.append(lbl)

    return raw_hands, labels


def train_and_eval(X_train, y_train, X_test, y_test):
    """Train Random Forest classifier and output detailed metrics."""
    print(f"\n[train] Fitting RandomForest on {len(X_train)} augmented samples...")
    t0 = time.time()
    clf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    clf_model.fit(X_train, y_train)
    fit_sec = time.time() - t0
    print(f"[train] Training complete in {fit_sec:.2f}s")

    # Evaluation on UNTOUCHED held-out test set
    y_pred = clf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    labels = sorted(np.unique(np.concatenate([y_train, y_test])))

    print(f"\n=======================================================")
    print(f"       HELD-OUT TEST ACCURACY: {acc * 100:.2f}% ({len(y_test)} pure samples)")
    print(f"=======================================================\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion Matrix (rows = True, cols = Predicted):")
    header = "     " + " ".join(f"{l:>3}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        print(f"{labels[i]:>3} | " + " ".join(f"{c:3d}" for c in row))

    return clf_model, acc, labels, cm


def main():
    ap = argparse.ArgumentParser(description="Train ISL Alphabet Classifier")
    ap.add_argument("--data-csv", type=str, default="backend/data/isl_landmarks/my_own_data.csv", help="Path to own CSV")
    ap.add_argument("--output", type=str, default="backend/models/isl_alphabet_classifier.pkl", help="Model output path")
    ap.add_argument("--test-size", type=float, default=0.2, help="Held-out test split ratio")
    ap.add_argument("--aug-factor", type=int, default=15, help="Number of augmented samples per train sample")
    args = ap.parse_args()

    csv_path = Path(args.data_csv)
    if not csv_path.exists():
        print(f"Error: dataset not found at {csv_path}")
        sys.exit(1)

    print(f"[train] Loading {csv_path}...")
    raw_hands, labels = prepare_dataset_from_csv(csv_path)
    total_raw = len(raw_hands)
    print(f"[train] Loaded {total_raw} raw recordings across {len(set(labels))} classes: {sorted(set(labels))}")

    # 1. Split BEFORE Augmentation (Zero Leakage)
    indices = np.arange(total_raw)
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=42, stratify=labels
    )
    print(f"[train] Split: {len(train_idx)} train recordings, {len(test_idx)} held-out test recordings (untouched)")

    # 2. Build Held-Out Test Set (Features without augmentation)
    X_test = []
    y_test = []
    for idx in test_idx:
        feat = landmarks_to_feature(raw_hands[idx], is_left=False)
        X_test.append(feat)
        y_test.append(labels[idx])
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test)

    # 3. Augment ONLY Training Set
    rng = np.random.default_rng(42)
    X_train = []
    y_train = []
    print(f"[train] Augmenting training set ({args.aug_factor}x per sample)...")
    for idx in train_idx:
        aug_feats = augment_single_hand(raw_hands[idx], n_aug=args.aug_factor, rng=rng)
        for f in aug_feats:
            X_train.append(f)
            y_train.append(labels[idx])
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)
    print(f"[train] Augmented Training Set: {len(X_train)} samples ({X_train.shape[1]} features)")

    # 4. Train and Evaluate
    clf_model, acc, labels_list, cm = train_and_eval(X_train, y_train, X_test, y_test)

    # 5. Save Model
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": clf_model,
        "labels": labels_list,
        "feature_type": "geometric_90dim_palm_norm_angles_dists",
        "source": f"own data {csv_path.name} with {args.aug_factor}x train augmentation",
        "held_out_accuracy": float(acc),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n[train] Saved upgraded 90-dim model to {out_path}")


if __name__ == "__main__":
    main()
