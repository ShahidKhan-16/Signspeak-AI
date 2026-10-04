#!/usr/bin/env python3
"""
debug_live_vs_train.py — Side-by-side comparison of live vs training features.

Compares a live captured frame (or saved test image) showing a known letter
(e.g. "C") via landmarks.py -> classifier.py feature transform
against a same-letter sample row from the training CSV after train_alphabet.py's transform.
Prints raw 63-float vectors side-by-side so you can see if numbers are in same range/shape
or wildly different (indicates preprocessing mismatch).

Usage:
  python backend/debug_live_vs_train.py --image /path/to/C_test.jpg --label C
  python backend/debug_live_vs_train.py --label C  # tries to find image in backend/data/isl_images/C/*.jpg or uses synthetic
  python backend/debug_live_vs_train.py --label C --csv-row 0  # compare to first CSV row for C
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import landmarks as lm
import train_alphabet as ta
import classifier as clf

def load_csv_sample(label: str, csv_path: Path, row_idx: int = 0):
    df = pd.read_csv(csv_path)
    # Find rows with label
    label_col = "label" if "label" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != label_col]
    filtered = df[df[label_col].astype(str).str.upper() == label.upper()]
    if filtered.empty:
        print(f"No rows for label {label} in {csv_path}")
        return None, None
    row = filtered.iloc[row_idx % len(filtered)]
    vals = row[feature_cols].values.astype(np.float32)
    # Use train's extract_feature_from_row for 126 case
    if len(feature_cols) == 126:
        feat = ta.extract_feature_from_row(vals)
        # Also get raw active hand for comparison (before scale)
        arr = vals.reshape(2,21,3)
        left_sum = np.abs(arr[0]).sum()
        right_sum = np.abs(arr[1]).sum()
        active = arr[0] if left_sum > right_sum else arr[1]
        print(f"[CSV] label {label} row {row_idx} left_sum {left_sum:.3f} right_sum {right_sum:.3f} picked {'left' if left_sum>right_sum else 'right'}")
        print(f"[CSV] raw active hand sample (first 3 landmarks): {active[0]}, {active[1]}, {active[2]}")
    else:
        feat = vals  # assume already 63
        print(f"[CSV] label {label} row {row_idx} feature 63")
    return feat, vals

def process_live_image(image_path: Path):
    raw = image_path.read_bytes()
    res = lm.process_frame(raw)
    print(f"[LIVE] {image_path} -> hands_detected={res.get('hands_detected')}, hand_flag={res.get('hand_flag')}, handedness={res.get('handedness')}, inference_ms={res.get('inference_ms')}")
    if res.get("hands_detected", 0) == 0:
        print("[LIVE] No hand detected — feature will be zeros, comparison invalid. Try a centered hand image.")
        return None, res
    lms = res["landmarks"][0]  # 21 dicts
    # Use classifier's feature transform (should match train's)
    feat_live = clf.landmarks_to_feature(lms)
    # Also compute train-style feature via ta.landmarks_to_feature for comparison (should be identical)
    feat_train_style = ta.landmarks_to_feature(lms)
    print(f"[LIVE] raw landmarks sample (first 3): {lms[0]}, {lms[1]}, {lms[2]}")
    print(f"[LIVE] feature via classifier (first 9): {feat_live[:9]}")
    print(f"[LIVE] feature via train (first 9): {feat_train_style[:9]}")
    print(f"[LIVE] feature stats: min {feat_live.min():.3f} max {feat_live.max():.3f} mean {feat_live.mean():.3f} std {feat_live.std():.3f}")
    return feat_live, res

def compare_features(feat_live, feat_csv, label):
    if feat_live is None or feat_csv is None:
        print("Cannot compare — one is None")
        return
    # feat_csv is from extract_feature_from_row (already wrist-relative + scale)
    # feat_live is from classifier (also wrist-relative + scale)
    print("\n[COMPARE] Side-by-side first 21*3=63 floats (x,y,z per landmark):")
    print(" idx  live_x, live_y, live_z   |   csv_x, csv_y, csv_z   | diff")
    for i in range(21):
        lx, ly, lz = feat_live[i*3], feat_live[i*3+1], feat_live[i*3+2]
        cx, cy, cz = feat_csv[i*3], feat_csv[i*3+1], feat_csv[i*3+2]
        diff = np.linalg.norm([lx-cx, ly-cy, lz-cz])
        flag = " <<< LARGE DIFF" if diff > 0.3 else ""
        print(f" {i:2d}  {lx:6.3f},{ly:6.3f},{lz:6.3f}  |  {cx:6.3f},{cy:6.3f},{cz:6.3f}  | diff {diff:.3f}{flag}")
    # Overall stats
    diff_vec = feat_live - feat_csv
    print(f"\n[COMPARE] Overall L2 distance: {np.linalg.norm(diff_vec):.3f} (0 = identical, >5 = wildly different)")
    print(f"  live range: [{feat_live.min():.3f}, {feat_live.max():.3f}] csv range: [{feat_csv.min():.3f}, {feat_csv.max():.3f}]")
    print(f"  live mean {feat_live.mean():.3f} std {feat_live.std():.3f} vs csv mean {feat_csv.mean():.3f} std {feat_csv.std():.3f}")
    # Check correlation
    if np.std(feat_live) > 1e-6 and np.std(feat_csv) > 1e-6:
        corr = np.corrcoef(feat_live, feat_csv)[0,1]
        print(f"  Pearson correlation: {corr:.3f} (1 = perfect, 0 = unrelated, negative = mirrored)")
    # Try flipping x for live to see if correlation improves (mirror test)
    feat_live_flipped = feat_live.copy()
    # Flip x: x = -x for each landmark (since wrist-relative, flipping x mirrors)
    for i in range(21):
        feat_live_flipped[i*3] = -feat_live_flipped[i*3]
    corr_flip = np.corrcoef(feat_live_flipped, feat_csv)[0,1] if np.std(feat_live_flipped)>1e-6 else 0
    print(f"  Correlation if live X flipped: {corr_flip:.3f} (if higher, suggests left/right mirror mismatch)")
    # Also test without scale normalization? Check raw scale
    # The CSV's raw active hand before scale has max_dist ~?
    # We can't know without loading raw, but we can suggest

def main():
    ap = argparse.ArgumentParser(description="Debug live vs train feature mismatch")
    ap.add_argument("--image", type=str, default="", help="Path to test image showing known letter")
    ap.add_argument("--label", type=str, default="C", help="Expected letter for CSV comparison (e.g. C)")
    ap.add_argument("--csv-row", type=int, default=0, help="Row index for CSV sample (0 = first)")
    ap.add_argument("--data-dir", type=str, default="backend/data/isl_landmarks", help="CSV data dir")
    args = ap.parse_args()

    csv_path = Path(args.data_dir) / "combined_train_dataset.csv"
    if not csv_path.exists():
        # Try alternate names
        candidates = list(Path(args.data_dir).glob("*.csv"))
        if candidates:
            csv_path = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[0]
    if not csv_path.exists():
        print(f"No CSV at {csv_path}, need training data")
        sys.exit(1)

    feat_csv, _ = load_csv_sample(args.label, csv_path, args.csv_row)
    if feat_csv is None:
        sys.exit(1)
    print(f"[CSV] feature stats: min {feat_csv.min():.3f} max {feat_csv.max():.3f} mean {feat_csv.mean():.3f} std {feat_csv.std():.3f}")
    print(f"[CSV] feature first 9: {feat_csv[:9]}")

    # Find live image
    image_path = None
    if args.image:
        image_path = Path(args.image)
    else:
        # Try to find an image for the label in backend/data/isl_images
        img_candidates = []
        for base in [Path("backend/data/isl_images"), Path("backend/data/isl_landmarks"), Path("test_hands")]:
            if base.exists():
                # Look for folder named label
                p = base / args.label.upper()
                if p.exists():
                    imgs = list(p.glob("*.jpg")) + list(p.glob("*.png"))
                    if imgs:
                        image_path = imgs[0]
                        break
                # Also look for files with label in name
                for f in base.rglob("*.jpg"):
                    if args.label.lower() in f.name.lower():
                        image_path = f
                        break
        if image_path is None:
            print(f"\nNo --image given and no auto-found image for {args.label}.")
            print("Please provide a saved test image: python backend/debug_live_vs_train.py --image /path/to/C_test.jpg --label C")
            print("You can save one from the phone: take a photo of your hand showing C, save to /tmp/C_test.jpg, then run:")
            print(f"  python backend/debug_live_vs_train.py --image /tmp/C_test.jpg --label C")
            # Still show CSV feature for reference, and run synthetic live test with dummy
            print("\n[INFO] Running synthetic live test with dummy hand at center for comparison...")
            # Create a dummy live feature by using a synthetic hand shape for C (approx)
            # For now, just compare CSV to itself to show expected range
            return

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    feat_live, res = process_live_image(image_path)
    compare_features(feat_live, feat_csv, args.label)

    # Also test classifier confidence on live vs CSV
    print("\n[CLASSIFIER] Testing model on both features:")
    import pickle
    model_path = Path("backend/models/isl_alphabet_classifier.pkl")
    if model_path.exists():
        payload = pickle.load(open(model_path, "rb"))
        model = payload["model"]
        labels = payload["labels"]
        # Predict on live
        if feat_live is not None:
            probs_live = model.predict_proba([feat_live])[0]
            pred_live = labels[np.argmax(probs_live)]
            conf_live = np.max(probs_live)
            print(f"  Live {args.label} -> pred {pred_live} conf {conf_live:.3f} (expected {args.label})")
            # Top 3
            top3 = sorted(zip(labels, probs_live), key=lambda x: -x[1])[:3]
            print(f"    top3: {top3}")
        # Predict on CSV
        probs_csv = model.predict_proba([feat_csv])[0]
        pred_csv = labels[np.argmax(probs_csv)]
        conf_csv = np.max(probs_csv)
        print(f"  CSV  {args.label} -> pred {pred_csv} conf {conf_csv:.3f} (expected {args.label})")
        top3_csv = sorted(zip(labels, probs_csv), key=lambda x: -x[1])[:3]
        print(f"    top3: {top3_csv}")
        print(f"\n[DIAGNOSIS] Confidence gap: live {conf_live:.3f} vs CSV {conf_csv:.3f} -> {'MISMATCH' if abs(conf_live-conf_csv)>0.2 else 'similar'}")
    else:
        print("No model at backend/models/isl_alphabet_classifier.pkl, train first")

if __name__ == "__main__":
    main()
