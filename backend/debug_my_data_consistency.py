#!/usr/bin/env python3
"""
debug_my_data_consistency.py — Check if my_own_data.csv is wrist-relative (fixed) or absolute (broken).

For each row, the active hand's wrist landmark (first x,y,z triplet) should be at or very near (0,0,0)
if the file was saved AFTER the wrist-relative fix in backend/app.py's collect_sample handler.
If saved BEFORE the fix, wrist will be at absolute image coords (e.g. 0.5,0.7 etc, not zero).

Reports per-letter and overall percentages.

Usage:
  python backend/debug_my_data_consistency.py
  python backend/debug_my_data_consistency.py --csv backend/data/isl_landmarks/my_own_data.csv
  python backend/debug_my_data_consistency.py --threshold 0.001
"""
import argparse
import csv
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ALLOWED = [chr(ord("A")+i) for i in range(26) if chr(ord("A")+i) not in ("H","J","Y")]
THRESH = 0.001  # wrist at 0,0,0 within 1e-3 is considered wrist-relative

def is_wrist_near_zero(row_vals, thresh=THRESH):
    """
    row_vals: 126 floats left+right
    Returns (is_wrist_zero, active_hand, wrist_xyz, max_abs_wrist)
    Active hand is the one with larger sum (non-zero).
    Wrist is first triplet of active hand.
    """
    arr = np.array(row_vals, dtype=np.float64).reshape(2,21,3)
    left_sum = np.abs(arr[0]).sum()
    right_sum = np.abs(arr[1]).sum()
    if left_sum < 1e-6 and right_sum < 1e-6:
        return None, None, None, None  # no hand
    # Pick active hand (larger sum, like train)
    if left_sum > right_sum:
        active = arr[0]
        hand = "left"
    else:
        active = arr[1]
        hand = "right"
    wrist = active[0]  # x,y,z of landmark 0
    max_abs = np.max(np.abs(wrist))
    is_zero = max_abs < thresh
    # Also check if wrist is exactly 0,0,0 vs 0.5,0.5 etc
    return is_zero, hand, wrist, max_abs

def main():
    ap = argparse.ArgumentParser(description="Check my_own_data.csv wrist-relative consistency")
    ap.add_argument("--csv", type=str, default="backend/data/isl_landmarks/my_own_data.csv", help="Path to CSV")
    ap.add_argument("--threshold", type=float, default=THRESH, help="Threshold for wrist near zero")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"No file at {csv_path}")
        return

    # Also check timestamps
    import os, time
    stat = csv_path.stat()
    print(f"File: {csv_path}")
    print(f"  Size: {stat.st_size} bytes, Rows: {sum(1 for _ in open(csv_path)) - 1} data rows")
    print(f"  Modified: {time.ctime(stat.st_mtime)}")
    print(f"  Birth (if available): {time.ctime(stat.st_birthtime) if hasattr(stat, 'st_birthtime') else 'n/a'}")
    # Check app.py timestamp
    app_path = Path("backend/app.py")
    if app_path.exists():
        s = app_path.stat()
        print(f"backend/app.py modified: {time.ctime(s.st_mtime)}")
        # Check if fix is present
        content = app_path.read_text()
        has_fix = "wrist_x = float(hand_lms[0].get" in content and "rel[:,0]" not in content and "wrist_x" in content
        # Actually check for our fix: wrist_x, wrist_y, wrist_z subtraction
        has_wrist_fix = 'wrist_x = float(hand_lms[0].get("x"' in content
        print(f"  app.py has wrist-relative fix: {has_wrist_fix}")
        if has_wrist_fix:
            # Find line numbers
            for i, line in enumerate(content.splitlines(), 1):
                if "wrist_x" in line:
                    print(f"    line {i}: {line.strip()}")
                    break

    print(f"\nThreshold for wrist near (0,0,0): {args.threshold}")
    print("Per-row check: active hand wrist should be ~0,0,0 if fixed, else ~0.3-0.7 if absolute")
    print("-" * 80)

    total = 0
    near_zero = 0
    per_letter_total = Counter()
    per_letter_zero = Counter()
    per_letter_wrist_vals = defaultdict(list)
    worst_examples = []  # store non-zero wrists

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        # Validate header
        expected_cols = 127  # 126 + label
        print(f"Header cols: {len(header)} expected 127, has label: {'label' in header}")
        for row in reader:
            label = str(row.get("label", "")).strip().upper()
            # Get 126 values in order
            vals = []
            for col in header:
                if col == "label":
                    continue
                try:
                    vals.append(float(row[col]))
                except:
                    vals.append(0.0)
            if len(vals) != 126:
                print(f"WARN: row has {len(vals)} vals, expected 126")
                continue
            is_zero, hand, wrist, max_abs = is_wrist_near_zero(vals, thresh=args.threshold)
            if is_zero is None:
                continue
            total += 1
            per_letter_total[label] += 1
            if is_zero:
                near_zero += 1
                per_letter_zero[label] += 1
            per_letter_wrist_vals[label].append(wrist)
            if not is_zero and len(worst_examples) < 5:
                worst_examples.append((label, hand, wrist, max_abs, vals[:6]))

    print(f"\nOverall: {near_zero}/{total} rows ({near_zero/total*100:.1f}%) have wrist ~0,0,0 (fixed)")
    print(f"         {total-near_zero}/{total} rows ({(total-near_zero)/total*100:.1f}%) have NON-ZERO wrist (broken absolute)")
    print("\nPer-letter breakdown:")
    print(f"{'Letter':6} {'Total':6} {'Wrist~0':8} {'%_fixed':8} {'Status'}")
    print("-"*50)
    for lbl in sorted(ALLOWED):
        tot = per_letter_total.get(lbl, 0)
        z = per_letter_zero.get(lbl, 0)
        pct = (z/tot*100) if tot else 0
        status = "OK (fixed)" if pct > 99 else "BROKEN (absolute)" if pct < 1 else "MIXED"
        flag = " <<< FLAG <10 samples" if tot < 10 else ""
        if lbl == "Z":
            # Z should be 0, but we check
            if tot == 0:
                status = "MISSING (Z not recorded, expected)"
        print(f"{lbl:6} {tot:6} {z:8} {pct:7.1f}%  {status}{flag}")
        if tot > 0 and tot < 10:
            print(f"  -> WARNING: {lbl} has only {tot} samples, too few for reliable training")

    # Also check for missing letters
    missing = [lbl for lbl in ALLOWED if per_letter_total.get(lbl,0)==0]
    if missing:
        print(f"\nMissing letters (0 samples): {missing} — model will NOT predict these")
        if "Z" in missing:
            print("  Z is expected missing (you said Z not yet recorded) — OK, known gap")
        others = [m for m in missing if m not in ("Z",)]
        if others:
            print(f"  Other missing: {others} — will be gaps")

    if worst_examples:
        print("\nExample BROKEN rows (wrist not at 0,0,0):")
        for lbl, hand, wrist, max_abs, vals in worst_examples:
            print(f"  {lbl} hand={hand} wrist={wrist} max_abs={max_abs:.3f} first6={vals[:6]}")

    # Verdict
    print("\n" + "="*80)
    if total > 0 and near_zero == total:
        print("VERDICT: ALL rows are wrist-relative (0,0,0) — file is CORRECT, fix was BEFORE recording.")
        print("  → Wrong-prediction issue lies elsewhere (check video_frame vs collect_sample preprocessing).")
    elif total > 0 and near_zero == 0:
        print("VERDICT: ALL rows are ABSOLUTE (non-zero wrist) — file is BROKEN, entire file predates fix.")
        print("  → This is all-or-nothing as you recorded A-X in one session. Entire file needs migration.")
        print("  → Migration is mechanically recoverable: recompute each row wrist-relative (subtract wrist).")
    elif total > 0 and near_zero < total * 0.9:
        print(f"VERDICT: MIXED {near_zero}/{total} fixed, {total-near_zero} broken — partial file, some sessions before/after fix.")
        print("  → Migration should be applied to non-zero rows only, or re-record.")
    else:
        print(f"VERDICT: {near_zero}/{total} fixed — mostly fixed but some broken rows remain.")

    # Also check Z
    if "Z" in missing:
        print("\nNote: Z is missing (0 samples) — training will produce 22-class model, Z will never be predicted. Known gap.")

if __name__ == "__main__":
    main()
