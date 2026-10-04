#!/usr/bin/env python3
"""
verify_landmarks.py — Rigorous sanity check for backend/landmarks.py

Tests whether hand_flag gate, decoding, normalization, and letterbox
math produce *numerically correct* landmarks, not just "no crash".

Why this exists:
  process_frame() currently SKIPS palm detection/cropping and runs
  hand_landmarks_detector.tflite on the full letterboxed frame.
  That will hallucinate or misplace landmarks when the hand is:
    off-center, small/far, near image edge, partially occluded,
    or not present. This script makes that failure mode *visible*
    before you build OpenHands classification on top.

Usage:
  python backend/verify_landmarks.py --input test_hands --output verify_out
  python backend/verify_landmarks.py --input test_hands --output verify_out --save-all

  Input folder: 10-15 JPG/PNG of hands (centered, off-side, edge, far/small,
    occluded, no-hand). Filenames hint expected behavior:
    - *nohand*, *empty*, *background*, *no_hand* → expect hands_detected==0
    - *edge*, *side*, *offcenter* → expect degradation on full-frame path
    - *far*, *small* → expect low confidence / small bbox
  Output folder: annotated images + report. Use DEBUG_LANDMARKS logic but richer.

Checks per image (flags are strict — they fire on *obviously wrong* numbers):
  - hand_flag raw value + near-threshold band [0.40,0.60] (model is unsure)
  - hands_detected vs hand_flag consistency
  - landmarks clustered at one point (bbox width/height <0.05 or std <0.02)
  - landmarks clamped at border (many points at exactly 0 or 1 after undo)
  - bbox area <0.3% of image (hand far/small on full-frame → model sees tiny hand)
  - landmarks spread >0.95 (hallucinated hand filling entire frame)
  - world-z sanity (z outside ~ -0.5..0.5 suggests bad decode)
  - inference time >120ms (5fps budget) or dropped=True
  - no-hand control: expected no-hand but got detection → false positive
  - expected hand but hand_flag <0.5 → false negative (likely due to missing palm crop)

Exit code 0 if no *systematic* failure, 2 if many false positives/negatives
or clustered landmarks suggest the full-frame path is not trustworthy.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# allow `python backend/verify_landmarks.py` from project root or backend/
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import landmarks as lm  # backend/landmarks.py

HAND_CONNECTIONS = lm.HAND_CONNECTIONS

# Heuristic thresholds — tuned to be *obviously* wrong, not nit-picky
NEAR_THRESH_LOW, NEAR_THRESH_HIGH = 0.40, 0.60
CLUSTER_BBOX_THRESH = 0.05          # bbox w or h in normalized 0-1
CLUSTER_STD_THRESH = 0.02
SMALL_AREA_THRESH = 0.003           # bbox area <0.3% of image
LARGE_SPREAD_THRESH = 0.95
Z_MIN, Z_MAX = -0.5, 0.5
INFER_BUDGET_MS = 120

NO_HAND_HINTS = ("nohand", "no_hand", "empty", "background", "none", "no-hand")
EDGE_HINTS = ("edge", "side", "offcenter", "off_center", "corner")
FAR_HINTS = ("far", "small", "distant", "tiny")


def is_no_hand_expected(filename: str) -> bool:
    n = filename.lower()
    return any(h in n for h in NO_HAND_HINTS)


def is_edge_case(filename: str) -> bool:
    n = filename.lower()
    return any(h in n for h in EDGE_HINTS)


def is_far_case(filename: str) -> bool:
    n = filename.lower()
    return any(h in n for h in FAR_HINTS)


def load_image_bytes(path: Path) -> bytes:
    return path.read_bytes()


def compute_metrics(result: dict):
    hands = result.get("hands_detected", 0)
    lms = result.get("landmarks", [])
    # landmarks is list per hand: [[21*{x,y,z}]]
    flat = []
    if lms and len(lms) > 0 and len(lms[0]) > 0:
        flat = lms[0]
    if not flat:
        return {
            "bbox_w": 0, "bbox_h": 0, "bbox_area": 0,
            "std_x": 0, "std_y": 0, "spread": 0,
            "at_border": 0, "z_min": 0, "z_max": 0, "flat": [],
        }
    xs = np.array([p["x"] for p in flat], dtype=np.float64)
    ys = np.array([p["y"] for p in flat], dtype=np.float64)
    zs = np.array([p["z"] for p in flat], dtype=np.float64)
    bbox_w = float(xs.max() - xs.min())
    bbox_h = float(ys.max() - ys.min())
    bbox_area = bbox_w * bbox_h
    std_x = float(xs.std())
    std_y = float(ys.std())
    spread = float(np.sqrt(std_x**2 + std_y**2))
    # count points exactly at clamp border (0 or 1) — indicates pre-clamp OOB
    at_border = int(np.sum((xs == 0) | (xs == 1) | (ys == 0) | (ys == 1)))
    return {
        "bbox_w": bbox_w, "bbox_h": bbox_h, "bbox_area": bbox_area,
        "std_x": std_x, "std_y": std_y, "spread": spread,
        "at_border": at_border, "z_min": float(zs.min()), "z_max": float(zs.max()),
        "flat": flat,
    }


def flag_issues(path: Path, result: dict, metrics: dict):
    flags = []
    hand_flag = result.get("hand_flag", None)
    # hand_flag may be absent when no hand (result doesn't include it)
    # Try to infer from result: if hands_detected==0, hand_flag <0.5
    if hand_flag is None:
        # Use 0.0 as placeholder for no-hand case without flag
        hand_flag_val = 0.0 if result.get("hands_detected", 0) == 0 else 0.5
    else:
        hand_flag_val = float(hand_flag)

    # 1) near-threshold band — model is unsure
    if NEAR_THRESH_LOW <= hand_flag_val <= NEAR_THRESH_HIGH:
        flags.append(f"NEAR_THRESHOLD hand_flag={hand_flag_val:.3f} in [{NEAR_THRESH_LOW},{NEAR_THRESH_HIGH}]")

    # 2) clustered
    if result.get("hands_detected", 0) > 0:
        if metrics["bbox_w"] < CLUSTER_BBOX_THRESH and metrics["bbox_h"] < CLUSTER_BBOX_THRESH:
            flags.append(f"CLUSTERED bbox {metrics['bbox_w']:.3f}x{metrics['bbox_h']:.3f} <{CLUSTER_BBOX_THRESH}")
        if metrics["std_x"] < CLUSTER_STD_THRESH and metrics["std_y"] < CLUSTER_STD_THRESH:
            flags.append(f"CLUSTERED std x={metrics['std_x']:.3f} y={metrics['std_y']:.3f} <{CLUSTER_STD_THRESH}")

    # 3) at border (clamped OOB)
    if metrics["at_border"] >= 5:
        flags.append(f"BORDER_CLAMP {metrics['at_border']}/21 points at 0/1 (pre-clamp OOB)")

    # 4) suspiciously small bbox (far hand on full-frame)
    if result.get("hands_detected", 0) > 0 and metrics["bbox_area"] < SMALL_AREA_THRESH:
        flags.append(f"SMALL_BBOX area={metrics['bbox_area']:.5f} <{SMALL_AREA_THRESH} (hand far/small → full-frame path sees tiny hand)")

    # 5) suspiciously large spread (hallucinated)
    if metrics["bbox_w"] > LARGE_SPREAD_THRESH or metrics["bbox_h"] > LARGE_SPREAD_THRESH:
        flags.append(f"LARGE_SPREAD bbox {metrics['bbox_w']:.2f}x{metrics['bbox_h']:.2f} >{LARGE_SPREAD_THRESH}")

    # 6) z sanity
    if metrics["z_min"] < Z_MIN or metrics["z_max"] > Z_MAX:
        flags.append(f"Z_OOB z=[{metrics['z_min']:.3f},{metrics['z_max']:.3f}] outside [{Z_MIN},{Z_MAX}] (bad z decode)")

    # 7) inference budget
    infer = result.get("inference_ms", 0)
    if infer > INFER_BUDGET_MS:
        flags.append(f"SLOW inference {infer:.1f}ms >{INFER_BUDGET_MS}ms (will miss 5fps budget)")

    # 8) dropped
    if result.get("dropped"):
        flags.append("DROPPED frame skipped due to busy lock")

    # 9) no-hand control
    if is_no_hand_expected(path.name):
        if result.get("hands_detected", 0) > 0:
            flags.append(f"FALSE_POSITIVE expected no-hand ({path.name}) but got hands_detected={result['hands_detected']} flag={hand_flag_val:.3f}")

    # 10) expected hand but got no detection (likely full-frame without crop fails on edge/far)
    else:
        # Only flag as warning if not an edge/far case (those are expected to degrade)
        if result.get("hands_detected", 0) == 0 and hand_flag_val < 0.5:
            if is_edge_case(path.name) or is_far_case(path.name):
                flags.append(f"EXPECTED_DEGRADATION edge/far hand not detected on full-frame (flag={hand_flag_val:.3f}) — palm crop would fix")
            else:
                # For centered hand we expect detection; no detection here is a false negative
                # but we don't know if center or not — treat as INFO, not hard fail, unless many
                flags.append(f"NO_HAND_DETECTED flag={hand_flag_val:.3f} (if hand was centered, this is a false negative)")

    # 11) handedness near threshold
    hs = result.get("handedness_score", None)
    if hs is not None and 0.45 <= float(hs) <= 0.55:
        flags.append(f"HANDEDNESS_UNCERTAIN score={float(hs):.3f} near 0.5")

    return flags


def annotate_and_save(bgr: np.ndarray, result: dict, metrics: dict, flags, out_path: Path):
    h, w = bgr.shape[:2]
    vis = bgr.copy()
    flat = metrics.get("flat", [])

    # Draw bbox of landmarks
    if flat:
        xs = [p["x"] * w for p in flat]
        ys = [p["y"] * h for p in flat]
        x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 200, 0), 1)
        # draw points + connections
        for p in flat:
            cx, cy = int(p["x"] * w), int(p["y"] * h)
            cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)
        for a, b in HAND_CONNECTIONS:
            if a < len(flat) and b < len(flat):
                pa = (int(flat[a]["x"] * w), int(flat[a]["y"] * h))
                pb = (int(flat[b]["x"] * w), int(flat[b]["y"] * h))
                cv2.line(vis, pa, pb, (0, 255, 0), 1)
        # wrist vs fingertip distance sanity
        try:
            wrist, tip = flat[0], flat[8]
            dist = ((wrist["x"]-tip["x"])**2 + (wrist["y"]-tip["y"])**2)**0.5
            cv2.putText(vis, f"wrist-tip dist {dist:.2f}", (x0, max(12, y0-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)
        except Exception:
            pass

    # Header text
    hand_flag = result.get("hand_flag", 0.0 if result.get("hands_detected", 0)==0 else -1)
    handed = result.get("handedness", ["?"])[0] if result.get("handedness") else "?"
    hs = result.get("handedness_score", -1)
    txt1 = f"flag={hand_flag:.3f} hands={result.get('hands_detected',0)} {handed}({hs:.2f}) {result.get('inference_ms',0):.1f}ms"
    color1 = (0, 255, 0) if result.get("hands_detected", 0) else (0, 0, 255) if "FALSE_POSITIVE" in " ".join(flags) else (200, 200, 200)
    cv2.putText(vis, txt1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, 1, cv2.LINE_AA)

    # Metrics line
    txt2 = f"bbox {metrics['bbox_w']:.2f}x{metrics['bbox_h']:.2f} area={metrics['bbox_area']:.4f} std {metrics['std_x']:.2f},{metrics['std_y']:.2f}"
    cv2.putText(vis, txt2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Flags
    y = 58
    for f in flags:
        is_error = any(k in f for k in ("CLUSTERED", "FALSE_POSITIVE", "Z_OOB", "BORDER_CLAMP"))
        col = (0, 0, 255) if is_error else (0, 255, 255)
        cv2.putText(vis, f"! {f[:90]}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
        y += 14
        if y > h - 10:
            break

    # Expected hint
    if is_no_hand_expected(out_path.name):
        cv2.putText(vis, "expected: NO HAND", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
    elif is_edge_case(out_path.name):
        cv2.putText(vis, "expected: HAND at EDGE (full-frame will degrade)", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 1)

    cv2.imwrite(str(out_path), vis)


def verify_one(path: Path, out_dir: Path, args):
    raw = load_image_bytes(path)
    # landmarks.process_frame expects JPEG bytes; if file is PNG it still works via cv2.imdecode
    t0 = time.perf_counter()
    result = lm.process_frame(raw)
    wall_ms = (time.perf_counter() - t0) * 1000  # not used: result already has inference_ms
    # Load bgr for annotation (separate decode to keep vis faithful)
    nparr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        # Try PIL fallback for weird formats
        from PIL import Image
        import io as bio
        pil = Image.open(path).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    metrics = compute_metrics(result)
    flags = flag_issues(path, result, metrics)

    # Save annotated
    out_path = out_dir / f"annotated_{path.stem}.jpg"
    if bgr is not None:
        annotate_and_save(bgr, result, metrics, flags, out_path)
    else:
        print(f"  WARN: could not decode {path.name} for annotation")

    # Console line
    flag_str = " | ".join(flags) if flags else "OK"
    # Colorize via prefix
    prefix = "⚠️ " if flags else "✅ "
    # Use consistent ordering
    hand_flag = result.get("hand_flag", 0.0 if result.get("hands_detected", 0)==0 else -1)
    hs = result.get("handedness_score", -1)
    print(f"{prefix}{path.name:28} flag={hand_flag:.3f} hands={result.get('hands_detected',0)} "
          f"handed={result.get('handedness', ['-'])[0] if result.get('handedness') else '-'}({hs:.2f}) "
          f"infer={result.get('inference_ms',0):5.1f}ms "
          f"bbox={metrics['bbox_w']:.2f}x{metrics['bbox_h']:.2f} "
          f"=> {flag_str}")

    return {"path": path, "result": result, "metrics": metrics, "flags": flags, "out": out_path}


def find_images(input_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts and not p.name.startswith(".")])
    return files


def run_self_tests():
    """Unit checks for the math that was most likely to be wrong."""
    print("[self-test] decoding + letterbox math")
    # 1) _decode_landmarks: raw 112,112 -> 0.5,0.5
    raw = np.zeros(63, dtype=np.float32)
    raw[0], raw[1], raw[2] = 112.0, 112.0, 0.0  # wrist at center
    decoded = lm._decode_landmarks(raw)
    assert abs(decoded[0]["x"] - 0.5) < 1e-6 and abs(decoded[0]["y"] - 0.5) < 1e-6, f"decode failed {decoded[0]}"
    print("  ✅ _decode_landmarks center 112,112 -> 0.5,0.5")
    # 2) z scaling: raw 89.6 (224*0.4) should give z~1.0
    raw[2] = 89.6
    decoded = lm._decode_landmarks(raw)
    assert abs(decoded[0]["z"] - 1.0) < 1e-6, f"z decode failed {decoded[0]['z']}"
    print("  ✅ _decode_landmarks z 89.6 -> 1.0 (normalize_z 0.4)")
    # 3) letterbox: 640x480 -> scale 224/640=0.35, new 224x168, pad_h 28
    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas, pad, scale, pad_w, pad_h = lm._letterbox(bgr, 224)
    assert abs(scale - 0.35) < 0.01 and pad_h == 28 and pad_w == 0, f"letterbox {scale} {pad_w},{pad_h}"
    print(f"  ✅ _letterbox 640x480 -> scale {scale:.3f} pad {pad_w},{pad_h}")
    # 4) undo: landmark at letterbox center 0.5,0.5 should map to bgr center 0.5,0.5
    # Simulate lm_letterboxed at 0.5,0.5 -> bgr 0.5,0.5
    target, w, h = 224, 640, 480
    lm_lb = {"x": 0.5, "y": 0.5, "z": 0}
    x_bgr = (lm_lb["x"]*target - pad_w) / (w*scale)
    y_bgr = (lm_lb["y"]*target - pad_h) / (h*scale)
    assert abs(x_bgr-0.5) < 1e-6 and abs(y_bgr-0.5) < 1e-6, f"undo {x_bgr},{y_bgr}"
    print("  ✅ letterbox undo center 0.5,0.5 -> bgr 0.5,0.5")
    # 5) clamp: raw at 300 (>224) should clamp to 1.0
    raw[0], raw[1] = 300.0, -10.0
    decoded = lm._decode_landmarks(raw)
    assert decoded[0]["x"] == 1.0 and decoded[0]["y"] == 0.0, f"clamp failed {decoded[0]}"
    print("  ✅ clamp OOB 300,-10 -> 1.0,0.0")
    print("[self-test] all decode tests passed\n")


def main():
    ap = argparse.ArgumentParser(description="Rigorous verify for backend/landmarks.py (full-frame without palm crop)")
    ap.add_argument("--input", "-i", type=str, default="test_hands", help="Folder with test images (10-15 hands)")
    ap.add_argument("--output", "-o", type=str, default="backend/verify_out", help="Output folder for annotated images + report")
    ap.add_argument("--save-all", action="store_true", help="Save even OK images (default saves all anyway)")
    ap.add_argument("--json", type=str, default="", help="Also write JSON report to this path")
    ap.add_argument("--self-test", action="store_true", help="Run internal decode/letterbox unit tests and exit")
    args = ap.parse_args()

    if args.self_test:
        run_self_tests()
        # also run a quick process_frame no-hand check
        print("[self-test] process_frame no-hand black image")
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", black, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        res = lm.process_frame(jpeg.tobytes())
        assert res["hands_detected"] == 0, f"black should be no-hand {res}"
        print(f"  ✅ black no-hand flag={res.get('hand_flag',0):.3f} hands=0")
        print("[self-test] all self-tests passed")
        return

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"[verify] input folder not found: {in_dir}")
        print(f"  Create it and add 10-15 photos, e.g.:")
        print(f"    mkdir -p {in_dir}")
        print(f"    # then copy: centered.jpg, side.jpg, edge.jpg, far.jpg, occluded.jpg, nohand.jpg, etc.")
        print(f"  Filenames with 'nohand'/'empty' are treated as no-hand controls.")
        print(f"  Running synthetic self-test instead...")
        # Synthetic self-test: black, noise, etc. to prove no-hand path is not hallucinating
        # Create 3 synthetic images in out_dir as demo
        synth_dir = out_dir / "_synthetic"
        synth_dir.mkdir(parents=True, exist_ok=True)
        h, w = 480, 640
        cases = []
        # black
        black = np.zeros((h, w, 3), dtype=np.uint8)
        p = synth_dir / "synthetic_nohand_black.jpg"
        cv2.imwrite(str(p), black)
        cases.append(p)
        # noise
        noise = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        p = synth_dir / "synthetic_nohand_noise.jpg"
        cv2.imwrite(str(p), noise)
        cases.append(p)
        # white
        white = np.full((h, w, 3), 255, dtype=np.uint8)
        p = synth_dir / "synthetic_nohand_white.jpg"
        cv2.imwrite(str(p), white)
        cases.append(p)
        in_dir = synth_dir
        print(f"  Synthetic images written to {synth_dir}/ — verify they all show hands=0 and flag<0.5")

    files = find_images(in_dir)
    if not files:
        print(f"[verify] no images found in {in_dir} (jpg/png)")
        sys.exit(1)

    print(f"[verify] landmarks MEDIAPIPE_AVAILABLE={lm.MEDIAPIPE_AVAILABLE}")
    print(f"[verify] {len(files)} images from {in_dir} -> {out_dir}")
    print(f"[verify] NOTE: current pipeline is FULL-FRAME without palm crop — edge/far hands WILL degrade. "
          f"This script flags that as EXPECTED_DEGRADATION so you can see where palm detection is needed.")
    print()

    # Warmup
    try:
        _ = lm.process_frame((in_dir / files[0]).read_bytes())
    except Exception:
        pass

    results = []
    for p in files:
        try:
            r = verify_one(p, out_dir, args)
            results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"❌ {p.name} crashed: {e}")

    # Summary
    print("\n" + "="*78)
    total = len(results)
    detected = sum(1 for r in results if r["result"].get("hands_detected", 0) > 0)
    flagged = sum(1 for r in results if r["flags"])
    false_pos = sum(1 for r in results if any("FALSE_POSITIVE" in f for f in r["flags"]))
    false_neg_center = sum(1 for r in results if any("NO_HAND_DETECTED" in f and "edge" not in f.lower() and "far" not in f.lower() for f in r["flags"]))
    clustered = sum(1 for r in results if any("CLUSTERED" in f for f in r["flags"]))
    near_thresh = sum(1 for r in results if any("NEAR_THRESHOLD" in f for f in r["flags"]))
    avg_infer = float(np.mean([r["result"].get("inference_ms", 0) for r in results])) if results else 0
    max_infer = float(np.max([r["result"].get("inference_ms", 0) for r in results])) if results else 0

    print(f"Summary: {total} images, {detected} with hands_detected=1, {total-detected} with 0")
    print(f"  flagged: {flagged}/{total}  false_pos: {false_pos}  clustered: {clustered}  near_thresh: {near_thresh}  avg_infer {avg_infer:.1f}ms max {max_infer:.1f}ms")
    nohand_total = sum(1 for r in results if is_no_hand_expected(r["path"].name))
    if nohand_total:
        nohand_ok = nohand_total - false_pos
        print(f"  no-hand controls: {nohand_ok}/{nohand_total} correct (expect 100%)")
    edge_total = sum(1 for r in results if is_edge_case(r["path"].name) or is_far_case(r["path"].name))
    if edge_total:
        edge_detected = sum(1 for r in results if (is_edge_case(r["path"].name) or is_far_case(r["path"].name)) and r["result"].get("hands_detected", 0) > 0)
        print(f"  edge/far hands: {edge_detected}/{edge_total} detected on full-frame (low rate = expected without palm crop)")

    # Write report
    report = {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "mediapipe_available": lm.MEDIAPIPE_AVAILABLE,
        "total": total,
        "summary": {
            "detected": detected,
            "flagged": flagged,
            "false_pos": false_pos,
            "clustered": clustered,
            "near_thresh": near_thresh,
            "avg_infer_ms": avg_infer,
            "max_infer_ms": max_infer,
        },
        "per_image": [
            {
                "file": r["path"].name,
                "hand_flag": r["result"].get("hand_flag", None),
                "hands_detected": r["result"].get("hands_detected", 0),
                "handedness": r["result"].get("handedness", []),
                "handedness_score": r["result"].get("handedness_score", None),
                "inference_ms": r["result"].get("inference_ms", 0),
                "bbox_w": r["metrics"]["bbox_w"],
                "bbox_h": r["metrics"]["bbox_h"],
                "bbox_area": r["metrics"]["bbox_area"],
                "std_x": r["metrics"]["std_x"],
                "std_y": r["metrics"]["std_y"],
                "at_border": r["metrics"]["at_border"],
                "z_min": r["metrics"]["z_min"],
                "z_max": r["metrics"]["z_max"],
                "flags": r["flags"],
                "annotated": str(r["out"]),
            }
            for r in results
        ],
    }
    json_path = Path(args.json) if args.json else out_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport JSON: {json_path}")
    print(f"Annotated images: {out_dir}/annotated_*.jpg")

    # Verdict on full-frame without palm crop
    print("\n" + "-"*78)
    if edge_total and edge_detected < edge_total * 0.5:
        print("VERDICT: Full-frame WITHOUT palm crop degrades on edge/far hands (as flagged).")
        print("  → This is EXPECTED. Add palm detection (hand_detector.tflite 192×192) → crop ROI → landmarks.")
        print("  → For ISL, that step is required before classification; otherwise numbers are wrong for off-center hands.")
    if false_pos > 0 or clustered > 0:
        print("VERDICT: Some outputs are obviously wrong (clustered/false-pos) — DO NOT build classifier on these numbers yet.")
    elif flagged == 0:
        print("VERDICT: All checks passed for supplied images (but still test edge/far if not already).")
    else:
        print("VERDICT: Some warnings (often near-threshold/edge) — review annotated images in output folder.")

    # Exit code: 2 if systematic failure
    if false_pos >= 2 or clustered >= 2 or (nohand_total and false_pos / max(1, nohand_total) > 0.3):
        print("\nExit 2: systematic landmark quality issue (full-frame without crop or threshold).")
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
