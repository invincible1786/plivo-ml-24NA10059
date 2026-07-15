import os
import csv
import sys
import pickle
import glob
import numpy as np

# Dynamically add the starter code path to import features.py
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "starter", "starter")
sys.path.append(src_dir)
from features import load_wav, extract_unified_features

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    # Locate labels.csv files recursively
    labels_pattern = os.path.join(args.data_dir, "**", "labels.csv")
    labels_files = glob.glob(labels_pattern, recursive=True)
    
    if not labels_files:
        sys.exit(f"Error: No labels.csv found in {args.data_dir}")

    # Load calibrated model
    model_name = "model_calibrated.pkl"
    model_paths = [
        model_name,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name),
        os.path.join(src_dir, model_name)
    ]
    model_clf = None
    for p in model_paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                model_clf = pickle.load(f)
            break

    if model_clf is None:
        sys.exit(f"Error: Could not find model file {model_name} in paths: {model_paths}")

    all_predictions = []

    # Process each labels file found
    for lf in labels_files:
        rows = []
        with open(lf) as f:
            for r in csv.DictReader(f):
                # Read ONLY target metadata columns. Ignore 'label' and 'pause_end'.
                rows.append({
                    "turn_id": r["turn_id"],
                    "audio_file": r["audio_file"],
                    "pause_index": int(r["pause_index"]),
                    "pause_start": float(r["pause_start"])
                })

        if not rows:
            continue

        # Group by turn_id and sort chronologically by pause_index
        turns = {}
        for r in rows:
            tid = r["turn_id"]
            if tid not in turns:
                turns[tid] = []
            turns[tid].append(r)

        for tid in turns:
            turns[tid].sort(key=lambda x: x["pause_index"])

        cache = {}
        preds_map = {}
        data_parent = os.path.dirname(lf)

        for tid, pz_list in turns.items():
            path = os.path.join(data_parent, pz_list[0]["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]

            prev_pauses_starts = []
            for r in pz_list:
                feats = extract_unified_features(x, sr, r, prev_pauses_starts)
                p_val = model_clf.predict_proba(np.array([feats]))[0, 1]
                preds_map[(r["turn_id"], r["pause_index"])] = p_val
                
                prev_pauses_starts.append(r["pause_start"])

        for r in rows:
            p_eot = preds_map.get((r["turn_id"], r["pause_index"]), 0.5)
            all_predictions.append((r["turn_id"], r["pause_index"], p_eot))

    # Write out predictions to a single CSV file
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for tid, pi, p_eot in all_predictions:
            w.writerow([tid, pi, f"{p_eot:.4f}"])

    print(f"Wrote {len(all_predictions)} predictions -> {args.out}")

if __name__ == "__main__":
    main()
