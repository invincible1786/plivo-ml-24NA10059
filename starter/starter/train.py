"""Rigorous training pipeline for EOT prediction.

Trains a HistGradientBoostingClassifier wrapped in CalibratedClassifierCV.
Evaluates using a custom sequential validation loop simulating a live agent.
"""
import os
import csv
import pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold

from features import load_wav, extract_unified_features

TIMEOUT_S = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 3)
DELAYS = np.round(np.arange(0.10, 1.65, 0.05), 3)


def run_agent_simulation(pauses_by_turn, oof_preds_by_turn, threshold, delay):
    """Mimics a live voice agent: iterates chronologically and stops at first false cutoff."""
    turns_cut = 0
    latencies = []
    
    for tid in pauses_by_turn:
        pz_list = pauses_by_turn[tid]
        preds = oof_preds_by_turn[tid]
        
        for pz, p in zip(pz_list, preds):
            fires = p >= threshold
            dur = float(pz["pause_end"]) - float(pz["pause_start"])
            
            if pz["label"] == "hold":
                if fires and delay < dur:
                    turns_cut += 1
                    break  # Stop immediately at first false cutoff!
            else:  # true end-of-turn
                latencies.append(delay if fires else TIMEOUT_S)
                
    cutoff_rate = turns_cut / len(pauses_by_turn)
    mean_latency = np.mean(latencies) if latencies else TIMEOUT_S
    return cutoff_rate, mean_latency


def main():
    base_dir = "../../eot_data/eot_data"
    languages = ["english", "hindi"]
    
    X, y, groups, keys, raw_rows = [], [], [], [], []
    cache = {}
    
    for lang in languages:
        data_dir = os.path.join(base_dir, lang)
        labels_path = os.path.join(data_dir, "labels.csv")
        print(f"Loading and extracting features for {lang} from {labels_path}...")
        
        rows = list(csv.DictReader(open(labels_path)))
        
        # Group by turn_id to process chronologically
        turns = {}
        for r in rows:
            tid = r["turn_id"]
            if tid not in turns:
                turns[tid] = []
            turns[tid].append(r)
            
        for tid in turns:
            turns[tid].sort(key=lambda x: int(x["pause_index"]))
            
        for tid, pz_list in turns.items():
            path = os.path.join(data_dir, pz_list[0]["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            
            prev_pauses_starts = []
            for r in pz_list:
                global_tid = f"{lang}_{r['turn_id']}"
                feats = extract_unified_features(x, sr, r, prev_pauses_starts)
                X.append(feats)
                y.append(1 if r["label"] == "eot" else 0)
                groups.append(global_tid)
                keys.append((lang, r["turn_id"], int(r["pause_index"])))
                raw_rows.append(r)
                
                prev_pauses_starts.append(float(r["pause_start"]))
                
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    
    print(f"Features shape: {X.shape}, labels: {y.shape}")
    
    # 5-fold GroupKFold CV
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y))
    
    print("Running GroupKFold Cross-Validation...")
    for tr, te in gkf.split(X, y, groups):
        base_clf = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.08,
            max_iter=200,
            l2_regularization=1.0,
            class_weight='balanced',
            random_state=42
        )
        clf = CalibratedClassifierCV(estimator=base_clf, method='isotonic', cv=3)
        clf.fit(X[tr], y[tr])
        oof_preds[te] = clf.predict_proba(X[te])[:, 1]
        
    # Group predictions by global turn_id to run sequential simulation
    pauses_by_turn = {}
    oof_preds_by_turn = {}
    
    for i, (lang, tid, pi) in enumerate(keys):
        global_tid = f"{lang}_{tid}"
        if global_tid not in pauses_by_turn:
            pauses_by_turn[global_tid] = []
            oof_preds_by_turn[global_tid] = []
        pauses_by_turn[global_tid].append(raw_rows[i])
        oof_preds_by_turn[global_tid].append(oof_preds[i])
        
    # Sort chronologically for sequential simulation
    for global_tid in pauses_by_turn:
        zipped = list(zip(pauses_by_turn[global_tid], oof_preds_by_turn[global_tid]))
        zipped.sort(key=lambda item: int(item[0]["pause_index"]))
        pauses_by_turn[global_tid] = [item[0] for item in zipped]
        oof_preds_by_turn[global_tid] = [item[1] for item in zipped]
        
    # Sweep threshold and delay using the sequential simulation
    best = None
    for t in THRESHOLDS:
        for d in DELAYS:
            cut, lat = run_agent_simulation(pauses_by_turn, oof_preds_by_turn, t, d)
            if cut <= 0.05 and (best is None or lat < best["latency"]):
                best = {"latency": lat, "cutoff": cut, "threshold": t, "delay": d}
                
    if best is None:
        best = {"latency": TIMEOUT_S, "cutoff": 0.0, "threshold": 1.0, "delay": TIMEOUT_S}
        
    print("\n--- Out-of-Fold Agent Simulation Results (Combined) ---")
    print(f"Best Mean Latency : {best['latency']*1000:.0f} ms")
    print(f"Cutoff Rate (FPR) : {best['cutoff']*100:.1f}%")
    print(f"Operating Point   : threshold={best['threshold']}, delay={best['delay']*1000:.0f} ms")
    
    # Train final model on entire dataset
    print("\nTraining final model on full combined dataset...")
    base_clf_final = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=200,
        l2_regularization=1.0,
        class_weight='balanced',
        random_state=42
    )
    clf_final = CalibratedClassifierCV(estimator=base_clf_final, method='isotonic', cv=3)
    clf_final.fit(X, y)
    
    # Save the model
    with open("model_calibrated.pkl", "wb") as f:
        pickle.dump(clf_final, f)
    print("Saved final calibrated model to model_calibrated.pkl.")


if __name__ == "__main__":
    main()
