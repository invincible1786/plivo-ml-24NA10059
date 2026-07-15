import os
import csv
import numpy as np
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier

from features import load_wav, extract_unified_features

def main():
    base_dir = "../../eot_data/eot_data"
    languages = ["english", "hindi"]
    
    features_by_lang = {lang: [] for lang in languages}
    labels_by_lang = {lang: [] for lang in languages}
    cache = {}
    
    for lang in languages:
        data_dir = os.path.join(base_dir, lang)
        labels_path = os.path.join(data_dir, "labels.csv")
        print(f"Extracting features for {lang} from {labels_path}...")
        
        rows = list(csv.DictReader(open(labels_path)))
        
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
            
            prev_pauses = []
            for r in pz_list:
                feats = extract_unified_features(x, sr, r, prev_pauses)
                features_by_lang[lang].append(feats)
                labels_by_lang[lang].append(1 if r["label"] == "eot" else 0)
                
                prev_pauses.append({
                    "pause_start": float(r["pause_start"]),
                    "pause_end": float(r["pause_end"]),
                    "label": r["label"]
                })
                
    X_en = np.array(features_by_lang["english"])
    y_en = np.array(labels_by_lang["english"])
    
    X_hi = np.array(features_by_lang["hindi"])
    y_hi = np.array(labels_by_lang["hindi"])
    
    # Combined dataset (English + Hindi)
    X_comb = np.vstack([X_en, X_hi])
    y_comb = np.concatenate([y_en, y_hi])
    
    print(f"English features: {X_en.shape}, Hindi features: {X_hi.shape}")
    print(f"Combined features: {X_comb.shape}")
    
    # 1. Train English model (on Combined dataset with best hyperparams)
    print("Training English model (on combined dataset)...")
    clf_en = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_depth=4,
        l2_regularization=1.0,
        min_samples_leaf=20,
        random_state=42
    )
    clf_en.fit(X_comb, y_comb)
    
    # 2. Train Hindi model (on Hindi dataset with best hyperparams)
    print("Training Hindi model (on Hindi-only dataset)...")
    clf_hi = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.03,
        max_depth=2,
        l2_regularization=0.0,
        min_samples_leaf=10,
        random_state=42
    )
    clf_hi.fit(X_hi, y_hi)
    
    # Save models
    with open("model_english.pkl", "wb") as f:
        pickle.dump(clf_en, f)
    with open("model_hindi.pkl", "wb") as f:
        pickle.dump(clf_hi, f)
        
    print("Saved model_english.pkl and model_hindi.pkl successfully.")

if __name__ == "__main__":
    main()
