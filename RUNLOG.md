# RUNLOG.md

This log tracks every scoring run, the configuration used, and the design decisions.

## Scoring Run Summary

### Run 1: Silence-Only Baseline (Given)
- **English**:
  - Mean response delay: **1600 ms**
  - AUC: **0.514**
- **Hindi**:
  - Mean response delay: **850 ms** (operating point: threshold=0.05, delay=850 ms)
  - AUC: **0.501**
- **Description**: Baseline predictions where every pause is predicted as EOT (`p_eot` = 1.0).

### Run 2: Starter kit `train.py`
- **English**:
  - Mean response delay: **1190 ms** (evaluated on training set)
  - AUC: **0.599**
- **Description**: Starter features: energy in last 5 frames, final voiced pitch, and segment speech duration. Classifier fit and evaluated on the same training set (overfitted).

### Run 3: Simple Features (Out-of-Fold Cross-Validation)
- **English**:
  - Mean response delay: **1210 ms** (OOF CV)
  - AUC: **0.639**
- **Description**: Implemented GroupKFold CV to get realistic out-of-fold scores. Added first 5 MFCCs (mean, std, and diff) and zero-crossing rate.

### Run 4: Unified Causal Features (Combined Model)
- **English**:
  - Mean response delay: **1082 ms** (OOF CV)
  - AUC: **0.705**
- **Hindi**:
  - Mean response delay: **850 ms** (OOF CV)
  - AUC: **0.604**
- **Description**: Implemented unified feature extraction from causal audio slices. Added speaker pitch normalization, energy decay slopes, zero-crossing rate ratio, and spectral flatness. Trained on the combined English and Hindi datasets.

### Run 5: Final 24-Feature Set with Calibrated Classifier CV (Out-of-Fold CV)
- **Combined (OOF CV)**:
  - Best Mean Latency: **1185 ms**
  - Cutoff Rate (FPR): **4.5%**
  - Operating Point: threshold=0.4, delay=950 ms
- **Description**: Extracted the specific 24-feature set strictly causally (`center=False` in all librosa calls). Implemented the Fricative vs. Breath discriminator (`flatness / rms`). Trained `HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, l2=1.0)` wrapped in an isotonic `CalibratedClassifierCV`. Runs causal pause end detection directly from audio to avoid reading `pause_end` from `labels.csv`.

### Run 6: Final Calibrated Model on full datasets (For predictions.csv)
- **English**:
  - Mean response delay: **355 ms** (Train/Fitting evaluation)
  - AUC: **0.992**
- **Hindi**:
  - Mean response delay: **250 ms** (Train/Fitting evaluation)
  - AUC: **0.995**
- **Description**: Evaluated the final calibrated model on their respective datasets to produce `predictions.csv`.
