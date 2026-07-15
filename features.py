"""Audio utilities and causal feature extraction for the EOT assignment.

This module implements the 24-feature set:
- 4 Conversational/State features (pause_index, cumulative_speech_duration, is_first_pause, slope_of_previous_pauses)
- 8 Acoustic features (mean and std of RMS, Centroid, Flatness, and Flatness/RMS ratio)
- 12 Mel-Frequency Cepstral Coefficients (mean and std of the first 6 MFCCs)

All feature extractions use center=False to completely prevent look-ahead window bias.
"""
import numpy as np
import soundfile as sf
import librosa

FRAME_MS = 25
HOP_MS = 10


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def detect_speech_resume(x, sr, pause_start, threshold_db=-45.0):
    """Causally detects when speech resumes after pause_start.
    
    Analyses the audio after pause_start to find the first frame where energy
    exceeds threshold_db. This is causal since it is only called on completed
    historical pauses in the turn.
    """
    start_idx = int(pause_start * sr)
    # Search up to 3 seconds of audio after the pause_start
    x_segment = x[start_idx : start_idx + int(3.0 * sr)]
    if len(x_segment) == 0:
        return pause_start + 0.15
        
    frame_len = int(sr * 0.02)
    hop_len = int(sr * 0.01)
    if len(x_segment) < frame_len:
        return pause_start + 0.15
        
    n_frames = 1 + (len(x_segment) - frame_len) // hop_len
    energies = []
    for i in range(n_frames):
        frame = x_segment[i*hop_len : i*hop_len + frame_len]
        energies.append(np.sqrt(np.mean(frame**2) + 1e-12))
        
    energies = 20 * np.log10(np.array(energies) + 1e-12)
    speech_frames = np.where(energies > threshold_db)[0]
    if len(speech_frames) > 0:
        return pause_start + speech_frames[0] * 0.01
    return pause_start + 0.15


def get_previous_pauses_slope(x, sr, prev_pauses_starts):
    """Calculates the slope of previous completed pause durations.
    
    Falls back to a global training-set statistic (0.0) if fewer than 2 pauses.
    """
    K = len(prev_pauses_starts)
    if K < 2:
        return 0.0
    durs = []
    for p_start in prev_pauses_starts:
        p_end = detect_speech_resume(x, sr, p_start)
        durs.append(p_end - p_start)
    slope, _ = np.polyfit(np.arange(K), durs, 1)
    return slope


def extract_unified_features(x, sr, r, prev_pauses_starts):
    """Extracts the 24-feature set from the causal audio up to pause_start.
    
    All librosa feature extractions set center=False.
    """
    pause_start = float(r["pause_start"])
    pause_index = int(r["pause_index"])
    
    # 1. Slice audio strictly causally
    x_causal = x[0 : int(pause_start * sr)]
    
    # 2. Conversational State Features (4 features)
    is_first_pause = 1.0 if pause_index == 0 else 0.0
    
    # Cumulative active speech duration up to pause_start
    prev_pauses_dur = 0.0
    for p_start in prev_pauses_starts:
        p_end = detect_speech_resume(x, sr, p_start)
        prev_pauses_dur += (p_end - p_start)
    cumulative_speech_duration = max(0.0, pause_start - prev_pauses_dur)
    
    slope_of_previous_pauses = get_previous_pauses_slope(x, sr, prev_pauses_starts)
    
    state_feats = [
        pause_index,
        cumulative_speech_duration,
        is_first_pause,
        slope_of_previous_pauses
    ]
    
    # If the segment is too short, return state features + NaNs for acoustic features
    if len(x_causal) < sr // 10:
        return np.array(state_feats + [np.nan] * 20, dtype=np.float32)
        
    # Analyze the last 1.5s of speech preceding the pause
    seg = x_causal[-int(1.5 * sr):]
    
    # 3. Acoustic features (8 features)
    # Compute RMS energy (center=False)
    rms = librosa.feature.rms(y=seg, center=False)[0]
    rms_mean = rms.mean()
    rms_std = rms.std()
    
    # Spectral Centroid (center=False)
    centroid = librosa.feature.spectral_centroid(y=seg, sr=sr, center=False)[0]
    centroid_mean = centroid.mean()
    centroid_std = centroid.std()
    
    # Spectral Flatness (center=False)
    flatness = librosa.feature.spectral_flatness(y=seg, center=False)[0]
    flatness_mean = flatness.mean()
    flatness_std = flatness.std()
    
    # Fricative vs. Breath discriminator: ratio = flatness / (rms + 1e-7)
    ratio = flatness / (rms + 1e-7)
    ratio_mean = ratio.mean()
    ratio_std = ratio.std()
    
    acoustic_feats = [
        rms_mean, rms_std,
        centroid_mean, centroid_std,
        flatness_mean, flatness_std,
        ratio_mean, ratio_std
    ]
    
    # 4. MFCC Features (12 features)
    # Compute first 6 MFCCs (center=False)
    mfccs = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=6, center=False)
    mfcc_mean = mfccs.mean(axis=1)
    mfcc_std = mfccs.std(axis=1)
    
    mfcc_feats = list(mfcc_mean) + list(mfcc_std)
    
    # Combine all to make exactly 24 features
    feats = state_feats + acoustic_feats + mfcc_feats
    return np.array(feats, dtype=np.float32)


def run_causality_test():
    """Unit test function: asserts center=False prevents look-ahead bias."""
    sr = 16000
    audio = np.random.randn(sr * 3).astype(np.float32)
    p_start = 1.5
    
    # Slice 1: up to p_start
    y_short = audio[:int(p_start * sr)]
    # Slice 2: up to p_start + 0.05
    y_long = audio[:int((p_start + 0.05) * sr)]
    
    # Extract features at frame-level using center=False
    rms_short = librosa.feature.rms(y=y_short, center=False)
    rms_long = librosa.feature.rms(y=y_long, center=False)
    
    centroid_short = librosa.feature.spectral_centroid(y=y_short, sr=sr, center=False)
    centroid_long = librosa.feature.spectral_centroid(y=y_long, sr=sr, center=False)
    
    N = rms_short.shape[1]
    
    # Assert that the first N frames are mathematically identical
    assert np.allclose(rms_short[:, :N], rms_long[:, :N]), "Causality check failed for RMS!"
    assert np.allclose(centroid_short[:, :N], centroid_long[:, :N]), "Causality check failed for Spectral Centroid!"
    print("Built-in causality unit test passed successfully (center=False is causal).")


if __name__ == "__main__":
    run_causality_test()
