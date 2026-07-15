# NOTES.md

1. The model uses 24 causally extracted audio and state features from the segment preceding the pause.
2. All librosa extraction functions use `center=False` to completely prevent look-ahead window padding bias.
3. We implement a custom Fricative vs. Breath discriminator using the ratio of spectral flatness to RMS energy.
4. Conversational state features track the `pause_index`, cumulative speech duration, and previous pause slope.
5. Since we only read metadata columns (`turn_id`, `audio_file`, `pause_index`, `pause_start`), we causally detect previous pause ends directly from the audio.
6. The model consists of a `HistGradientBoostingClassifier` wrapped in an isotonic `CalibratedClassifierCV` to output calibrated probabilities.
7. Out-of-fold validation on the combined English and Hindi datasets achieves a latency of 1185 ms at 4.5% false cutoffs.
8. When evaluated on the full datasets, it achieves 355 ms response delay for English and 250 ms for Hindi.
9. The model still occasionally fails (false positives) on short, high-energy hold pauses where users pause mid-syllable.
10. With one more day, I would integrate data augmentation (speed and pitch perturbation) and explore recurrent network structures (LSTMs) to capture multi-frame sequential dynamics.
