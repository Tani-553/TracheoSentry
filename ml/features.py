"""
TracheoSentry - features.py

Audio -> feature vector pipeline for the piezo airway-vibration signal.

Pipeline:
    WAV file
      -> load + convert to mono
      -> resample to 16 kHz
      -> remove DC offset
      -> safe normalization
      -> 40-bin log-Mel spectrogram
      -> fixed-length statistical feature vector (mean, std, max per Mel bin)

BENCHTOP PROTOTYPE ONLY. Not for human use. Not clinically validated.
"""

import numpy as np
import librosa

# ---- Fixed pipeline constants (keep train/predict in sync) ----
TARGET_SR = 16000
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160          # 10 ms hop at 16 kHz
WIN_LENGTH = 400          # 25 ms window at 16 kHz
FMIN = 20
FMAX = TARGET_SR // 2

# Feature vector length = N_MELS * 3 (mean, std, max)
FEATURE_LENGTH = N_MELS * 3


def load_audio(path, target_sr=TARGET_SR):
    """
    Load a WAV file, force mono, resample to target_sr.
    Returns a 1D float32 numpy array.
    """
    # librosa.load already converts to mono (mono=True default) and resamples
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32)


def remove_dc_offset(audio):
    """Subtract the mean to remove any DC bias from the piezo/ADC front end."""
    if audio.size == 0:
        return audio
    return audio - np.mean(audio)


def safe_normalize(audio, eps=1e-8):
    """
    Normalize amplitude to roughly [-1, 1] based on peak absolute value,
    guarding against divide-by-zero on silent/near-silent clips.
    """
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak < eps:
        # Silent or near-silent clip: return as-is (already ~zero)
        return audio
    return audio / peak


def preprocess_audio(path):
    """Full preprocessing chain: load -> mono/resample -> DC removal -> normalize."""
    audio = load_audio(path)
    audio = remove_dc_offset(audio)
    audio = safe_normalize(audio)
    return audio


def compute_log_mel_spectrogram(audio, sr=TARGET_SR):
    """
    Compute a 40-bin log-Mel spectrogram (dB scale) from a preprocessed
    1D audio array. Returns array of shape (N_MELS, time_frames).
    """
    if audio.size == 0:
        # Degenerate case: return a single silent frame so downstream
        # stats don't crash on empty input.
        audio = np.zeros(WIN_LENGTH, dtype=np.float32)

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    log_mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_db


def summarize_to_feature_vector(log_mel_db):
    """
    Collapse the (N_MELS, time_frames) log-Mel spectrogram into a fixed-length
    feature vector using per-Mel-bin mean, std, and max across time.

    Returns a 1D numpy array of length FEATURE_LENGTH (= N_MELS * 3).
    """
    mean_vec = np.mean(log_mel_db, axis=1)
    std_vec = np.std(log_mel_db, axis=1)
    max_vec = np.max(log_mel_db, axis=1)

    feature_vector = np.concatenate([mean_vec, std_vec, max_vec]).astype(np.float32)
    return feature_vector


def extract_features_from_path(path):
    """
    Convenience wrapper: WAV file path -> fixed-length feature vector.
    This is the single function both train.py and predict.py should call
    so the feature pipeline never drifts between training and inference.
    """
    audio = preprocess_audio(path)
    log_mel_db = compute_log_mel_spectrogram(audio)
    feature_vector = summarize_to_feature_vector(log_mel_db)
    return feature_vector


def extract_features_from_array(audio, sr=TARGET_SR):
    """
    Same as extract_features_from_path, but starting from an in-memory
    audio array instead of a file path. Useful later for live streaming
    inference from the ESP32 once that path exists.

    NOTE: caller is responsible for ensuring `audio` is mono and at sr.
    DC removal and normalization are still applied here.
    """
    audio = remove_dc_offset(audio)
    audio = safe_normalize(audio)
    log_mel_db = compute_log_mel_spectrogram(audio, sr=sr)
    return summarize_to_feature_vector(log_mel_db)


if __name__ == "__main__":
    # Quick smoke test with a synthetic signal (no WAV file needed).
    # This does NOT represent real airway data - it's just to confirm the
    # pipeline runs end-to-end without crashing.
    print("Running features.py smoke test (synthetic sine wave, NOT real data)...")
    t = np.linspace(0, 1.0, TARGET_SR, endpoint=False)
    synthetic = 0.5 * np.sin(2 * np.pi * 120 * t).astype(np.float32)
    vec = extract_features_from_array(synthetic)
    print(f"Feature vector shape: {vec.shape} (expected {FEATURE_LENGTH})")
    print(f"Sample values: {vec[:5]}")
