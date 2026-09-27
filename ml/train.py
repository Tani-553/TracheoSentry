"""
TracheoSentry - train.py

Scans data/<class>/*.wav, extracts features via features.py, trains a
RandomForestClassifier, prints evaluation metrics, and saves the model.

Run from the project root:
    python ml/train.py

BENCHTOP PROTOTYPE ONLY. Not for human use. Not clinically validated.
No human patient recordings should ever be placed in data/.

*** DATA LEAKAGE WARNING (READ THIS) ***
This script currently does a RANDOM train/test split over individual audio
chunks. That is fine for smoke-testing the pipeline with placeholder/test
WAV files, but it is NOT valid for real evaluation.

For the REAL dataset, you must split by RECORDING RUN (i.e. by session /
by recording file group), not randomly by individual clip. If chunks from
the same recording run end up in both train and test, the classifier can
learn to recognize the *run* (background noise, mic placement, sensor
drift) instead of the actual class, and your reported accuracy will be
falsely inflated. Group your filenames by run ID (e.g. a prefix like
"run003_mucus_001.wav") and use sklearn.model_selection.GroupShuffleSplit
or GroupKFold keyed on that run ID before trusting any reported numbers.
"""

import os
import sys
import glob
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Make sure "ml" package imports work when run as `python ml/train.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features_from_path, FEATURE_LENGTH

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "tracheosentry_rf.pkl")

CLASSES = ["normal", "mucus", "cough", "motion"]

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
)


def scan_class_folders(data_dir=DATA_DIR, classes=CLASSES):
    """
    Walk data/<class>/*.wav for each class and return a list of
    (filepath, label) pairs.
    """
    samples = []
    for label in classes:
        class_dir = os.path.join(data_dir, label)
        wav_paths = sorted(glob.glob(os.path.join(class_dir, "*.wav")))
        if not wav_paths:
            print(f"  [warn] no .wav files found in {class_dir}")
        for wav_path in wav_paths:
            samples.append((wav_path, label))
    return samples


def build_feature_matrix(samples):
    """
    Extract features for every (path, label) pair.
    Returns X (n_samples, FEATURE_LENGTH) and y (n_samples,).
    Skips files that fail to load, with a warning.
    """
    X = []
    y = []
    for path, label in samples:
        try:
            vec = extract_features_from_path(path)
            if vec.shape[0] != FEATURE_LENGTH:
                print(f"  [warn] unexpected feature length for {path}, skipping")
                continue
            X.append(vec)
            y.append(label)
        except Exception as e:
            print(f"  [warn] failed to process {path}: {e}")
    if not X:
        return np.empty((0, FEATURE_LENGTH), dtype=np.float32), np.array([])
    return np.vstack(X), np.array(y)


def main():
    print("=" * 60)
    print("TracheoSentry - Random Forest training")
    print("BENCHTOP PROTOTYPE ONLY - not clinically validated")
    print("=" * 60)

    print(f"\nScanning class folders under: {DATA_DIR}")
    samples = scan_class_folders()
    print(f"Found {len(samples)} total WAV files across {len(CLASSES)} classes.")

    if len(samples) == 0:
        print(
            "\nNo WAV files found. Place placeholder/test .wav files in "
            "data/normal, data/mucus, data/cough, data/motion and re-run.\n"
            "Remember: synthetic/test data is for pipeline verification only, "
            "not for treating the trained model as clinically meaningful."
        )
        return

    print("\nExtracting features...")
    X, y = build_feature_matrix(samples)
    print(f"Feature matrix: X={X.shape}, y={y.shape}")

    if len(set(y)) < 2:
        print(
            "\n[error] Need at least 2 classes with data to train a classifier. "
            "Add WAV files to more class folders."
        )
        return

    # NOTE: random split - fine for a placeholder/smoke-test dataset only.
    # See the data leakage warning in the module docstring for the real dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"\nTraining RandomForestClassifier with params: {RF_PARAMS}")
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n--- Classification Report (test split) ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("--- Confusion Matrix (rows=true, cols=pred) ---")
    labels_sorted = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    print("Labels order:", labels_sorted)
    print(cm)

    print(
        "\n[reminder] These numbers are only meaningful once real recordings "
        "replace placeholder/test data, AND once train/test splitting is done "
        "by recording run rather than randomly by chunk. Do not report these "
        "as validated clinical performance."
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": clf, "classes": labels_sorted}, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
