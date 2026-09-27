"""
TracheoSentry - predict.py

Loads the saved Random Forest model and runs inference on a WAV file,
returning the predicted class and per-class probabilities.

Command-line usage (from project root):
    python ml/predict.py path/to/clip.wav

Programmatic usage (e.g. from dashboard/app.py):
    from ml.predict import load_model, predict_wav
    model_bundle = load_model()
    label, probs = predict_wav(model_bundle, "path/to/clip.wav")

BENCHTOP PROTOTYPE ONLY. Not for human use. Not clinically validated.
This module intentionally does NOT control any hardware - it only returns
a prediction. Actuation decisions live in dashboard/state.py and require
explicit caregiver confirmation.
"""

import os
import sys
import argparse
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features_from_path, extract_features_from_array

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "tracheosentry_rf.pkl")


def load_model(model_path=DEFAULT_MODEL_PATH):
    """
    Load the trained model bundle ({"model": clf, "classes": [...]})
    saved by train.py.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run ml/train.py first."
        )
    bundle = joblib.load(model_path)
    return bundle


def predict_wav(model_bundle, wav_path):
    """
    Run inference on a single WAV file.

    Returns:
        predicted_label (str)
        probabilities (dict: label -> probability)
    """
    clf = model_bundle["model"]
    classes = model_bundle["classes"]

    feature_vector = extract_features_from_path(wav_path)
    feature_vector = feature_vector.reshape(1, -1)

    predicted_label = clf.predict(feature_vector)[0]
    proba = clf.predict_proba(feature_vector)[0]

    # clf.classes_ is the authoritative order for predict_proba columns
    prob_dict = {label: float(p) for label, p in zip(clf.classes_, proba)}

    return predicted_label, prob_dict


def predict_array(model_bundle, audio_array, sr=16000):
    """
    Same as predict_wav but starting from an in-memory audio array.
    Useful later for live streaming inference instead of file-based clips.
    """
    clf = model_bundle["model"]

    feature_vector = extract_features_from_array(audio_array, sr=sr)
    feature_vector = feature_vector.reshape(1, -1)

    predicted_label = clf.predict(feature_vector)[0]
    proba = clf.predict_proba(feature_vector)[0]
    prob_dict = {label: float(p) for label, p in zip(clf.classes_, proba)}

    return predicted_label, prob_dict


def main():
    parser = argparse.ArgumentParser(
        description="TracheoSentry - run Random Forest prediction on a WAV file "
        "(benchtop prototype only, not for human use)."
    )
    parser.add_argument("wav_path", help="Path to a .wav file to classify")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained model .pkl (default: model/tracheosentry_rf.pkl)",
    )
    args = parser.parse_args()

    bundle = load_model(args.model)
    label, probs = predict_wav(bundle, args.wav_path)

    print(f"File: {args.wav_path}")
    print(f"Predicted class: {label}")
    print("Class probabilities:")
    for cls, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:10s}: {p:.3f}")


if __name__ == "__main__":
    main()
