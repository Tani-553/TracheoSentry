# TracheoSentry (Benchtop Prototype)

> **BENCHTOP PROTOTYPE ONLY. NOT FOR HUMAN USE. NOT CLINICALLY VALIDATED.**
> This is a hackathon software prototype built and tested on a simulated
> airway with artificial mucus (water/glycerine/thickener). It has not been
> tested on any human, animal, or clinical sample, and none of the numbers
> it produces should be treated as clinical performance.

## What the system does

TracheoSentry is a **human-in-the-loop** suction-assistance prototype for
spontaneously breathing, non-ventilated tracheostomy patients (benchtop
simulation only, for now).

Core workflow:

```
Sense -> Detect -> Decide -> Alert -> Confirm -> Actuate -> Log
```

A piezo contact microphone next to the simulated airway captures vibration.
A Random Forest classifier trained on log-Mel spectrogram statistics
classifies short windows into `normal`, `mucus`, `cough`, or `motion`. A
persistence rule (N-of-M positive windows) raises an alert. **The AI never
controls the pump directly.** A caregiver must explicitly press
**CONFIRM SUCTION** before a safety check gate and (eventually) the actual
hardware suction cycle can run. If there is no confirmation, the pump stays
off.

## Folder structure

```
TracheoSentry/
├── data/
│   ├── normal/       # WAV clips, one subfolder per class
│   ├── mucus/
│   ├── cough/
│   └── motion/
├── recordings/        # scratch space for ad-hoc / uploaded clips
├── ml/
│   ├── features.py     # audio -> feature vector pipeline
│   ├── train.py         # trains and saves the Random Forest
│   └── predict.py       # loads the model and runs inference
├── model/
│   └── tracheosentry_rf.pkl   # produced by train.py
├── dashboard/
│   ├── app.py       # Streamlit dashboard
│   └── state.py     # safety state machine
├── requirements.txt
└── README.md
```

## Installation

From the project root:

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to collect data

Place `.wav` files into the matching class folder:

```
data/normal/*.wav
data/mucus/*.wav
data/cough/*.wav
data/motion/*.wav
```

For the **first milestone**, since real piezo hardware recordings are not
yet available, you can use placeholder/synthetic `.wav` clips (e.g. short
tones, noise bursts, or any test audio) just to verify the pipeline runs
end-to-end. **Clearly treat these as test data, not as a real training
set** — a model trained only on synthetic placeholders tells you nothing
about real secretion detection, only that the code works.

When real recordings arrive, name files so you can group them by
**recording run**, e.g.:

```
data/mucus/run003_mucus_001.wav
data/mucus/run003_mucus_002.wav
data/mucus/run004_mucus_001.wav
```

## How to train the Random Forest

```bash
python ml/train.py
```

This scans all four `data/<class>/` folders, extracts features with
`ml/features.py`, trains a `RandomForestClassifier` (`n_estimators=200`,
`max_depth=15`, `class_weight="balanced"`, `random_state=42`), prints a
classification report and confusion matrix, and saves the model to
`model/tracheosentry_rf.pkl`.

### ⚠️ Avoiding data leakage (important for the real dataset)

`train.py` currently does a **random** train/test split over individual
audio chunks. That's acceptable only for smoke-testing the pipeline with
placeholder data. **For the real dataset, split by recording run, not
randomly by chunk.** If windows from the same recording session land in
both the train and test sets, the model can learn to recognize
session-specific artifacts (background noise, mic placement, sensor
drift) instead of the actual airway sound class — this inflates reported
accuracy in a way that will not hold up on new recordings.

To fix this once you have multiple runs per class, group filenames by a
run ID prefix and use `sklearn.model_selection.GroupShuffleSplit` or
`GroupKFold` keyed on that run ID instead of the current random
`train_test_split`.

## How to run prediction

Command line, on a single WAV file:

```bash
python ml/predict.py recordings/some_clip.wav
```

Programmatically (e.g. from the dashboard):

```python
from ml.predict import load_model, predict_wav

bundle = load_model()
label, probs = predict_wav(bundle, "recordings/some_clip.wav")
```

`predict.py` intentionally has no hardware control code in it — it only
returns a label and probabilities. It is written to be reusable later for
live streaming inference once ESP32 -> laptop streaming exists.

## How to start the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

The dashboard lets you upload a test WAV clip, run inference, and watch
the safety state machine respond. It shows airway status, secretion
probability, the waveform, RF prediction + confidence, persistence count,
IMU status (placeholder for now), vacuum level (placeholder), and the
current system state. It exposes three caregiver controls: **CONFIRM
SUCTION**, **DISMISS**, and **EMERGENCY STOP**.

## How the safety state machine works

`dashboard/state.py` implements:

```
NORMAL
  -> POSSIBLE_SECRETION
  -> ALERT
  -> WAIT_CONFIRM
  -> CONFIRMED
  -> SAFETY_CHECK
  -> SUCTION
  -> COOLDOWN
  -> NORMAL
```

- Every classifier result is pushed into a rolling window of the last
  **M = 8** windows. If **N = 6** of the last 8 windows were classified as
  `mucus`, the machine moves from `POSSIBLE_SECRETION` to `ALERT`, then to
  `WAIT_CONFIRM`.
- **`WAIT_CONFIRM` is a hard gate.** The classifier's `update()` call
  cannot move the state past `WAIT_CONFIRM` under any circumstances. Only
  a caregiver pressing **CONFIRM SUCTION** (`confirm_suction()`) can do
  that.
- After confirmation, the machine passes through `CONFIRMED` and
  `SAFETY_CHECK` (a placeholder gate meant to be expanded with real
  hardware interlocks — watchdog heartbeat, emergency-stop status, vacuum
  gauge range, etc.) before reaching `SUCTION`.
- `is_pump_allowed()` is the single source of truth for whether the pump
  may run, and it is only `True` while `state == SUCTION`.
- **`EMERGENCY STOP`** can be pressed from any state and immediately
  latches the machine into `EMERGENCY_STOPPED`, forcing the pump off. It
  requires an explicit manual reset to leave that state. On real hardware
  this should also be wired directly into the independent hardware
  watchdog / pump cutoff, not rely on software alone.
- **`DISMISS`** cancels an alert (valid from `POSSIBLE_SECRETION`,
  `ALERT`, or `WAIT_CONFIRM`) and returns to `NORMAL` without suctioning.

This mirrors the required architecture: **AI detection never directly
actuates the pump.** It only ever produces a signal that can, at most,
raise an alert for a human to act on.

## Benchtop prototype status

- Tested only on a simulated airway with artificial mucus
  (water/glycerine/thickener).
- No human, animal, or clinical recordings have been used or should be
  used with this codebase.
- No clinical performance claims are made or should be inferred from any
  printed metrics — those numbers only describe how well the model
  separates whatever WAV clips you fed it.
- Pressure/vacuum sensing in this milestone assumes a physical vacuum
  gauge on a side branch of the T-junction; no analog pressure sensor
  (e.g. MPX5010DP) is used or wired in software.
- Hardware actuation (solenoid/pump control) is out of scope for this
  software milestone — `state.py` exposes `is_pump_allowed()` as the
  single point where a future hardware driver should check before
  energizing anything, alongside the independent hardware watchdog.
