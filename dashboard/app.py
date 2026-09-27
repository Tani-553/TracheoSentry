"""
TracheoSentry - dashboard/app.py

Streamlit dashboard for the benchtop TracheoSentry prototype.

Run from the project root:
    streamlit run dashboard/app.py

BENCHTOP PROTOTYPE ONLY. Not for human use. Not clinically validated.

This dashboard is deliberately AI-cannot-actuate: the Random Forest model
only ever produces a prediction + confidence, which feeds the persistence
counter in state.py. Suction only occurs if the caregiver explicitly
presses CONFIRM SUCTION while the state machine is in WAIT_CONFIRM.
"""

import os
import sys
import time

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# --- path setup so `ml` and local `state` both import cleanly ---
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DASHBOARD_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ml"))
sys.path.insert(0, DASHBOARD_DIR)

from state import TracheoSentryStateMachine, State  # noqa: E402
from ml.predict import load_model, predict_wav, DEFAULT_MODEL_PATH  # noqa: E402

st.set_page_config(
    page_title="TracheoSentry",
    page_icon="\U0001FA7A",
    layout="wide",
)

# ---------------- styling ----------------

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .ts-banner {
        background-color: #7a1f1f;
        color: white;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
    }
    .ts-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #e6edf3;
        margin-bottom: 0;
    }
    .ts-subtitle {
        font-size: 1.0rem;
        color: #8b949e;
        margin-top: 0;
        margin-bottom: 1.2rem;
    }
    .ts-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .ts-card-label {
        color: #8b949e;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .ts-card-value {
        color: #e6edf3;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .ts-state-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- state machine + model (persisted across reruns) ----------------

if "sm" not in st.session_state:
    st.session_state.sm = TracheoSentryStateMachine()

if "model_bundle" not in st.session_state:
    try:
        st.session_state.model_bundle = load_model()
        st.session_state.model_error = None
    except FileNotFoundError as e:
        st.session_state.model_bundle = None
        st.session_state.model_error = str(e)

if "last_waveform" not in st.session_state:
    st.session_state.last_waveform = None
if "last_probs" not in st.session_state:
    st.session_state.last_probs = {}
if "last_label" not in st.session_state:
    st.session_state.last_label = "—"
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0
if "vacuum_level" not in st.session_state:
    st.session_state.vacuum_level = 0.0
if "imu_status" not in st.session_state:
    st.session_state.imu_status = "Stationary"

sm = st.session_state.sm

STATE_COLORS = {
    State.NORMAL: ("#1f6f3f", "white"),
    State.POSSIBLE_SECRETION: ("#8a6d1f", "white"),
    State.ALERT: ("#a83232", "white"),
    State.WAIT_CONFIRM: ("#a83232", "white"),
    State.CONFIRMED: ("#a86a32", "white"),
    State.SAFETY_CHECK: ("#a86a32", "white"),
    State.SUCTION: ("#1f4e8a", "white"),
    State.COOLDOWN: ("#3a3f47", "white"),
    State.EMERGENCY_STOPPED: ("#000000", "#ff4444"),
}

# ---------------- header ----------------

st.markdown(
    '<div class="ts-banner">BENCHTOP PROTOTYPE &mdash; NOT FOR HUMAN USE '
    '&mdash; NOT CLINICALLY VALIDATED</div>',
    unsafe_allow_html=True,
)

col_title, col_state = st.columns([3, 1])
with col_title:
    st.markdown('<div class="ts-title">TracheoSentry</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ts-subtitle">Intelligent Tracheostomy Monitoring '
        '&mdash; human-in-the-loop suction assistance (research prototype)</div>',
        unsafe_allow_html=True,
    )
with col_state:
    bg, fg = STATE_COLORS.get(sm.state, ("#333", "white"))
    st.markdown(
        f'<div class="ts-state-pill" style="background-color:{bg}; color:{fg};">'
        f'SYSTEM STATE: {sm.state.value}</div>',
        unsafe_allow_html=True,
    )

if st.session_state.model_bundle is None:
    st.warning(
        f"No trained model found at `{DEFAULT_MODEL_PATH}`. "
        f"Run `python ml/train.py` first (with at least placeholder/test WAV "
        f"data in each data/<class>/ folder) so this dashboard has a model to load."
    )

st.divider()

# ---------------- input: simulate a sensing window ----------------

st.subheader("Airway Vibration Input (piezo signal window)")
input_col1, input_col2 = st.columns([2, 1])

with input_col2:
    uploaded = st.file_uploader(
        "Upload a WAV clip (test data)", type=["wav"], key="wav_upload"
    )
    run_clicked = st.button("Run inference on this window", use_container_width=True)
    st.caption(
        "For the first milestone, feed placeholder/test WAV clips here to "
        "verify the pipeline. This is not a substitute for real piezo hardware "
        "data once available."
    )

if run_clicked and uploaded is not None and st.session_state.model_bundle is not None:
    tmp_path = os.path.join(PROJECT_ROOT, "recordings", "_dashboard_tmp.wav")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    label, probs = predict_wav(st.session_state.model_bundle, tmp_path)
    confidence = probs.get(label, 0.0)

    st.session_state.last_label = label
    st.session_state.last_confidence = confidence
    st.session_state.last_probs = probs

    # Load waveform just for display purposes
    try:
        import soundfile as sf

        wav_data, _ = sf.read(tmp_path)
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
        st.session_state.last_waveform = wav_data
    except Exception:
        st.session_state.last_waveform = None

    # Feed the state machine (AI signal only - cannot actuate by itself)
    sm.update(label, confidence)

with input_col1:
    if st.session_state.last_waveform is not None:
        fig, ax = plt.subplots(figsize=(7, 2.2))
        ax.plot(st.session_state.last_waveform, linewidth=0.6, color="#58a6ff")
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.set_xlabel("Sample", color="#8b949e")
        ax.set_ylabel("Amplitude", color="#8b949e")
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Upload and run a WAV clip to see the waveform here.")

st.divider()

# ---------------- status cards ----------------

st.subheader("Status")

n_hits, m_total = sm.persistence_ratio()

card1, card2, card3, card4 = st.columns(4)
with card1:
    st.markdown(
        f'<div class="ts-card"><div class="ts-card-label">Airway Status</div>'
        f'<div class="ts-card-value">{sm.state.value}</div></div>',
        unsafe_allow_html=True,
    )
with card2:
    st.markdown(
        f'<div class="ts-card"><div class="ts-card-label">Secretion Probability</div>'
        f'<div class="ts-card-value">{st.session_state.last_probs.get("mucus", 0.0)*100:.1f}%</div></div>',
        unsafe_allow_html=True,
    )
with card3:
    st.markdown(
        f'<div class="ts-card"><div class="ts-card-label">RF Prediction</div>'
        f'<div class="ts-card-value">{st.session_state.last_label}</div></div>',
        unsafe_allow_html=True,
    )
with card4:
    st.markdown(
        f'<div class="ts-card"><div class="ts-card-label">Confidence</div>'
        f'<div class="ts-card-value">{st.session_state.last_confidence*100:.1f}%</div></div>',
        unsafe_allow_html=True,
    )

card5, card6, card7, card8 = st.columns(4)
with card5:
    st.markdown(
        f'<div class="ts-card"><div class="ts-card-label">Persistence Count</div>'
        f'<div class="ts-card-value">{n_hits} / {m_total}</div></div>',
        unsafe_allow_html=True,
    )
with card6:
    st.markdown(
        f'<div class="ts-card"><div class="ts-card-label">IMU Status</div>'
        f'<div class="ts-card-value">{st.session_state.imu_status}</div></div>',
        unsafe_allow_html=True,
    )
with card7:
    st.markdown(
        f'<div class="ts-card"><div class="ts-card-label">Vacuum Level</div>'
        f'<div class="ts-card-value">{st.session_state.vacuum_level:.1f} kPa</div></div>',
        unsafe_allow_html=True,
    )
with card8:
    pump_state = "ON" if sm.is_pump_allowed() else "OFF"
    pump_color = "#1f6f3f" if pump_state == "ON" else "#8b949e"
    st.markdown(
        f'<div class="ts-card"><div class="ts-card-label">Pump</div>'
        f'<div class="ts-card-value" style="color:{pump_color};">{pump_state}</div></div>',
        unsafe_allow_html=True,
    )

if st.session_state.last_probs:
    with st.expander("Full class probabilities"):
        for cls, p in sorted(st.session_state.last_probs.items(), key=lambda kv: -kv[1]):
            st.write(f"{cls}: {p*100:.1f}%")
            st.progress(min(max(p, 0.0), 1.0))

st.divider()

# ---------------- caregiver controls ----------------

st.subheader("Caregiver Controls")
st.caption(
    "The AI prediction alone can never trigger suction. Suction only occurs "
    "after an explicit CONFIRM SUCTION press while the system is awaiting "
    "confirmation, followed by an automated safety check."
)

btn1, btn2, btn3 = st.columns(3)

with btn1:
    confirm_disabled = sm.state != State.WAIT_CONFIRM
    if st.button(
        "✅ CONFIRM SUCTION",
        disabled=confirm_disabled,
        use_container_width=True,
        type="primary",
    ):
        sm.confirm_suction()
        st.session_state.vacuum_level = 12.5  # placeholder simulated reading
        st.rerun()

with btn2:
    dismiss_disabled = sm.state not in (
        State.ALERT,
        State.WAIT_CONFIRM,
        State.POSSIBLE_SECRETION,
    )
    if st.button("DISMISS", disabled=dismiss_disabled, use_container_width=True):
        sm.dismiss()
        st.rerun()

with btn3:
    if st.button("🛑 EMERGENCY STOP", use_container_width=True):
        sm.emergency_stop()
        st.session_state.vacuum_level = 0.0
        st.rerun()

if sm.state == State.EMERGENCY_STOPPED:
    st.error(
        "EMERGENCY STOP engaged. Pump is forced OFF. "
        "Requires manual reset by an operator."
    )
    if st.button("Reset from Emergency Stop (operator action)"):
        sm.reset_from_emergency()
        st.rerun()

if sm.state == State.SUCTION:
    sm.poll_suction_progress()
    st.info("Suction cycle active (simulated/benchtop). Will auto-advance to COOLDOWN.")
    time.sleep(0.3)
    st.rerun()
elif sm.state == State.COOLDOWN:
    sm.update(st.session_state.last_label, st.session_state.last_confidence)
    time.sleep(0.3)
    st.rerun()

st.divider()

# ---------------- event log ----------------

st.subheader("Event Log")
recent = sm.get_recent_log(15)
if recent:
    for entry in reversed(recent):
        ts_str = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
        st.text(f"[{ts_str}] {entry['state']:<18s} {entry['event']:<12s} {entry['detail']}")
else:
    st.caption("No events yet.")

st.divider()
st.caption(
    "TracheoSentry benchtop prototype. All predictions, states, and controls "
    "above operate on a simulated airway and synthetic/test data unless real "
    "piezo recordings have been substituted. Not for human use."
)
