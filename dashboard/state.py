"""
TracheoSentry - state.py

Deterministic safety state machine for the suction-assist workflow.

    NORMAL
      -> POSSIBLE_SECRETION
      -> ALERT
      -> WAIT_CONFIRM
      -> CONFIRMED
      -> SAFETY_CHECK
      -> SUCTION
      -> COOLDOWN
      -> NORMAL

SAFETY RULE (do not weaken this):
    The Random Forest model NEVER directly triggers suction. It only
    contributes detections that feed an N-of-M persistence counter.
    Reaching ALERT requires explicit caregiver action (CONFIRM) before the
    state machine may proceed toward SUCTION. If no confirmation is given,
    the pump stays OFF and the machine returns to a non-suction state.

    Similarly, DISMISS or EMERGENCY_STOP always wins over any AI signal.

BENCHTOP PROTOTYPE ONLY. Not for human use. Not clinically validated.
"""

from enum import Enum
from collections import deque
import time


class State(Enum):
    NORMAL = "NORMAL"
    POSSIBLE_SECRETION = "POSSIBLE_SECRETION"
    ALERT = "ALERT"
    WAIT_CONFIRM = "WAIT_CONFIRM"
    CONFIRMED = "CONFIRMED"
    SAFETY_CHECK = "SAFETY_CHECK"
    SUCTION = "SUCTION"
    COOLDOWN = "COOLDOWN"
    EMERGENCY_STOPPED = "EMERGENCY_STOPPED"


# --- Persistence rule tuning ---
PERSISTENCE_N = 6          # need N positive "mucus" windows...
PERSISTENCE_M = 8          # ...out of the last M windows, to raise an alert

# --- Timing tuning (seconds) ---
SUCTION_DURATION_S = 5.0     # simulated/actual suction duration for this prototype
COOLDOWN_DURATION_S = 10.0   # minimum time before another suction cycle can start

# --- Safety-check gating (placeholder logic for the benchtop prototype) ---
# These are meant to be replaced/expanded as real sensors come online.
MAX_VACUUM_KPA = 20.0        # simple placeholder ceiling for the mechanical relief valve context


class TracheoSentryStateMachine:
    """
    Deterministic, single-threaded state machine. Call `update()` once per
    control loop tick (e.g. once per classifier inference), and call the
    button handlers (`confirm_suction`, `dismiss`, `emergency_stop`) from
    the dashboard UI event handlers.
    """

    def __init__(self, persistence_n=PERSISTENCE_N, persistence_m=PERSISTENCE_M):
        self.state = State.NORMAL
        self.persistence_n = persistence_n
        self.persistence_m = persistence_m
        self.window_history = deque(maxlen=persistence_m)  # True/False per window

        self.last_prediction_label = None
        self.last_prediction_confidence = 0.0

        self._suction_start_time = None
        self._cooldown_start_time = None

        self.log = []  # simple in-memory event log: list of (timestamp, event)
        self._record("INIT", "State machine initialized in NORMAL")

    # ---------------- internal helpers ----------------

    def _record(self, event_type, detail=""):
        self.log.append(
            {
                "timestamp": time.time(),
                "state": self.state.value,
                "event": event_type,
                "detail": detail,
            }
        )

    def _persistence_met(self):
        if len(self.window_history) < self.persistence_m:
            return False
        return sum(self.window_history) >= self.persistence_n

    # ---------------- main update (AI-driven, non-actuating) ----------------

    def update(self, predicted_label, confidence):
        """
        Feed one classifier result into the state machine. This function can
        move the state from NORMAL -> POSSIBLE_SECRETION -> ALERT -> WAIT_CONFIRM,
        but it can NEVER move the state into SUCTION by itself. Reaching
        SUCTION always requires confirm_suction() to have been called.
        """
        self.last_prediction_label = predicted_label
        self.last_prediction_confidence = confidence

        if self.state == State.EMERGENCY_STOPPED:
            # Latched. Requires an explicit reset() call, not AI input.
            return self.state

        is_mucus_window = predicted_label == "mucus"
        self.window_history.append(is_mucus_window)

        if self.state == State.NORMAL:
            if is_mucus_window:
                self.state = State.POSSIBLE_SECRETION
                self._record("TRANSITION", "NORMAL -> POSSIBLE_SECRETION")

        elif self.state == State.POSSIBLE_SECRETION:
            if self._persistence_met():
                # Persistence threshold reached: raise ALERT and immediately
                # arm WAIT_CONFIRM in the same tick so the caregiver-facing
                # buttons go live right away (ALERT is logged as a distinct
                # event for the UI/log, but there is no reason to make the
                # caregiver wait an extra inference cycle to be prompted).
                self.state = State.ALERT
                self._record(
                    "TRANSITION",
                    f"POSSIBLE_SECRETION -> ALERT "
                    f"({sum(self.window_history)}/{len(self.window_history)} windows)",
                )
                self.state = State.WAIT_CONFIRM
                self._record("TRANSITION", "ALERT -> WAIT_CONFIRM (awaiting caregiver)")
            elif len(self.window_history) >= self.persistence_m:
                # Buffer is full (M windows seen) and threshold still not met:
                # drop back to NORMAL rather than waiting indefinitely.
                self.state = State.NORMAL
                self._record("TRANSITION", "POSSIBLE_SECRETION -> NORMAL")
            # else: buffer still filling up toward M windows - stay in
            # POSSIBLE_SECRETION and keep accumulating.

        elif self.state == State.ALERT:
            # Defensive fallback: normally we arm WAIT_CONFIRM directly from
            # POSSIBLE_SECRETION above, but if the machine is ever left in
            # ALERT for another tick, always advance to WAIT_CONFIRM so the
            # caregiver controls stay live rather than the alert going stale.
            self.state = State.WAIT_CONFIRM
            self._record("TRANSITION", "ALERT -> WAIT_CONFIRM (awaiting caregiver)")

        elif self.state == State.WAIT_CONFIRM:
            # Do nothing here on AI input alone - only confirm_suction() or
            # dismiss() may move us out of WAIT_CONFIRM. This is the key
            # safety gate: AI persistence cannot self-approve suction.
            pass

        elif self.state == State.COOLDOWN:
            if self._cooldown_start_time is not None:
                elapsed = time.time() - self._cooldown_start_time
                if elapsed >= COOLDOWN_DURATION_S:
                    self.state = State.NORMAL
                    self.window_history.clear()
                    self._cooldown_start_time = None
                    self._record("TRANSITION", "COOLDOWN -> NORMAL")

        # CONFIRMED, SAFETY_CHECK, SUCTION are advanced explicitly via
        # confirm_suction()/run_safety_check()/finish_suction(), not by update().

        return self.state

    # ---------------- caregiver-triggered actions ----------------

    def confirm_suction(self):
        """
        Caregiver pressed CONFIRM SUCTION. Only valid from WAIT_CONFIRM.
        This is the ONLY path that can lead toward SUCTION.
        """
        if self.state != State.WAIT_CONFIRM:
            self._record(
                "REJECTED",
                f"confirm_suction() ignored - not in WAIT_CONFIRM (state={self.state.value})",
            )
            return self.state

        self.state = State.CONFIRMED
        self._record("TRANSITION", "WAIT_CONFIRM -> CONFIRMED (caregiver confirmed)")

        # Immediately run the safety check gate (kept separate for clarity
        # and so it's easy to expand into real hardware interlock checks).
        self._run_safety_check()
        return self.state

    def _run_safety_check(self):
        """
        Placeholder safety-check gate between CONFIRMED and SUCTION.
        Expand this with real interlocks as hardware comes online, e.g.:
          - hardware watchdog heartbeat OK
          - emergency stop not engaged
          - vacuum gauge within expected range
          - solenoid/MOSFET self-test OK
        For now this always passes through in software, but the state
        step exists so those checks have a clear place to live.
        """
        self.state = State.SAFETY_CHECK
        self._record("TRANSITION", "CONFIRMED -> SAFETY_CHECK")

        safety_ok = True  # placeholder - wire in real interlocks later
        if safety_ok:
            self.state = State.SUCTION
            self._suction_start_time = time.time()
            self._record("TRANSITION", "SAFETY_CHECK -> SUCTION (pump enabled)")
        else:
            self.state = State.NORMAL
            self._record("TRANSITION", "SAFETY_CHECK -> NORMAL (safety check failed)")

    def poll_suction_progress(self):
        """
        Call periodically while in SUCTION to auto-advance to COOLDOWN once
        the fixed suction duration has elapsed. Keeps actuation time-bounded
        even without additional caregiver interaction.
        """
        if self.state == State.SUCTION and self._suction_start_time is not None:
            elapsed = time.time() - self._suction_start_time
            if elapsed >= SUCTION_DURATION_S:
                self.state = State.COOLDOWN
                self._cooldown_start_time = time.time()
                self._suction_start_time = None
                self._record("TRANSITION", "SUCTION -> COOLDOWN")
        return self.state

    def dismiss(self):
        """
        Caregiver pressed DISMISS. Valid from ALERT or WAIT_CONFIRM.
        Cancels the alert without suctioning and resets persistence.
        """
        if self.state in (State.ALERT, State.WAIT_CONFIRM, State.POSSIBLE_SECRETION):
            self.state = State.NORMAL
            self.window_history.clear()
            self._record("TRANSITION", "DISMISS -> NORMAL (caregiver dismissed)")
        else:
            self._record(
                "REJECTED", f"dismiss() ignored - state={self.state.value}"
            )
        return self.state

    def emergency_stop(self):
        """
        Caregiver/operator pressed EMERGENCY STOP. Valid from ANY state.
        This latches into EMERGENCY_STOPPED and must be explicitly reset.
        In real hardware this should ALSO be wired directly to the
        independent hardware watchdog / pump cutoff, not rely on software
        alone.
        """
        self.state = State.EMERGENCY_STOPPED
        self.window_history.clear()
        self._suction_start_time = None
        self._cooldown_start_time = None
        self._record("TRANSITION", "EMERGENCY_STOP -> EMERGENCY_STOPPED (pump forced OFF)")
        return self.state

    def reset_from_emergency(self):
        """Explicit operator action required to leave EMERGENCY_STOPPED."""
        if self.state == State.EMERGENCY_STOPPED:
            self.state = State.NORMAL
            self._record("TRANSITION", "EMERGENCY_STOPPED -> NORMAL (manual reset)")
        return self.state

    # ---------------- status helpers for the dashboard ----------------

    def is_pump_allowed(self):
        """
        Single source of truth for whether the pump may physically run.
        The dashboard/hardware layer should check this (AND the independent
        hardware watchdog) before ever energizing the solenoid/pump.
        """
        return self.state == State.SUCTION

    def persistence_ratio(self):
        return sum(self.window_history), len(self.window_history)

    def get_recent_log(self, n=20):
        return self.log[-n:]
