"""
sim_backend.py — a MuJoCo-backed stand-in for the SO-101 + Feetech bus, so the
tuning loop in calibrate.py can be smoke-tested on the digital twin (no hardware,
no noise).

Why this exists
---------------
servo_tuning.py / calibrate.py talk to a real FeetechMotorsBus: they read
Present_Position, write Goal_Position, and read/write the servo's P/I/D and
Acceleration registers. MuJoCo has none of those. So SimBus presents the *same
small interface* those scripts use, backed by the SO-101 MuJoCo model.

How the "servo" is modelled
---------------------------
The MuJoCo model already ships a position actuator per joint (kp≈998, kd≈2.7,
force-limited to ±3.35 N·m) — effectively the STS3215's internal position loop
running at the 500 Hz sim rate. We treat that as the servo and map the tunable
registers onto it at runtime:

  P_Coefficient -> actuator kp        (gainprm[0] and biasprm[1] = -kp)
  D_Coefficient -> actuator kv        (biasprm[2] = -kv), on top of joint damping
  I_Coefficient -> an integral torque added via qfrc_applied (no native term)
  Acceleration  -> rate limit on the commanded setpoint (trapezoidal-ish profiling)
  Goal_Position -> actuator ctrl target

A daemon thread advances physics in real time, so pre-settle sleeps and recorded
response times come out in real seconds, just like the hardware path.

Scaling is chosen so the factory-ish P=32/D=32 gives a stable but mildly
under-damped step (visible overshoot the auto-tuner can improve).
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import mujoco

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
from config import SCENE_XML

# Register -> physical-gain scaling (see module docstring).
KP_SCALE = 0.84     # P=32  -> kp ≈ 27  (wn ≈ 25 rad/s ≈ 4 Hz for the ~0.043 kg·m² joints)
KD_SCALE = 0.02     # D=32  -> kv ≈ 0.64 (under-damped baseline; auto-tuner can raise it)
KI_SCALE = 0.30     # integral-term gain
TAU_I_MAX = 1.0     # clamp on the integral torque (N·m)
ACCEL_RATE = 6.0    # deg/s of setpoint ramp per Acceleration unit (0 = profiler off)


class SimBus:
    """Mimics the subset of FeetechMotorsBus used by servo_tuning.py."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self._lock = threading.RLock()

        self.pid = {n: {"P": 32, "I": 0, "D": 32} for n in MOTOR_NAMES}
        self.accel = {n: 254 for n in MOTOR_NAMES}
        self.lock_reg = {n: 1 for n in MOTOR_NAMES}
        self.torque_enable = {n: 1 for n in MOTOR_NAMES}
        self.integral = {n: 0.0 for n in MOTOR_NAMES}
        self._last_tau = {n: 0.0 for n in MOTOR_NAMES}

        self.qadr, self.dofadr, self.actid = {}, {}, {}
        for n in MOTOR_NAMES:
            jid = model.joint(n).id
            self.qadr[n] = model.jnt_qposadr[jid]
            self.dofadr[n] = model.jnt_dofadr[jid]
            self.actid[n] = model.actuator(n).id

        # Initialise goals/setpoints to the current pose and apply baseline gains.
        self.goal = {n: self._q_deg(n) for n in MOTOR_NAMES}
        self.setpoint = dict(self.goal)
        for n in MOTOR_NAMES:
            self.data.ctrl[self.actid[n]] = math.radians(self.goal[n])
            self._apply_gains(n)

        self._running = False
        self._thread = threading.Thread(target=self._physics_loop, daemon=True)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def connect(self):
        self._running = True
        self._t_last = time.perf_counter()
        self._sim_time = 0.0
        self._thread.start()

    def disconnect(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    # ── physics ──────────────────────────────────────────────────────────────
    def _q_deg(self, motor):
        return math.degrees(self.data.qpos[self.qadr[motor]])

    def _apply_gains(self, motor):
        """Push the current P/D coefficients into the MuJoCo position actuator."""
        aid = self.actid[motor]
        kp = KP_SCALE * self.pid[motor]["P"]
        kv = KD_SCALE * self.pid[motor]["D"]
        self.model.actuator_gainprm[aid, 0] = kp
        self.model.actuator_biasprm[aid, 1] = -kp
        self.model.actuator_biasprm[aid, 2] = -kv

    def _physics_loop(self):
        dt = self.model.opt.timestep
        while self._running:
            now = time.perf_counter()
            target_sim = self._sim_time + min(now - self._t_last, 0.05)  # cap catch-up
            self._t_last = now
            with self._lock:
                while self._sim_time < target_sim:
                    self._step(dt)
                    self._sim_time += dt
            time.sleep(0.0005)

    def _step(self, dt):
        for n in MOTOR_NAMES:
            # Setpoint ramp (Acceleration register as a rate limit on the target).
            rate = self.accel[n] * ACCEL_RATE
            if rate <= 0:
                self.setpoint[n] = self.goal[n]
            else:
                d = self.goal[n] - self.setpoint[n]
                step = rate * dt
                self.setpoint[n] = self.goal[n] if abs(d) <= step else self.setpoint[n] + math.copysign(step, d)

            if self.torque_enable[n]:
                self.data.ctrl[self.actid[n]] = math.radians(self.setpoint[n])
                # Integral term (the actuator handles P and D natively).
                err = self.setpoint[n] - self._q_deg(n)
                self.integral[n] = float(np.clip(self.integral[n] + math.radians(err) * dt, -50, 50))
                tau_i = float(np.clip(KI_SCALE * self.pid[n]["I"] * self.integral[n], -TAU_I_MAX, TAU_I_MAX))
                self.data.qfrc_applied[self.dofadr[n]] = tau_i
                self._last_tau[n] = self.data.actuator_force[self.actid[n]] if self.data.actuator_force.size else 0.0
            else:
                self.data.ctrl[self.actid[n]] = 0.0
                self.data.qfrc_applied[self.dofadr[n]] = 0.0
        mujoco.mj_step(self.model, self.data)

    # ── bus interface (subset) ───────────────────────────────────────────────
    def read(self, name, motor, *, normalize=True, num_retry=0):
        with self._lock:
            if name == "Present_Position":
                return self._q_deg(motor)                       # degrees (use_degrees)
            if name == "P_Coefficient":
                return self.pid[motor]["P"]
            if name == "I_Coefficient":
                return self.pid[motor]["I"]
            if name == "D_Coefficient":
                return self.pid[motor]["D"]
            if name == "Acceleration":
                return self.accel[motor]
            if name == "Lock":
                return self.lock_reg[motor]
            if name == "Torque_Enable":
                return self.torque_enable[motor]
            if name == "Present_Voltage":
                return 74                                       # /10 -> 7.4 V
            if name == "Present_Current":
                return abs(self._last_tau[motor]) / 6.5 / 0.001  # ~mA-ish, decoded *6.5
            if name == "Present_Load":
                pct = float(np.clip(self._last_tau[motor] / 3.35 * 100, -100, 100))
                raw = int(abs(pct) * 10) | (0x400 if pct < 0 else 0)
                return raw
        raise KeyError(f"SimBus has no register '{name}'")

    def write(self, name, motor, value, *, normalize=True, num_retry=0):
        with self._lock:
            if name == "P_Coefficient":
                self.pid[motor]["P"] = int(value); self._apply_gains(motor)
            elif name == "I_Coefficient":
                self.pid[motor]["I"] = int(value)
            elif name == "D_Coefficient":
                self.pid[motor]["D"] = int(value); self._apply_gains(motor)
            elif name == "Acceleration":
                self.accel[motor] = int(value)
            elif name == "Goal_Position":
                self.goal[motor] = float(value)
            elif name == "Lock":
                self.lock_reg[motor] = int(value)
            elif name == "Torque_Enable":
                self.torque_enable[motor] = int(value)
            # silently ignore config registers we don't model (Return_Delay_Time, ...)

    def sync_read(self, name, motors=None, *, normalize=True, num_retry=0):
        if motors is None:
            motors = MOTOR_NAMES
        if isinstance(motors, str):
            motors = [motors]
        return {m: self.read(name, m, normalize=normalize) for m in motors}

    def sync_write(self, name, values, *, normalize=True, num_retry=0):
        if not isinstance(values, dict):
            values = {m: values for m in MOTOR_NAMES}
        with self._lock:
            for m, v in values.items():
                self.write(name, m, v, normalize=normalize)

    def disable_torque(self, motors=None, num_retry=0):
        for m in (MOTOR_NAMES if motors is None else ([motors] if isinstance(motors, str) else motors)):
            self.torque_enable[m] = 0
            self.lock_reg[m] = 0

    def enable_torque(self, motors=None, num_retry=0):
        for m in (MOTOR_NAMES if motors is None else ([motors] if isinstance(motors, str) else motors)):
            self.torque_enable[m] = 1
            self.lock_reg[m] = 1


class SimRobot:
    """Mimics the subset of SOFollower used by calibrate.py, backed by MuJoCo."""

    def __init__(self, scene_xml: str = SCENE_XML):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.bus = SimBus(self.model, self.data)

    def connect(self):
        self.bus.connect()
        print("  [SIM] MuJoCo digital twin connected (no hardware).")

    def disconnect(self):
        self.bus.disconnect()

    def get_observation(self):
        return {f"{n}.pos": self.bus.read("Present_Position", n) for n in MOTOR_NAMES}

    def send_action(self, action):
        goal = {k.removesuffix(".pos"): v for k, v in action.items() if k.endswith(".pos")}
        self.bus.sync_write("Goal_Position", goal)
        return action
