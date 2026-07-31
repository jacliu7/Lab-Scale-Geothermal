#!/usr/bin/env python3
"""
geo_common.py

Shared hardware I/O and thermal/power math for the geothermal cooling rig.
Pulled out of benchmark.py so that benchmark.py (continuous AI-load runs)
and step_load_test.py (step-response / system-ID runs) both read sensors
and compute COP/PUE the same way, instead of two copies drifting apart.

Nothing in here is specific to a test protocol. Protocol logic (when to
step the load, when to call a trial "stable", what to log) lives in the
scripts that import this module.

CHANGE LOG (this file)
-----------------------
- Added PowerSampler: a background thread that polls get_pi_power_w() at
  a much higher rate than the main state-machine loop runs at, and hands
  back a time-weighted average power over an arbitrary interval. The main
  loop in step_load_test.py / bursty_test.py used to take a single
  instantaneous power sample once per iteration (gated by inference-call
  duration / --poll-interval), which under-resolves short power bursts
  during LLM inference relative to the die's fast (~6s) thermal time
  constant. A time-weighted average over the actual elapsed interval is
  both a better estimate of Q(t) and the physically correct quantity to
  feed the RC model (which needs average power over dt, not one sample
  of it).
- Removed the fluid-side energy balance (flow_rate * cp * (T_in - T_out))
  as the source of Q. Without a reliable coolant inlet temperature, that
  balance can't be computed anyway, and it was always noisier and more
  assumption-laden (fixed flow-vs-duty curve, distilled-water cp/density)
  than the number already being logged for PUE. Q is now taken directly
  from measured CPU/SoC electrical power (compute_q_cpu_w), since nearly
  all electrical power dissipated by the SoC converts to heat. Ambient
  temperature is still measured and still used for delta_T fitting; only
  the inlet/outlet coolant temperatures and the flow-based heat balance
  are gone. See compute_q_cpu_w() below.
"""

import subprocess
import os
import glob
import json
import time
import threading

try:
    from gpiozero import PWMOutputDevice
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


# Constants (from the build spec)

MAX_PUMP_FLOW_LPH = 100.0      # L/hr at 100% duty, per pump spec sheet -- kept for
                                # pump-status telemetry only; no longer used in any
                                # heat/Q calculation.
PUMP_VOLTAGE_NOMINAL = 5.0     # V, mid of the 3-5V spec range
PUMP_CURRENT_MIN_A = 0.10      # A, 100 mA
PUMP_CURRENT_MAX_A = 0.20      # A, 200 mA
GROUND_TEMP_F = 55.0
MAX_SAFE_TEMP_F = 185.0
MAX_SAFE_TEMP_C = (MAX_SAFE_TEMP_F - 32.0) * 5.0 / 9.0   # ~85.0 C, matches the Pi's own throttle point
PUMP_PWM_PIN = 18              # BCM numbering, change to match wiring
PUMP_PWM_FREQ_HZ = 1000


# System metric helpers

# Sysfs paths for temperature/clock. A plain file read here costs on the
# order of tens of microseconds; spawning `vcgencmd` costs on the order of
# 100-300ms per call (process fork/exec overhead), worse under CPU load.
# With three vcgencmd calls per logged row (temp, clock, throttle), that
# alone was adding ~0.5-1s of fixed overhead to every loop iteration in
# step_load_test.py, regardless of --poll-interval/--fast-poll-interval --
# confirmed directly from a trial CSV where --fast-poll-interval=0.2 was
# requested but actual row spacing right after the step was ~1.6-1.8s,
# same as the un-accelerated idle baseline. Shrinking the requested sleep/
# burst duration had no effect because the subprocess spawns, not the
# sleep, were the bottleneck. Reading temp and clock from sysfs removes
# two of the three subprocess calls per row and should let
# --fast-poll-interval actually be achieved.
_THERMAL_ZONE_PATH = "/sys/class/thermal/thermal_zone0/temp"
_SCALING_FREQ_PATH = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"


def get_cpu_temp():
    """Read CPU temperature. Tries sysfs first (fast, no subprocess spawn);
    falls back to vcgencmd if the sysfs path isn't present (e.g. non-Pi
    hardware, different kernel thermal zone numbering). Returns float in
    Celsius."""
    try:
        with open(_THERMAL_ZONE_PATH, "r") as f:
            milli_c = int(f.read().strip())
        return milli_c / 1000.0
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True, text=True, timeout=2
        )
        return float(result.stdout.strip().split("=")[1].replace("'C", ""))
    except Exception:
        return None


def get_cpu_clock():
    """Read ARM CPU clock speed in MHz. Tries sysfs first (fast, no
    subprocess spawn); falls back to vcgencmd if unavailable."""
    try:
        with open(_SCALING_FREQ_PATH, "r") as f:
            khz = int(f.read().strip())
        return khz // 1000
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_clock", "arm"],
            capture_output=True, text=True, timeout=2
        )
        hz = int(result.stdout.strip().split("=")[1])
        return hz // 1_000_000
    except Exception:
        return None


def get_throttle_status():
    """
    Read throttle flags from vcgencmd.
    Bit 0: currently under-voltage
    Bit 1: currently frequency capped
    Bit 2: currently throttled
    Bit 3: currently soft temperature limit active
    """
    try:
        result = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True, text=True, timeout=2
        )
        hex_val = int(result.stdout.strip().split("=")[1], 16)
        return {
            "raw_hex": hex(hex_val),
            "under_voltage": bool(hex_val & 0x1),
            "freq_capped": bool(hex_val & 0x2),
            "throttled": bool(hex_val & 0x4),
            "soft_temp_limit": bool(hex_val & 0x8),
        }
    except Exception:
        return {
            "raw_hex": "error",
            "under_voltage": False,
            "freq_capped": False,
            "throttled": False,
            "soft_temp_limit": False,
        }


def list_available_temp_sensors():
    """List 1-wire DS18B20 sensor IDs currently visible to the Pi."""
    return [os.path.basename(p) for p in glob.glob("/sys/bus/w1/devices/28*")]


def read_ds18b20(sensor_id):
    """Read a single DS18B20 1-wire temp sensor by its device folder name.
    Generic on purpose: this is what the ambient-room sensor uses. (It's
    also fine to point this at a coolant sensor if you want to log loop
    temperature for reference, just note it's no longer used in any Q
    calculation -- see compute_q_cpu_w.)"""
    if not sensor_id:
        return None
    path = f"/sys/bus/w1/devices/{sensor_id}/w1_slave"
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        if lines[0].strip()[-3:] != "YES":
            return None
        temp_str = lines[1].split("t=")[-1]
        return float(temp_str) / 1000.0
    except Exception:
        return None


def get_pi_power_w():
    """
    Estimate total Pi power draw in watts from the PMIC (Pi 5 only), by
    summing voltage * current across every rail reported by vcgencmd.
    Returns None on boards where pmic_read_adc isn't supported.
    """
    try:
        result = subprocess.run(
            ["vcgencmd", "pmic_read_adc"],
            capture_output=True, text=True, timeout=2
        )
        volts, amps = {}, {}
        for line in result.stdout.strip().split("\n"):
            if "=" not in line:
                continue
            label, value = line.split("=")
            label = label.strip().split(" ")[0]  # e.g. "VDD_CORE_V" or "VDD_CORE_A"
            value = float(value.strip().rstrip("VA"))
            if label.endswith("_V"):
                volts[label[:-2]] = value
            elif label.endswith("_A"):
                amps[label[:-2]] = value
        total_w = sum(v * amps[rail] for rail, v in volts.items() if rail in amps)
        return round(total_w, 2) if total_w > 0 else None
    except Exception:
        return None


class PowerSampler:
    """
    Background thread that polls get_pi_power_w() at a much higher rate
    than the main test loop runs at, and hands back the time-weighted
    average power over an arbitrary interval via mean_since().

    Why this exists: the main state-machine loop in step_load_test.py /
    bursty_test.py takes one power reading per iteration, and the
    iteration cadence is set by inference-call duration and
    --poll-interval (often ~1-2s). The die's fast thermal time constant
    (a few seconds, per the two-exponential fan/geothermal fits) can
    respond to power fluctuations well within that window, so a single
    instantaneous sample per iteration under-resolves real Q(t) -- it's
    effectively an undersampled, possibly-aliased view of the true power
    trace. Continuously sampling in the background and time-weight-
    averaging over the interval the caller actually cares about (the gap
    since their last call) is both a better Q(t) estimate and the
    physically correct input to an RC model, which needs average power
    over dt, not one instantaneous point.

    Note: this only helps up to whatever rate the underlying hardware/
    vcgencmd path can actually refresh at. If `pmic_read_adc`'s reported
    values are themselves stale across consecutive fast calls (check this
    before relying on a very small --interval-s), the PMIC's own ADC
    refresh is the bottleneck and no amount of extra polling here will
    resolve sub-refresh-interval bursts -- that needs a dedicated external
    sensor (e.g. INA219/INA260 on the core rail) instead.
    """

    def __init__(self, interval_s=0.02, history_s=5.0):
        self.interval_s = interval_s
        self.history_s = history_s
        self._buf = []  # list of (timestamp, power_w), oldest first
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            p = get_pi_power_w()
            now = time.time()
            if p is not None:
                with self._lock:
                    self._buf.append((now, p))
                    cutoff = now - self.history_s
                    while self._buf and self._buf[0][0] < cutoff:
                        self._buf.pop(0)
            time.sleep(self.interval_s)

    def mean_since(self, since_ts):
        """
        Time-weighted mean power over samples with timestamp >= since_ts.
        Falls back to the single latest sample if fewer than two points
        fall in that range (e.g. interval shorter than the sampling
        period, or sampler just started).
        """
        with self._lock:
            pts = [(t, p) for t, p in self._buf if t >= since_ts]
        if not pts:
            return None
        if len(pts) < 2:
            return pts[-1][1]
        total, weight = 0.0, 0.0
        for i in range(1, len(pts)):
            dt = pts[i][0] - pts[i - 1][0]
            avg = (pts[i][1] + pts[i - 1][1]) / 2.0
            total += avg * dt
            weight += dt
        return (total / weight) if weight > 0 else pts[-1][1]

    def latest(self):
        with self._lock:
            return self._buf[-1][1] if self._buf else None

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


# Pump control (PWM)

_pump = None

def init_pump(duty_pct=0):
    """Start PWM on the pump control pin. Returns False if gpiozero isn't
    available (e.g. running this off-Pi for a dry run), in which case pump
    values are still logged but nothing is actually actuated."""
    global _pump
    if not GPIO_AVAILABLE:
        return False
    duty_pct = max(0, min(100, duty_pct))
    _pump = PWMOutputDevice(PUMP_PWM_PIN, frequency=PUMP_PWM_FREQ_HZ,
                             initial_value=duty_pct / 100.0)
    return True


def set_pump_speed(duty_pct):
    """Change pump PWM duty cycle (0-100) on the fly, e.g. from a dashboard slider."""
    global _pump
    duty_pct = max(0, min(100, duty_pct))
    if _pump is not None:
        _pump.value = duty_pct / 100.0
    return duty_pct


def stop_pump():
    global _pump
    if _pump is not None:
        _pump.off()
        _pump.close()


def read_pump_command(command_path, last_mtime):
    """
    Check whether the dashboard has dropped a new pump duty command since we
    last looked. The dashboard writes {"pump_duty_pct": <int>} to this file
    whenever the user moves the slider. Returns (new_duty_or_None, mtime).
    Using mtime avoids re-applying the same command every loop iteration.
    """
    try:
        mtime = os.path.getmtime(command_path)
        if mtime == last_mtime:
            return None, last_mtime
        with open(command_path, "r") as f:
            data = json.load(f)
        return data.get("pump_duty_pct"), mtime
    except Exception:
        return None, last_mtime


# Loop thermal / power math

def duty_to_flow_rate_lph(duty_pct):
    """Approximate linear map from PWM duty cycle to flow rate, scaled off the
    pump's rated max flow. Kept for pump-status telemetry (what speed was the
    pump actually running at during this condition) -- no longer feeds into
    any heat/Q calculation, since Q is now derived from electrical power
    (see compute_q_cpu_w), not a fluid-side energy balance."""
    return MAX_PUMP_FLOW_LPH * (duty_pct / 100.0)


def compute_q_cpu_w(pi_power_w):
    """
    Heat generated by the CPU/SoC (Q_cpu), estimated directly from measured
    electrical power draw rather than a fluid-side energy balance.

    This replaces the old flow_rate * cp * (T_in - T_out) computation. That
    approach needed a reliable coolant inlet temperature, which isn't
    available here, and even when it was, it carried more assumptions
    (linear duty-to-flow curve, distilled-water cp/density) and more sensor
    noise than the power number already being logged for PUE.

    Nearly all electrical power dissipated by the SoC converts to heat.
    Some fraction radiates or convects off the board/PCB traces before
    ever reaching the cold plate, so this is a slight overestimate of heat
    actually captured by the coolant -- but for a Pi 5 that fraction is
    small, and it's a fixed, easily-acknowledged bias rather than a
    per-sample source of noise. Returns watts, or None if power wasn't
    measured for this sample.
    """
    if pi_power_w is None:
        return None
    return round(pi_power_w, 3)


def compute_pump_power_w(voltage=PUMP_VOLTAGE_NOMINAL, current_a=None):
    """Pump electrical power, P = V * I. Falls back to the midpoint of the
    spec sheet current range (100-200 mA) if you haven't measured it directly."""
    if current_a is None:
        current_a = (PUMP_CURRENT_MIN_A + PUMP_CURRENT_MAX_A) / 2.0
    return round(voltage * current_a, 3)


def compute_cop(q_cpu_w, work_input_w):
    """COP = heat generated by the CPU (Q_cpu, from electrical power) /
    work input (pump power). Higher means more heat handled per watt spent
    running the pump. Note this is now a power-based Q, not a measured
    fluid-side heat-removed number -- see compute_q_cpu_w for why."""
    if not q_cpu_w or not work_input_w:
        return None
    return round(q_cpu_w / work_input_w, 3)


def compute_pue(pi_power_w, pump_power_w):
    """PUE = (Pi power + pump power) / Pi power. 1.0 is the ideal floor, no cooling overhead."""
    if not pi_power_w:
        return None
    pump_power_w = pump_power_w or 0.0
    return round((pi_power_w + pump_power_w) / pi_power_w, 3)


def lock_cpu_clock():
    """Best-effort pin of the CPU governor to 'performance' so clock speed
    holds constant across conditions instead of drifting with thermal load.
    Needs root; silently no-ops if it can't write to sysfs."""
    try:
        paths = glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor")
        for p in paths:
            with open(p, "w") as f:
                f.write("performance")
        print(f"CPU governor set to performance on {len(paths)} core(s).")
        return True
    except PermissionError:
        print("Could not set CPU governor (needs sudo). Clock speed may vary during the run.")
        return False
    except Exception as e:
        print(f"Could not lock CPU clock: {e}")
        return False


def write_live_status(path, status):
    """Write the latest metrics snapshot to a JSON file a dashboard can poll.
    Writes to a temp file then swaps it in so the dashboard never reads a
    half-written file mid-update."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(status, f)
    os.replace(tmp_path, path)