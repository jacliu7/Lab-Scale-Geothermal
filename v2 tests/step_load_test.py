#!/usr/bin/env python3
"""
step_load_test.py

Step Load Test (the calibration run) for the geothermal cooling rig.
Combines the sensor/pump/inference plumbing from benchmark.py with the
first-order (and two-exponential) fitting from thermal_system_id.py into
a single script that runs one trial end to end:

    idle (pre)  ->  step up (full inference load)  ->  hold  ->
    step down (idle)  ->  idle (post)

and fits tau / R_th / C_th on the step-up and step-down transitions as
soon as the trial finishes, instead of requiring a separate offline pass
over a saved CSV.

Protocol, per the test plan
----------------------------
- Idle until CPU temp stabilizes (dT/dt < --stabilize-slope, default
  0.05 C/min, sustained over a --stabilize-window, default 120 s), then
  instantly start the LLM inference workload at full load and hold until
  the same stabilization criterion is met again, then stop the load and
  idle back down to steady state again.
- Ambient room temperature is logged continuously throughout, and for
  the geothermal condition the coolant inlet temperature is logged too
  (both optional DS18B20 sensors -- if you don't have an ambient sensor
  wired up, delta-T fitting falls back to absolute temperature, see
  thermal_system_id.py).
- Fits are done on delta_T = T_die - T_ambient, not absolute temperature,
  so ambient drift across trials doesn't bleed into the R_th estimate.
- The no_cooling and fan conditions likely reduce to a single-node RC
  model. The geothermal loop is a multi-node chain (die -> heatsink ->
  coolant -> loop -> reservoir) and may need a two-exponential fit.
  thermal_system_id.py fits both and picks whichever is actually
  justified by the residuals -- this script doesn't force one model onto
  a condition ahead of time.
- Run this script once per trial. Randomize condition order across
  trials and across days yourself (--trial and --run-order-note just
  tag the output for bookkeeping; they don't do the randomization for
  you, since that has to happen at the scheduling level, not inside a
  single run).

Usage
-----
  python3 step_load_test.py --model tinyllama.gguf --condition geothermal \
      --trial 1 --output ./trial_data \
      --ambient-sensor 28-0000ambient1 \
      --t-in-sensor 28-0000abcd1111 --t-out-sensor 28-0000abcd2222 \
      --pump-duty 100 --lock-clock

  # Dry run without hardware (useful for testing the state machine/fit
  # pipeline off-Pi): temps/power will be logged blank or simulated.
  python3 step_load_test.py --condition no_cooling --trial 1 \
      --output ./trial_data --simulate

Safety
------
Same MAX_SAFE_TEMP_C ceiling as benchmark.py (geo_common.py). If it's
hit at any point in any phase, the run stops immediately and whatever
was captured up to that point is still fit and saved -- a safety stop
during "hold" still gives you a partial step-up transition, which is
useful information even though the trial as a whole should be flagged
and probably re-run.

CHANGE LOG (this file)
-----------------------
- pi_power is now read via geo_common.PowerSampler, a background thread
  that samples get_pi_power_w() at a much higher rate than the main loop
  iterates, and pi_power for each logged row is the time-weighted average
  power over the interval since the previous row (mean_since), not a
  single instantaneous snapshot. This gives Q_ss (and therefore R_th/C_th)
  a much better-resolved power signal to fit against, since a single
  ~1-2s-cadence sample can miss real power fluctuations that happen within
  the die's ~6s fast time constant. --power-sample-interval controls the
  background sampler's polling period; tune it based on how fast
  `vcgencmd pmic_read_adc` actually refreshes on your hardware (see the
  latency/staleness check described alongside PowerSampler in
  geo_common.py).
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np

import geo_common as gc
import thermal_system_id as tsid

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

STRESS_PROMPT = (
    "Explain in detail how geothermal heat exchange systems work, "
    "including the thermodynamic principles, the role of the ground loop, "
    "heat pumps, and why underground temperatures remain stable year-round. "
    "Include comparisons with conventional cooling systems."
)

PHASES = ["idle_pre", "step_up", "step_down", "idle_post"]

CSV_HEADERS = [
    "timestamp", "time_s", "phase", "load_state", "cooling_type", "trial",
    "cpu_temp_c", "ambient_c", "delta_t_die_ambient_c",
    "t_in_c", "t_out_c",
    "cpu_clock_mhz", "throttled",
    "pump_duty_pct", "flow_rate_lph", "heat_removed_w",
    "pump_power_w", "pi_power_w", "power_w", "cop", "pue",
    "tokens_per_sec", "time_to_first_token_sec",
    "stabilization_slope_c_per_min",
]


class SlopeTracker:
    """
    Rolling window of (t, temp) samples used to decide whether the rig
    has reached steady state. Slope is estimated by ordinary least
    squares over the trailing --stabilize-window seconds, reported in
    C/min. This is deliberately a regression over the whole window
    rather than a two-point difference, so a single noisy sensor read
    doesn't trigger a false "stable" call.
    """

    def __init__(self, window_s):
        self.window_s = window_s
        self.buf = deque()  # (t, temp)

    def add(self, t, temp):
        if temp is None:
            return
        self.buf.append((t, temp))
        cutoff = t - self.window_s
        while self.buf and self.buf[0][0] < cutoff:
            self.buf.popleft()

    def slope_c_per_min(self):
        """Returns (slope_c_per_min, window_full) where window_full tells
        the caller whether there's enough history to trust the slope yet."""
        if len(self.buf) < 5:
            return None, False
        ts = np.array([p[0] for p in self.buf], dtype=float)
        temps = np.array([p[1] for p in self.buf], dtype=float)
        span = ts[-1] - ts[0]
        window_full = span >= (self.window_s * 0.9)
        A = np.vstack([ts, np.ones_like(ts)]).T
        slope_per_s, _ = np.linalg.lstsq(A, temps, rcond=None)[0]
        return slope_per_s * 60.0, window_full

    def is_stable(self, threshold_c_per_min):
        slope, window_full = self.slope_c_per_min()
        if slope is None or not window_full:
            return False, slope
        return bool(abs(slope) < threshold_c_per_min), slope


def run_inference_once(llm, prompt, max_tokens):
    """One inference call. Returns (tokens_generated, tok/s, ttft_sec).
    Mirrors benchmark.py's run_inference so tokens/sec numbers from both
    scripts are comparable.

    IMPORTANT: llm.reset() is called before every call. Without it, the
    KV cache/context from the previous call is still sitting in the
    Llama object, and because STRESS_PROMPT is byte-identical on every
    iteration, llama.cpp's prefix-caching matches the new prompt against
    the old (prompt + previously generated) context and reuses almost
    all of it -- and once the accumulated context creeps past n_ctx
    (512 here: ~stress prompt + max_tokens is already close to that
    after a single call), llama-cpp-python's automatic context-shift
    kicks in on every subsequent call. Net effect: after the first call,
    every call "generates" ~max_tokens in under a millisecond, which is
    physically impossible on this hardware and means the CPU stops
    actually being loaded for the rest of step_up -- exactly what showed
    up as pi_power_w barely rising above idle and a garbage tau fit
    (near-infinite single-exp tau, degenerate two-exp fit) on the
    thermal side. reset() clears n_past/the KV cache so every call does
    a real, fresh forward pass over the prompt again.
    """
    llm.reset()
    start = time.time()
    first_token_time = None
    tokens_generated = 0
    for _chunk in llm(prompt, max_tokens=max_tokens, temperature=0.7, echo=False, stream=True):
        if first_token_time is None:
            first_token_time = time.time()
        tokens_generated += 1
    end = time.time()
    ttft = (first_token_time - start) if first_token_time else None
    generation_time = (end - first_token_time) if first_token_time else (end - start)
    tps = (tokens_generated / generation_time) if generation_time > 0 else None

    # Sanity guard: a Pi 5 CPU cannot do meaningful forward passes fast
    # enough to hit these numbers. If it happens anyway (e.g. a future
    # llama-cpp-python version reintroduces a caching path we didn't
    # anticipate), flag it loudly in the console and null out the
    # sample's timing fields rather than silently logging a physically
    # impossible number that will corrupt the thermal fit again.
    MAX_PLAUSIBLE_TPS = 200.0  # generous upper bound for CPU inference on this hardware
    if tps is not None and tps > MAX_PLAUSIBLE_TPS:
        print(f"\n  WARNING: implausible tokens/sec ({tps:.0f}) -- likely a KV-cache/"
              f"context bug, not real inference. Discarding this sample's timing "
              f"(load-phase CPU stress will still have happened via the forward pass "
              f"up to the anomaly, but don't trust tokens_per_sec/ttft here).")
        tps = None
        ttft = None

    return tokens_generated, tps, ttft


def busy_stress(seconds=1.0):
    """CPU stress fallback when llama-cpp-python / a model file isn't
    available, so the state machine and thermal fit can still be
    exercised (e.g. for a --simulate dry run) without a GGUF on hand."""
    start = time.time()
    while time.time() - start < seconds:
        _ = sum(i * i for i in range(50_000))


def get_csv_path(output_dir, condition, trial):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"step_load_{condition}_trial{trial}_{ts}.csv")


def run_trial(args):
    os.makedirs(args.output, exist_ok=True)
    csv_path = get_csv_path(args.output, args.condition, args.trial)
    status_path = os.path.join(args.output, "dashboard_status.json")

    print(f"\n{'='*72}")
    print(f"  Step Load Test  |  condition={args.condition}  trial={args.trial}")
    print(f"{'='*72}")
    print(f"  Stabilization   : |slope| < {args.stabilize_slope} C/min over "
          f"{args.stabilize_window:.0f}s window")
    print(f"  Max idle wait   : {args.max_idle_wait/60:.0f} min")
    print(f"  Max hold wait   : {args.max_hold_wait/60:.0f} min")
    print(f"  Pump duty       : {args.pump_duty}%")
    print(f"  Power sampling  : every {args.power_sample_interval*1000:.0f} ms (background thread)")
    print(f"  Output CSV      : {csv_path}")
    if args.run_order_note:
        print(f"  Run-order note  : {args.run_order_note}")
    print(f"{'='*72}\n")

    if args.lock_clock:
        gc.lock_cpu_clock()

    pump_ready = gc.init_pump(args.pump_duty)
    if not pump_ready and not args.simulate:
        print("GPIO/pump control unavailable, pump_duty_pct will be logged but not actuated.\n")

    llm = None
    if not args.simulate and LLAMA_AVAILABLE and args.model and os.path.exists(args.model):
        print("Loading model... (this may take 30-60 seconds on Pi 5)")
        llm = Llama(model_path=args.model, n_ctx=512, n_threads=4, verbose=False)
        print("Model loaded.\n")
    elif not args.simulate:
        print("No usable model found, load phase will use CPU stress instead of real inference.\n")

    # Background high-rate power sampler. See geo_common.PowerSampler for
    # why a single per-iteration snapshot undersamples real Q(t) -- this
    # gives each logged row a time-weighted average power over the
    # interval since the previous row instead.
    sampler = None
    if not args.simulate:
        sampler = gc.PowerSampler(interval_s=args.power_sample_interval)
        sampler.start()

    idle_tracker = SlopeTracker(args.stabilize_window)
    load_tracker = SlopeTracker(args.stabilize_window)

    phase = "idle_pre"
    phase_start_t = 0.0
    t0 = time.time()
    last_power_ts = t0
    step_up_time = None
    step_down_time = None
    rows = []  # kept in memory too, so we can fit immediately without re-reading the CSV

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()

            print(f"[{phase}] waiting for stabilization...")

            while phase != "done":
                t = time.time() - t0

                temp = gc.get_cpu_temp()
                clock = gc.get_cpu_clock()
                throttle = gc.get_throttle_status()
                ambient = gc.read_ds18b20(args.ambient_sensor) if args.ambient_sensor else None
                t_in, t_out = gc.get_water_temps(args.t_in_sensor, args.t_out_sensor)

                if args.simulate:
                    temp, ambient, t_in, t_out, clock = _simulate_readings(
                        t, phase, args.condition, step_up_time, step_down_time
                    )
                    throttle = {"throttled": False, "freq_capped": False, "soft_temp_limit": False}

                if temp is not None and temp >= gc.MAX_SAFE_TEMP_C:
                    print(f"\n\nSAFETY STOP: CPU temp {temp}C reached the "
                          f"{gc.MAX_SAFE_TEMP_F}F limit during phase '{phase}'. "
                          f"Ending trial early -- flag this trial for exclusion/re-run.")
                    break

                delta_t_die_ambient = (temp - ambient) if (temp is not None and ambient is not None) else None

                flow_rate = gc.duty_to_flow_rate_lph(args.pump_duty)
                heat_removed = gc.compute_heat_removed_w(flow_rate, t_in, t_out)
                pump_power = gc.compute_pump_power_w(current_a=args.pump_current)
                if args.simulate:
                    pi_power = (8.0 if phase == "step_up" else 3.0) + np.random.normal(0, 0.1)
                else:
                    # Time-weighted average power since the last row, not a
                    # single instantaneous snapshot -- see PowerSampler.
                    now_ts = time.time()
                    pi_power = sampler.mean_since(last_power_ts)
                    if pi_power is None:
                        pi_power = sampler.latest()
                    last_power_ts = now_ts
                cop = gc.compute_cop(heat_removed, pump_power)
                pue = gc.compute_pue(pi_power, pump_power)
                total_power = (pi_power or 0) + (pump_power or 0) if pi_power is not None else None

                load_state = "load" if phase == "step_up" else "idle"

                tokens_generated = tps = ttft = None
                if phase == "step_up":
                    if llm is not None:
                        tokens_generated, tps, ttft = run_inference_once(llm, STRESS_PROMPT, args.max_tokens)
                    elif not args.simulate:
                        busy_stress(seconds=min(2.0, args.poll_interval))
                    else:
                        time.sleep(min(0.05, args.poll_interval))
                else:
                    time.sleep(args.poll_interval)

                # Stabilization check uses whichever tracker matches the
                # phase we're currently in (idle phases vs. the loaded hold).
                tracker = load_tracker if phase == "step_up" else idle_tracker
                tracker.add(t, temp)
                stable, slope = tracker.is_stable(args.stabilize_slope)

                row = {
                    "timestamp": datetime.now().isoformat(),
                    "time_s": round(t, 2),
                    "phase": phase,
                    "load_state": load_state,
                    "cooling_type": args.condition,
                    "trial": args.trial,
                    "cpu_temp_c": temp,
                    "ambient_c": ambient,
                    "delta_t_die_ambient_c": round(delta_t_die_ambient, 3) if delta_t_die_ambient is not None else "",
                    "t_in_c": round(t_in, 2) if t_in is not None else "",
                    "t_out_c": round(t_out, 2) if t_out is not None else "",
                    "cpu_clock_mhz": clock,
                    "throttled": int(throttle["throttled"]),
                    "pump_duty_pct": args.pump_duty,
                    "flow_rate_lph": round(flow_rate, 2) if flow_rate is not None else "",
                    "heat_removed_w": heat_removed if heat_removed is not None else "",
                    "pump_power_w": pump_power if pump_power is not None else "",
                    "pi_power_w": round(pi_power, 3) if pi_power is not None else "",
                    "power_w": round(total_power, 2) if total_power is not None else "",
                    "cop": cop if cop is not None else "",
                    "pue": pue if pue is not None else "",
                    "tokens_per_sec": round(tps, 2) if tps is not None else "",
                    "time_to_first_token_sec": round(ttft, 3) if ttft is not None else "",
                    "stabilization_slope_c_per_min": round(slope, 4) if slope is not None else "",
                }
                writer.writerow(row)
                rows.append(row)
                f.flush()

                gc.write_live_status(status_path, {
                    "elapsed_sec": round(t, 1),
                    "phase": phase,
                    "condition": args.condition,
                    "trial": args.trial,
                    "cpu_temp_c": temp,
                    "ambient_c": ambient,
                    "t_in_c": t_in,
                    "t_out_c": t_out,
                    "stabilization_slope_c_per_min": slope,
                    "stable": stable,
                    "tokens_per_sec": tps,
                    "pue": pue,
                })

                _print_status(t, phase, temp, ambient, t_in, slope, stable, args.stabilize_slope)

                phase_elapsed = t - phase_start_t
                phase = _advance_phase(
                    phase, stable, phase_elapsed, args,
                )
                if phase == "step_up" and step_up_time is None:
                    step_up_time = t
                    phase_start_t = t
                    load_tracker = SlopeTracker(args.stabilize_window)
                    print(f"\n[{phase}] load stepped ON at t={t:.0f}s, running to stabilization "
                          f"(max {args.max_hold_wait/60:.0f} min)...")
                elif phase == "step_down" and step_down_time is None:
                    step_down_time = t
                    phase_start_t = t
                    idle_tracker = SlopeTracker(args.stabilize_window)
                    print(f"\n[{phase}] load stepped OFF at t={t:.0f}s, idling to stabilization "
                          f"(max {args.max_idle_wait/60:.0f} min)...")
                elif phase == "idle_post" and phase_start_t != t and phase != "step_down":
                    pass

    except KeyboardInterrupt:
        print("\n\nStopped early by user.")
    finally:
        if sampler is not None:
            sampler.stop()
        gc.stop_pump()

    print(f"\n\n{'='*72}")
    print(f"  Trial complete. Data saved to {csv_path}")
    print(f"{'='*72}\n")

    if step_up_time is None:
        print("Never reached the step-up phase (idle never stabilized within --max-idle-wait). "
              "Nothing to fit. Consider a longer --max-idle-wait or check ambient conditions.")
        return csv_path, None

    _fit_and_report(csv_path, args, step_up_time, step_down_time)
    return csv_path, None


def _advance_phase(phase, stable, phase_elapsed, args):
    """State machine transitions. Each idle phase waits for stabilization
    (or a max-wait timeout, so a bad sensor or a drafty room can't hang
    the trial forever); step_up runs continuously until stabilization or
    its own timeout."""
    if phase == "idle_pre":
        if stable or phase_elapsed >= args.max_idle_wait:
            return "step_up"
        return "idle_pre"
    if phase == "step_up":
        if stable or phase_elapsed >= args.max_hold_wait:
            return "step_down"
        return "step_up"
    if phase == "step_down":
        if stable or phase_elapsed >= args.max_idle_wait:
            return "idle_post"
        return "step_down"
    if phase == "idle_post":
        if stable or phase_elapsed >= args.max_idle_wait:
            return "done"
        return "idle_post"
    return "done"


def _print_status(t, phase, temp, ambient, t_in, slope, stable, threshold):
    temp_str = f"{temp:.2f}C" if temp is not None else "n/a"
    amb_str = f"{ambient:.2f}C" if ambient is not None else "n/a"
    tin_str = f"{t_in:.2f}C" if t_in is not None else "n/a"
    slope_str = f"{slope:+.3f} C/min" if slope is not None else "warming up window..."
    stable_str = "STABLE" if stable else "..."
    print(
        f"\r[{t:>6.0f}s] {phase:<10} | CPU {temp_str:<8} amb {amb_str:<8} Tin {tin_str:<8} "
        f"| slope {slope_str:<16} (thresh {threshold}) | {stable_str}",
        end="", flush=True
    )


def _simulate_readings(t, phase, condition, step_up_time, step_down_time):
    """Synthetic first-order (or two-exponential, for geothermal) response
    generator, used only with --simulate, for exercising the state machine
    and the fit pipeline without real hardware attached."""
    ambient = 22.0 + 0.5 * np.sin(t / 900.0)
    tau_map = {"no_cooling": 220.0, "fan": 140.0, "geothermal": (60.0, 320.0)}
    dT_load = {"no_cooling": 35.0, "fan": 20.0, "geothermal": 12.0}[condition]
    baseline = ambient + 5.0

    if step_up_time is None:
        temp = baseline
    else:
        since_up = t - step_up_time
        if step_down_time is None:
            tau = tau_map[condition]
            if isinstance(tau, tuple):
                tau1, tau2 = tau
                temp = baseline + dT_load * (1 - 0.4 * np.exp(-since_up / tau1) - 0.6 * np.exp(-since_up / tau2))
            else:
                temp = baseline + dT_load * (1 - np.exp(-since_up / tau))
        else:
            since_down = t - step_down_time
            tau = tau_map[condition]
            peak = baseline + dT_load
            if isinstance(tau, tuple):
                tau1, tau2 = tau
                temp = baseline + dT_load * (0.4 * np.exp(-since_down / tau1) + 0.6 * np.exp(-since_down / tau2))
            else:
                temp = baseline + dT_load * np.exp(-since_down / tau)

    temp += np.random.normal(0, 0.15)
    t_in = ambient + 1.0 + np.random.normal(0, 0.1) if condition == "geothermal" else None
    t_out = t_in + 3.0 if t_in is not None else None
    clock = 2400
    return temp, ambient, t_in, t_out, clock


def _fit_and_report(csv_path, args, step_up_time, step_down_time):
    import pandas as pd
    df = pd.read_csv(csv_path)

    outdir = os.path.join(args.output, f"fit_{args.condition}_trial{args.trial}")
    os.makedirs(outdir, exist_ok=True)

    explicit_steps = [step_up_time]
    if step_down_time is not None:
        explicit_steps.append(step_down_time)

    print(f"\nFitting step transitions (step-up at t={step_up_time:.0f}s"
          + (f", step-down at t={step_down_time:.0f}s" if step_down_time else ", no step-down captured")
          + ")...")

    ambient_col = "ambient_c" if df["ambient_c"].notna().any() else None
    if ambient_col is None:
        print("No ambient sensor data logged -- falling back to absolute-temperature fitting. "
              "Wire up --ambient-sensor next run so R_th isn't sensitive to ambient drift.")

    results_df, results_csv, md_path, bench_plot = tsid.run_thermal_sysid(
        df, outdir,
        time_col="time_s", temp_col="cpu_temp_c", power_col="power_w",
        load_col="load_state", cooling_col="cooling_type",
        ambient_col=ambient_col,
        window_s=args.fit_window,
        explicit_steps=explicit_steps,
    )

    print(f"\n{'='*72}")
    print(f"  Fit results  ({args.condition}, trial {args.trial})")
    print(f"{'='*72}")
    if len(results_df) == 0:
        print("  No step could be fit (too few samples in the window -- check --fit-window "
              "and --poll-interval).")
    else:
        for _, row in results_df.iterrows():
            print(f"  {row['label']}")
            print(f"    basis          : {row['fit_basis']}")
            if row.get("window_clipped_to_next_step"):
                print(f"    window         : clipped to {row['effective_window_s']:.0f}s "
                      f"(next phase started before the requested --fit-window)")
            print(f"    chosen model   : {row['chosen_model']}"
                  + (f"  (delta AIC={row['delta_aic']}, residual z={row['residual_runs_z']})"
                     if row["chosen_model"] == "two_exponential" else ""))
            if row["chosen_model"] == "two_exponential":
                print(f"    tau1 / tau2    : {row['tau1_two_exp_s']:.1f}s / {row['tau2_two_exp_s']:.1f}s "
                      f"(R2={row['r2_two_exp']:.3f})")
            else:
                print(f"    tau (nonlinear): {row['tau_nonlinear_s']:.1f}s  (R2={row['r2_nonlinear']:.3f})")
                print(f"    tau (linearized): {row['tau_linearized_s']:.1f}s  (R2={row['r2_linearized']:.3f})")
            if "R_thermal_c_per_w" in row and not _is_nan(row["R_thermal_c_per_w"]):
                print(f"    R_th           : {row['R_thermal_c_per_w']:.3f} C/W")
                print(f"    C_th           : {row['C_thermal_j_per_c']:.1f} J/C")
    print(f"\n  Summary CSV  : {results_csv}")
    print(f"  Markdown     : {md_path}")
    print(f"  Benchmark plot: {bench_plot}")
    print(f"{'='*72}\n")


def _is_nan(x):
    try:
        return x != x
    except Exception:
        return True


def main():
    ap = argparse.ArgumentParser(
        description="Step Load Test: idle -> step up -> hold -> step down -> idle, "
                    "with live stabilization detection and automatic first-order/"
                    "two-exponential fitting at the end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", type=str, default="", help="Path to GGUF model file")
    ap.add_argument("--condition", choices=["no_cooling", "fan", "geothermal"], required=True)
    ap.add_argument("--trial", type=int, required=True, help="Trial number (1-5), for filenames/bookkeeping")
    ap.add_argument("--run-order-note", type=str, default="",
                     help="Free-text note on where this trial falls in your randomized run "
                          "order/day, purely for your own records in the console output")
    ap.add_argument("--output", type=str, default="./step_load_data")

    ap.add_argument("--ambient-sensor", type=str, default="",
                     help="DS18B20 device ID for ambient room temperature")
    ap.add_argument("--t-in-sensor", type=str, default="",
                     help="DS18B20 device ID for coolant inlet temp (geothermal condition)")
    ap.add_argument("--t-out-sensor", type=str, default="",
                     help="DS18B20 device ID for coolant outlet temp (geothermal condition)")

    ap.add_argument("--pump-duty", type=int, default=100)
    ap.add_argument("--pump-current", type=float, default=None)
    ap.add_argument("--lock-clock", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=150)

    ap.add_argument("--stabilize-slope", type=float, default=0.05,
                     help="Stabilization threshold in C/min (default: 0.05, per test plan)")
    ap.add_argument("--stabilize-window", type=float, default=120.0,
                     help="Rolling window in seconds over which the slope is estimated (default: 120)")
    ap.add_argument("--max-idle-wait", type=float, default=1800.0,
                     help="Max seconds to wait for an idle phase to stabilize before forcing "
                          "the transition anyway (default: 1800 = 30 min)")
    ap.add_argument("--max-hold-wait", type=float, default=2400.0,
                     help="Max seconds to hold the load phase before forcing step-down "
                          "(default: 2400 = 40 min)")
    ap.add_argument("--poll-interval", type=float, default=1.0,
                     help="Seconds between samples during idle phases / floor for the busy-stress "
                          "fallback during load (default: 1.0)")
    ap.add_argument("--fit-window", type=float, default=600.0,
                     help="Seconds of post-step data handed to the fitter (default: 600, i.e. "
                          "up to 10 min per transition -- increase for a slow geothermal tail)")
    ap.add_argument("--power-sample-interval", type=float, default=0.02,
                     help="Seconds between background power samples (default: 0.02 = 50Hz). "
                          "Each logged row's pi_power_w is the time-weighted average of these "
                          "samples over the interval since the previous row, not a single "
                          "instantaneous read. Lower this only as far as your hardware's real "
                          "refresh rate supports -- check whether consecutive `vcgencmd "
                          "pmic_read_adc` calls actually return distinct values at the rate "
                          "you're asking for before trusting a very small interval.")

    ap.add_argument("--simulate", action="store_true",
                     help="Generate synthetic sensor readings instead of touching real hardware. "
                          "For testing the state machine and fit pipeline off-Pi.")

    args = ap.parse_args()

    if args.condition == "geothermal" and not args.simulate and (not args.t_in_sensor or not args.t_out_sensor):
        print("Warning: geothermal condition without --t-in-sensor/--t-out-sensor -- "
              "coolant loop temps will be logged blank. Run with no sensor args first to "
              "list detected 1-wire IDs:")
        found = gc.list_available_temp_sensors()
        print(f"  Detected: {found if found else '(none found)'}")

    run_trial(args)


if __name__ == "__main__":
    main()