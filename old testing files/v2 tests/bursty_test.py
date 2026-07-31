#!/usr/bin/env python3
"""
bursty_test.py

Bursty Inference Test (validation run #1) for the geothermal cooling rig.

Runs repeated on/off load cycles (default: 2 min load, 1 min idle) for a
fixed total duration, mimicking a real inference server getting waves of
requests. This is deliberately NOT a new calibration: it loads the
tau/R_th/C_th (or two-exponential R1/tau1/R2/tau2) already fit by
step_load_test.py + thermal_system_id.py for the given condition, drives
that fitted RC model forward through the same on/off power sequence the
rig actually saw, and compares the model's predicted temperature curve
against what was actually measured. If the two track, the simple model
extracted from one clean step generalizes to a realistic load pattern;
if they diverge, that's the manuscript's evidence for where the lumped
model breaks down (e.g. under-damped multi-node behavior a single step
doesn't excite).

How the prediction works
-------------------------
Single-exponential calibration (no_cooling / fan, typically):
    dT/dt = (Q(t) - delta_T) / (R_th * C_th)
  Exact per-sample update (Q effectively constant over one poll interval):
    delta_T[n+1] = Q*R_th + (delta_T[n] - Q*R_th) * exp(-dt / tau)

Two-exponential calibration (geothermal, typically -- a Foster two-branch
network, i.e. two parallel first-order lags driven by the same Q(t)):
    x1[n+1] = Q*R1 + (x1[n] - Q*R1) * exp(-dt / tau1)
    x2[n+1] = Q*R2 + (x2[n] - Q*R2) * exp(-dt / tau2)
    delta_T_pred = x1 + x2

Both use whichever R/tau thermal_system_id.py already fit and saved to
tau_fit_summary.csv -- this script doesn't re-fit anything, only
simulates forward with those fixed parameters.

Q(t), the "how much heat is being generated right now" signal, is taken
from the rig's own measured CPU/SoC electrical power (q_cpu_w) when
available, falling back to fixed --load-power-w / --idle-power-w nominal
values if power telemetry isn't available (e.g. off-Pi testing). This has
to be the same convention thermal_system_id.py used for Q_ss when it
derived R_th, or the predicted delta_T will be scaled wrong -- which is
why this defaults to q_cpu_w, matching thermal_system_id.py's default
--power-col.

Usage
-----
    python3 bursty_test.py --model tinyllama.gguf --condition geothermal \
        --trial 1 --output ./trial_data \
        --tau-source ./trial_data/fit_geothermal_trial1/tau_fit_summary.csv \
        --ambient-sensor 28-0000ambient1 \
        --pump-duty 100 --lock-clock --total-minutes 25

    # Dry run without hardware:
    python3 bursty_test.py --condition geothermal --trial 1 \
        --output ./trial_data --tau-source .../tau_fit_summary.csv --simulate

Same MAX_SAFE_TEMP_C safety stop as the other two scripts applies.

CHANGE LOG (this file)
-----------------------
- pi_power is now read via geo_common.PowerSampler (background thread,
  time-weighted average over the interval since the last row) instead of
  a single instantaneous get_pi_power_w() snapshot per loop iteration,
  for the same reason it was changed in step_load_test.py: a single
  ~1s-cadence sample under-resolves real power fluctuations relative to
  the die's fast (~6s) time constant, which matters even more here since
  Q(t) is being fed straight into RCPredictor.step() every iteration.
- RCPredictor now supports seed(delta_T0), called once from the first
  row's real measured delta_T instead of assuming the rig starts at
  delta_T=0. Without this, the whole predicted curve tended to sit below
  measured by a near-constant offset for the length of a short trial,
  since the slow branch doesn't forget a wrong t=0 state quickly.
- Load generation during "load" phases no longer measures or reports
  tokens/sec. This script only needs the CPU to be genuinely busy running
  real forward passes for a known, controlled amount of wall-clock time
  each iteration -- nothing thermal depends on generation speed. Trying
  to measure it was the source of repeated false positives ("implausible
  tokens/sec"): KV-cache fast-returns, a small chat model hitting EOS
  after 1-2 tokens and producing a meaningless rate from near-zero
  elapsed time, and general timing noise all looked identical but needed
  different fixes. run_inference_burst() replaces run_inference_once(): it
  strings small generations together, resetting between each, until a
  fixed duration (--poll-interval, matching idle's cadence) has elapsed,
  regardless of how any individual completion behaves. This also fixes
  "load" rows being logged far less densely than "idle" rows, since
  iteration length is no longer at the mercy of however long a generation
  happened to take.
- Removed the coolant inlet/outlet sensors (--t-in-sensor/--t-out-sensor)
  and the fluid-side heat_removed_w / flow_rate_lph fields. Q(t) fed into
  the RC predictor (and used for COP) is now q_cpu_w, CPU/SoC electrical
  power, matching the q_cpu_w-based calibration thermal_system_id.py now
  produces by default. Ambient temperature logging is unaffected. Pump
  power is still computed (geothermal condition only, per the earlier
  phantom-power fix) and still feeds COP/PUE -- it's just no longer part
  of Q.
"""

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

import geo_common as gc

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

CSV_HEADERS = [
    "timestamp", "time_s", "cycle_num", "phase", "load_state", "cooling_type", "trial",
    "cpu_temp_c", "ambient_c", "delta_t_die_ambient_c", "cpu_clock_mhz", "throttled",
    "pump_duty_pct", "pump_power_w", "pi_power_w", "q_cpu_w", "power_w", "cop", "pue",
    "predicted_delta_t_c",
]


# ---------------------------------------------------------------------------
# Loading the step-test calibration
# ---------------------------------------------------------------------------

def load_calibration(tau_csv_path, condition, row_index=0):
    """
    Load one row of thermal_system_id.py's tau_fit_summary.csv for the
    given cooling condition and build a predictor from it. Every row is a
    two-exponential fit now -- thermal_system_id.py no longer fits or
    reports a single-exponential model, so there's no model-type branch
    here anymore.

    row_index selects which fit to use when a file has more than one
    (step_load_test.py fits step-up, then step-down, in that order --
    row 0 is step-up by default, which is the one you almost always want
    since it's the heating transient and has the larger, cleaner ΔT).
    """
    df = pd.read_csv(tau_csv_path)
    sub = df[df["cooling_type"] == condition].reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(
            f"No rows for cooling_type='{condition}' in {tau_csv_path}. "
            f"Found conditions: {df['cooling_type'].unique().tolist()}"
        )
    if row_index >= len(sub):
        raise ValueError(f"row_index={row_index} out of range, only {len(sub)} row(s) available.")
    row = sub.iloc[row_index]

    for col in ["tau1_two_exp_s", "tau2_two_exp_s",
                "R1_two_exp_c_per_w", "R2_two_exp_c_per_w"]:
        if col not in row or pd.isna(row[col]):
            raise ValueError(
                f"Calibration row '{row['label']}' is missing '{col}'. Either the "
                f"two-exponential fit didn't converge on that step (see the warning "
                f"thermal_system_id.py printed when it was generated), or q_cpu_w "
                f"wasn't logged during the step test. There is no single-exponential "
                f"fallback anymore -- re-run step_load_test.py with real (or --simulate) "
                f"power logging and, if the fit didn't converge, a longer --fit-window "
                f"on that phase."
            )

    tau1, tau2 = float(row["tau1_two_exp_s"]), float(row["tau2_two_exp_s"])
    R1, R2 = float(row["R1_two_exp_c_per_w"]), float(row["R2_two_exp_c_per_w"])

    # Near-degenerate check: when tau1 and tau2 are close together, the two
    # branches are nearly collinear and A1/A2 (hence R1/R2) become
    # numerically unstable -- they can individually blow up to unphysical
    # values while still summing to a perfectly reasonable total delta_T
    # and a good R^2 on the original step. There's no single-exponential
    # model to fall back to anymore, so this is now a warning rather than a
    # trigger for switching models -- the two-exponential fit is used
    # regardless, but flagged so you know to treat the forward-simulated
    # prediction with caution (or re-run the step test with a longer hold
    # to actually separate the branches).
    tau_ratio = max(tau1, tau2) / max(min(tau1, tau2), 1e-6)
    r_sum = R1 + R2
    r_single = float(row["R_thermal_c_per_w"]) if "R_thermal_c_per_w" in row and not pd.isna(row["R_thermal_c_per_w"]) else None
    r_ratio_bad = (r_single is not None and r_single > 0 and (r_sum / r_single > 5 or r_sum / r_single < 0.2))

    if tau_ratio < 1.3 or r_ratio_bad:
        print(f"  WARNING: two-exponential branches for '{row['label']}' are near-degenerate "
              f"(tau1={tau1:.1f}s, tau2={tau2:.1f}s, R1={R1:.2f}, R2={R2:.2f} C/W, "
              f"vs. summed R_th={r_single}) -- R1/R2 may not be trustworthy for forward "
              f"simulation. Proceeding with this fit anyway since there's no fallback "
              f"model; treat the predicted curve with caution, and consider re-running "
              f"the step test with a longer hold on this phase.")

    return {"label": row["label"], "source_row": row_index,
            "tau1": tau1, "tau2": tau2, "R1": R1, "R2": R2}


class RCPredictor:
    """
    Drives the calibrated two-exponential (Foster two-branch) thermal
    model forward through a measured Q(t) sequence, one sample at a time,
    using the exact analytic update for each branch with piecewise-
    constant input over each sample interval. This is not a new fit --
    every parameter comes from load_calibration() -- it's just forward
    simulation for comparison against the measured curve.
    """

    def __init__(self, calib):
        self.calib = calib
        self.x1 = 0.0
        self.x2 = 0.0

    def seed(self, delta_T0):
        """
        Initialize predictor state from a known starting delta_T (e.g. the
        rig's actual measured delta_T on the trial's first row), instead of
        assuming the model starts at equilibrium with zero power (delta_T=0).

        Without this, the predictor always starts from "die exactly at
        ambient," but the rig is rarely actually there at t=0 -- residual
        heat from whatever happened just before the bursty trial started
        typically leaves delta_T already elevated by several degrees. Since
        the slow branch's tau (tens of seconds or more) doesn't fully decay
        away within a short trial, an unseeded predictor spends a large
        fraction of the run still "remembering" the wrong starting point --
        which shows up as the whole predicted curve sitting below measured
        by a near-constant offset, even when the dynamics and amplitude are
        otherwise correct.

        The seed is split across the two branches proportional to their
        steady-state resistances (R1:R2), a reasonable approximation of how
        a system that had been running under some prior load would actually
        be distributed between the fast and slow thermal masses.
        """
        R1, R2 = self.calib["R1"], self.calib["R2"]
        total_R = R1 + R2
        if total_R > 0:
            self.x1 = delta_T0 * (R1 / total_R)
            self.x2 = delta_T0 * (R2 / total_R)
        else:
            self.x1 = delta_T0 / 2.0
            self.x2 = delta_T0 / 2.0

    def step(self, dt_s, Q_w):
        """Advance the model by dt_s seconds under constant power Q_w.
        Returns the predicted delta_T (die - ambient) after the step."""
        if dt_s <= 0:
            return self.predicted_delta_t()
        tau1, tau2 = self.calib["tau1"], self.calib["tau2"]
        R1, R2 = self.calib["R1"], self.calib["R2"]
        target1 = Q_w * R1
        target2 = Q_w * R2
        self.x1 = target1 + (self.x1 - target1) * np.exp(-dt_s / tau1)
        self.x2 = target2 + (self.x2 - target2) * np.exp(-dt_s / tau2)
        return self.predicted_delta_t()

    def predicted_delta_t(self):
        return self.x1 + self.x2


# ---------------------------------------------------------------------------
# Inference / stress helpers (shared pattern with the other two scripts)
# ---------------------------------------------------------------------------

def run_inference_burst(llm, prompt, duration_s, max_tokens_per_call=64):
    """
    Drive the model for approximately duration_s seconds of real forward-
    pass compute, purely to load the CPU for a known, controlled amount of
    wall-clock time. Returns nothing -- tokens/sec is deliberately not
    measured or reported anymore.

    This replaces the earlier run_inference_once(), which tried to also
    report generation speed and hit a long chain of false positives doing
    so: KV-cache fast-returns, a small chat model hitting EOS after 1-2
    tokens and producing a meaningless rate from near-zero elapsed time,
    and general timing noise. None of that mattered for the RC predictor
    -- it only ever consumed Q(t) (from PowerSampler), never a token rate.
    Rather than keep chasing every way a generation-speed measurement can
    go wrong, this sidesteps the whole category: no tokens are counted, no
    rate is computed, so there's nothing for a "bug" to corrupt. The only
    thing that matters is that llm() is genuinely running forward passes
    on the CPU for close to duration_s seconds.

    llm.reset() is called before each underlying call so every burst starts
    from a clean KV cache/context rather than accumulating state. Generation
    runs in a loop of small max_tokens_per_call chunks so elapsed time can
    be checked frequently and the burst stops close to duration_s regardless
    of how any individual completion behaves (including stopping early on
    EOS, which just triggers an immediate fresh reset()+restart within the
    same burst).
    """
    start = time.time()
    while time.time() - start < duration_s:
        llm.reset()
        for _chunk in llm(prompt, max_tokens=max_tokens_per_call, temperature=0.7,
                            echo=False, stream=True):
            if time.time() - start >= duration_s:
                break


def busy_stress(seconds=1.0):
    start = time.time()
    while time.time() - start < seconds:
        _ = sum(i * i for i in range(50_000))


def _simulate_readings(t, load_state, condition, ambient0=22.0):
    """Synthetic sensor readings for --simulate, independent of the
    calibration model (i.e. these are NOT generated from the RC predictor)
    so that comparing prediction vs. "measurement" in a simulated run is
    still a meaningful test of the comparison code, not a tautology."""
    ambient = ambient0 + 0.5 * np.sin(t / 900.0)
    rng_noise = np.random.normal(0, 0.15)
    tau_map = {"no_cooling": 220.0, "fan": 140.0, "geothermal": 90.0}
    dT_load = {"no_cooling": 35.0, "fan": 20.0, "geothermal": 12.0}[condition]
    tau = tau_map[condition]
    # crude single-lag response driven by load_state, just for a plausible
    # trace -- deliberately a slightly different tau than the "true" step
    # test simulation elsewhere, so the comparison isn't trivially perfect
    target = ambient + (dT_load if load_state == "load" else 0.0)
    if not hasattr(_simulate_readings, "_last_temp"):
        _simulate_readings._last_temp = ambient
    prev = _simulate_readings._last_temp
    dt = getattr(_simulate_readings, "_last_t", t)
    elapsed = max(t - dt, 0.01)
    temp = target + (prev - target) * np.exp(-elapsed / tau) + rng_noise
    _simulate_readings._last_temp = temp
    _simulate_readings._last_t = t
    return temp, ambient, 2400


# ---------------------------------------------------------------------------
# Main trial loop
# ---------------------------------------------------------------------------

def get_csv_path(output_dir, condition, trial):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"bursty_{condition}_trial{trial}_{ts}.csv")


def run_trial(args):
    os.makedirs(args.output, exist_ok=True)
    csv_path = get_csv_path(args.output, args.condition, args.trial)
    status_path = os.path.join(args.output, "dashboard_status.json")

    calib = load_calibration(args.tau_source, args.condition, args.calibration_row)
    predictor = RCPredictor(calib)

    print(f"\n{'='*72}")
    print(f"  Bursty Inference Test  |  condition={args.condition}  trial={args.trial}")
    print(f"{'='*72}")
    print(f"  Cycle           : {args.load_minutes} min load / {args.idle_minutes} min idle")
    print(f"  Total duration  : {args.total_minutes} min")
    print(f"  Calibration     : two_exponential from '{calib['label']}' "
          f"(row {args.calibration_row} of {args.tau_source})")
    print(f"    tau1={calib['tau1']:.1f}s R1={calib['R1']:.3f} C/W | "
          f"tau2={calib['tau2']:.1f}s R2={calib['R2']:.3f} C/W")
    print(f"  Power sampling  : every {args.power_sample_interval*1000:.0f} ms (background thread)")
    print(f"  Q source        : CPU electrical power (q_cpu_w)")
    print(f"  Output CSV      : {csv_path}")
    print(f"{'='*72}\n")

    if args.lock_clock:
        gc.lock_cpu_clock()

    pump_ready = gc.init_pump(args.pump_duty)
    if not pump_ready and not args.simulate:
        print("GPIO/pump control unavailable, pump_duty_pct will be logged but not actuated.\n")

    llm = None
    if not args.simulate and LLAMA_AVAILABLE and args.model and os.path.exists(args.model):
        print("Loading model...")
        llm = Llama(model_path=args.model, n_ctx=512, n_threads=4, verbose=False)
        # Explicitly disable any internal LlamaCache -- see run_inference_burst
        # for why this matters even with reset() in place.
        llm.set_cache(None)
        print("Model loaded.\n")
    elif not args.simulate:
        print("No usable model found, load phases will use CPU stress instead of real inference.\n")

    # Background high-rate power sampler -- see geo_common.PowerSampler
    # and the module docstring above for why a single per-iteration
    # snapshot undersamples real Q(t).
    sampler = None
    if not args.simulate:
        sampler = gc.PowerSampler(interval_s=args.power_sample_interval)
        sampler.start()

    load_s = args.load_minutes * 60.0
    idle_s = args.idle_minutes * 60.0
    cycle_s = load_s + idle_s
    total_s = args.total_minutes * 60.0

    t0 = time.time()
    last_t = 0.0
    last_power_ts = t0
    predictor_seeded = False
    rows = []

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            print("Running... (Ctrl+C to stop early)\n")

            try:
                while True:
                    t = time.time() - t0
                    if t >= total_s:
                        break

                    pos_in_cycle = t % cycle_s
                    load_state = "load" if pos_in_cycle < load_s else "idle"
                    cycle_num = int(t // cycle_s) + 1
                    phase = f"cycle{cycle_num}_{load_state}"

                    temp = gc.get_cpu_temp()
                    clock = gc.get_cpu_clock()
                    throttle = gc.get_throttle_status()
                    ambient = gc.read_ds18b20(args.ambient_sensor) if args.ambient_sensor else None

                    if args.simulate:
                        temp, ambient, clock = _simulate_readings(t, load_state, args.condition)
                        throttle = {"throttled": False, "freq_capped": False, "soft_temp_limit": False}

                    if temp is not None and temp >= gc.MAX_SAFE_TEMP_C:
                        print(f"\n\nSAFETY STOP: CPU temp {temp}C reached the "
                              f"{gc.MAX_SAFE_TEMP_F}F limit at t={t:.0f}s. Ending trial early.")
                        break

                    delta_t_die_ambient = (temp - ambient) if (temp is not None and ambient is not None) else None

                    # Seed the predictor from the real starting delta_T the
                    # first time it's available, instead of letting it start
                    # from an assumed delta_T=0. See RCPredictor.seed() --
                    # without this, the whole predicted curve tends to sit
                    # below measured by a near-constant offset for the length
                    # of a short trial, since the slow branch doesn't forget
                    # a wrong t=0 state quickly.
                    if not predictor_seeded and delta_t_die_ambient is not None:
                        predictor.seed(delta_t_die_ambient)
                        predictor_seeded = True

                    # Only geothermal actually has a pump doing work on a
                    # coolant loop -- see step_load_test.py for why this must
                    # not be treated as real electrical draw for fan/no_cooling
                    # trials. This no longer affects Q(t) (Q is q_cpu_w, not
                    # total power) but still matters for COP/PUE.
                    pump_power = gc.compute_pump_power_w(current_a=args.pump_current) if args.condition == "geothermal" else 0.0
                    if args.simulate:
                        pi_power = None
                    else:
                        # Time-weighted average power since the last row --
                        # this is what feeds Q(t) into the RC predictor below,
                        # so an undersampled/aliased power signal here directly
                        # distorts predicted_delta_t_c.
                        now_ts = time.time()
                        pi_power = sampler.mean_since(last_power_ts)
                        if pi_power is None:
                            pi_power = sampler.latest()
                        last_power_ts = now_ts
                    q_cpu = gc.compute_q_cpu_w(pi_power)
                    cop = gc.compute_cop(q_cpu, pump_power)
                    pue = gc.compute_pue(pi_power, pump_power)
                    total_power = (pi_power or 0) + (pump_power or 0) if pi_power is not None else None

                    # Q(t) for the predictor: prefer measured CPU electrical
                    # power; otherwise fall back to nominal load/idle wattage
                    # so the model can still be exercised without power
                    # telemetry (e.g. off-Pi or non-Pi-5 boards). This must
                    # match the Q convention thermal_system_id.py used when
                    # it fit R_th/tau (q_cpu_w by default), or the predicted
                    # delta_T will be scaled wrong.
                    if q_cpu is not None:
                        Q = q_cpu
                    else:
                        Q = args.load_power_w if load_state == "load" else args.idle_power_w

                    dt = max(t - last_t, 0.0)
                    pred_delta_t = predictor.step(dt, Q)
                    last_t = t

                    if load_state == "load":
                        if llm is not None:
                            # Bounded to --poll-interval, same cadence idle
                            # uses -- keeps "load" logging just as densely as
                            # "idle" instead of however long a generation
                            # happened to take.
                            run_inference_burst(llm, STRESS_PROMPT, args.poll_interval,
                                                 max_tokens_per_call=args.max_tokens)
                        elif not args.simulate:
                            busy_stress(seconds=min(2.0, args.poll_interval))
                        else:
                            time.sleep(min(0.05, args.poll_interval))
                    else:
                        time.sleep(args.poll_interval)

                    row = {
                        "timestamp": datetime.now().isoformat(),
                        "time_s": round(t, 2),
                        "cycle_num": cycle_num,
                        "phase": phase,
                        "load_state": load_state,
                        "cooling_type": args.condition,
                        "trial": args.trial,
                        "cpu_temp_c": temp,
                        "ambient_c": ambient,
                        "delta_t_die_ambient_c": round(delta_t_die_ambient, 3) if delta_t_die_ambient is not None else "",
                        "cpu_clock_mhz": clock,
                        "throttled": int(throttle["throttled"]),
                        "pump_duty_pct": args.pump_duty,
                        "pump_power_w": pump_power if pump_power is not None else "",
                        "pi_power_w": round(pi_power, 3) if pi_power is not None else "",
                        "q_cpu_w": round(q_cpu, 3) if q_cpu is not None else "",
                        "power_w": round(total_power, 2) if total_power is not None else "",
                        "cop": cop if cop is not None else "",
                        "pue": pue if pue is not None else "",
                        "predicted_delta_t_c": round(pred_delta_t, 3),
                    }
                    writer.writerow(row)
                    rows.append(row)
                    f.flush()

                    gc.write_live_status(status_path, {
                        "elapsed_sec": round(t, 1),
                        "total_sec": total_s,
                        "phase": phase,
                        "condition": args.condition,
                        "trial": args.trial,
                        "cpu_temp_c": temp,
                        "ambient_c": ambient,
                        "predicted_delta_t_c": pred_delta_t,
                        "measured_delta_t_c": delta_t_die_ambient,
                        "pue": pue,
                    })

                    _print_status(t, total_s, phase, temp, delta_t_die_ambient, pred_delta_t)

            except KeyboardInterrupt:
                print("\n\nStopped early by user.")
    finally:
        if sampler is not None:
            sampler.stop()
        gc.stop_pump()

    print(f"\n\n{'='*72}")
    print(f"  Trial complete. Data saved to {csv_path}")
    print(f"{'='*72}\n")

    _score_and_report(csv_path, args)
    return csv_path


def _print_status(t, total_s, phase, temp, measured_delta, pred_delta):
    temp_str = f"{temp:.2f}C" if temp is not None else "n/a"
    meas_str = f"{measured_delta:+.2f}C" if measured_delta is not None else "n/a"
    pred_str = f"{pred_delta:+.2f}C"
    print(
        f"\r[{t:>6.0f}s / {total_s:.0f}s] {phase:<16} | CPU {temp_str:<8} "
        f"| measured dT {meas_str:<8} | predicted dT {pred_str:<8}",
        end="", flush=True
    )


def _score_and_report(csv_path, args):
    """Compare the predicted delta_T trace against the measured one:
    RMSE, MAE, and a plot overlaying both curves against elapsed time."""
    df = pd.read_csv(csv_path)
    valid = df.dropna(subset=["delta_t_die_ambient_c", "predicted_delta_t_c"])

    outdir = os.path.join(args.output, f"bursty_fit_check_{args.condition}_trial{args.trial}")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  Prediction vs. measurement  ({args.condition}, trial {args.trial})")
    print(f"{'='*72}")

    if len(valid) < 5:
        print("  Not enough samples with both a measured and predicted delta_T "
              "to score (check --ambient-sensor was actually logging).")
        return

    measured = valid["delta_t_die_ambient_c"].values.astype(float)
    predicted = valid["predicted_delta_t_c"].values.astype(float)
    resid = measured - predicted
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(resid))
    corr = float(np.corrcoef(measured, predicted)[0, 1]) if len(measured) > 1 else np.nan

    print(f"  n samples   : {len(valid)}")
    print(f"  RMSE        : {rmse:.3f} C")
    print(f"  MAE         : {mae:.3f} C")
    print(f"  Bias        : {bias:+.3f} C  (positive = measured runs hotter than predicted)")
    print(f"  Correlation : {corr:.3f}")

    summary = {
        "condition": args.condition, "trial": args.trial,
        "n_samples": len(valid), "rmse_c": rmse, "mae_c": mae,
        "bias_c": bias, "correlation": corr,
    }
    summary_path = os.path.join(outdir, "prediction_score.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax1.plot(valid["time_s"], measured, label="Measured delta_T", color="#444444", lw=1.2)
        ax1.plot(valid["time_s"], predicted, label="Predicted delta_T (step-test model)",
                  color="#d95f02", lw=1.2, ls="--")
        ax1.set_ylabel("delta_T (die - ambient) [C]")
        ax1.set_title(f"Bursty test: predicted vs. measured, {args.condition} trial {args.trial}\n"
                      f"RMSE={rmse:.2f}C  MAE={mae:.2f}C  bias={bias:+.2f}C")
        ax1.legend(loc="best", fontsize=9)
        ax1.grid(alpha=0.3)

        ax2.plot(valid["time_s"], resid, color="#1b9e77", lw=1.0)
        ax2.axhline(0, color="black", lw=0.8)
        ax2.set_xlabel("Elapsed time [s]")
        ax2.set_ylabel("Residual (measured - predicted) [C]")
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        plot_path = os.path.join(outdir, "prediction_vs_measured.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Plot        : {plot_path}")
    except ImportError:
        print("  matplotlib not installed, skipping plot. pip install matplotlib")

    print(f"  Score CSV   : {summary_path}")
    print(f"{'='*72}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Bursty Inference Test: repeated on/off load cycles, scored "
                    "against the RC model fit by step_load_test.py + thermal_system_id.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", type=str, default="", help="Path to GGUF model file")
    ap.add_argument("--condition", choices=["no_cooling", "fan", "geothermal"], required=True)
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--output", type=str, default="./bursty_data")

    ap.add_argument("--tau-source", type=str, required=True,
                     help="Path to a tau_fit_summary.csv produced by step_load_test.py "
                          "(the fit_<condition>_trial<N>/tau_fit_summary.csv from a prior "
                          "step-load run for this same condition)")
    ap.add_argument("--calibration-row", type=int, default=0,
                     help="Which row (within this condition) of tau_fit_summary.csv to use "
                          "as the calibration. Default 0 = the step-up transition, which "
                          "step_load_test.py always fits first.")

    ap.add_argument("--load-minutes", type=float, default=2.0)
    ap.add_argument("--idle-minutes", type=float, default=1.0)
    ap.add_argument("--total-minutes", type=float, default=25.0,
                     help="Total trial duration (default 25, per the 20-30 min test plan)")

    ap.add_argument("--ambient-sensor", type=str, default="")

    ap.add_argument("--pump-duty", type=int, default=100)
    ap.add_argument("--pump-current", type=float, default=None)
    ap.add_argument("--lock-clock", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=64,
                     help="Max tokens per underlying generation call inside a load burst "
                          "(default: 64). This is a chunk size, not a target -- "
                          "run_inference_burst() strings calls together until "
                          "--poll-interval of wall-clock time has elapsed, regardless of "
                          "how many chunks that takes.")
    ap.add_argument("--poll-interval", type=float, default=1.0)
    ap.add_argument("--power-sample-interval", type=float, default=0.02,
                     help="Seconds between background power samples (default: 0.02 = 50Hz). "
                          "Each logged row's pi_power_w / q_cpu_w (and hence the Q(t) fed "
                          "into the RC predictor) is the time-weighted average of these "
                          "samples over the interval since the previous row, not a single "
                          "instantaneous read. See geo_common.PowerSampler.")

    ap.add_argument("--load-power-w", type=float, default=8.0,
                     help="Fallback Q(t) during load phases if power telemetry isn't "
                          "available (default: 8.0 W, adjust to your measured Pi 5 full-load draw)")
    ap.add_argument("--idle-power-w", type=float, default=3.0,
                     help="Fallback Q(t) during idle phases if power telemetry isn't available")

    ap.add_argument("--simulate", action="store_true",
                     help="Generate synthetic sensor readings instead of touching real hardware")

    args = ap.parse_args()

    run_trial(args)


if __name__ == "__main__":
    main()