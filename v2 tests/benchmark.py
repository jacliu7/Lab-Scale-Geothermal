#!/usr/bin/env python3
"""
Geothermal Cooling Demo - AI Performance Benchmark

Runs a small LLM (via llama.cpp) in a continuous loop while logging:
  - CPU temperature, clock speed, throttle status
  - Water loop inlet/outlet temperature (DS18B20, 1-wire)
  - Pump duty cycle, estimated flow rate, heat removed
  - Pi power draw, pump power draw, COP, PUE
  - Tokens per second, time to first token

Saves all data to a timestamped CSV, writes a live-updating JSON status
file for a dashboard to poll, and can generate the standard benchmark
graphs (CPU temp / tok-s / TTFT vs elapsed time) at the end of the run.

Sensor reads, pump control, and the COP/PUE/heat-removed math live in
geo_common.py, shared with step_load_test.py, so both scripts agree on
how those numbers are computed.

Usage:
  python3 benchmark.py --model path/to/model.gguf --condition geothermal \
      --duration 20 --pump-duty 100 \
      --t-in-sensor 28-0000abcd1111 --t-out-sensor 28-0000abcd2222 \
      --lock-clock --graphs

Requirements:
  pip install llama-cpp-python matplotlib gpiozero
  GGUF model file (e.g. TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf)
  Download from: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF

Hardware notes:
  - Water temp sensors are DS18B20s read over 1-wire. Enable the overlay
    with "dtoverlay=w1-gpio" in /boot/config.txt, then reboot. Run this
    script once without --t-in-sensor/--t-out-sensor to print the sensor
    IDs currently visible under /sys/bus/w1/devices/28-*, then pass those
    IDs in on the next run.
  - Pump speed control uses gpiozero's PWMOutputDevice (BCM pin 18 by
    default, change PUMP_PWM_PIN in geo_common.py if wired differently).
    gpiozero picks the correct backend automatically, including on Pi 5's
    RP1 I/O chip, which is why this script doesn't use RPi.GPIO directly.
  - Pi power draw is read from the PMIC via "vcgencmd pmic_read_adc",
    which is Pi 5 specific. On other boards this will return None and
    pi_power_w / pue will be logged blank.
"""

import csv
import time
import argparse
import os
import glob
from datetime import datetime

import pandas as pd

import geo_common as gc

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("Warning: llama-cpp-python not installed. Running in CPU stress-only mode.")
    print("Install with: pip install llama-cpp-python")

if not gc.GPIO_AVAILABLE:
    print("Warning: gpiozero not installed. Pump PWM control unavailable.")
    print("Install with: pip install gpiozero")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Prompt used for repeated LLM inference (fixed size, per the constants list).
# Extreme mode alternates between two prompts of different length/shape so
# back-to-back calls aren't all identical prompt-eval + generation profiles
# -- a more realistic (and slightly harder on the cache/scheduler) stand-in
# for "continuous varied demand" than hammering the exact same string.

STRESS_PROMPT = (
    "Explain in detail how geothermal heat exchange systems work, "
    "including the thermodynamic principles, the role of the ground loop, "
    "heat pumps, and why underground temperatures remain stable year-round. "
    "Include comparisons with conventional cooling systems."
)

STRESS_PROMPT_B = (
    "Write a detailed technical comparison of air-cooled, liquid-cooled, and "
    "immersion-cooled data center thermal management strategies. Cover capital "
    "cost, PUE impact, maintenance burden, and failure modes for each, and "
    "explain how facility scale changes which approach makes sense."
)


def load_max_tau_seconds(tau_csv_path, select="slowest", condition=None):
    """
    Read a tau_fit_summary.csv (thermal_system_id.py output, usually from
    step_load_test.py) and return a representative time constant in
    seconds, for sizing the extreme-stress duration.

    select="slowest" (default): max tau across every condition in the
    file, per the test plan's "use the slowest condition" rule -- this is
    what should drive a shared duration for all three conditions, so a
    fast condition's run isn't stopped before a slow condition would have
    reached steady state under the same protocol.
    select="this-condition": max tau for just `condition`.

    For a two-exponential row, tau2 (the slower/dominant branch) is used,
    since that's the one that governs how long it takes to approach
    steady state -- tau1 is the fast initial transient, not the limiting
    factor for "how long until this settles."
    """
    df = pd.read_csv(tau_csv_path)
    if select == "this-condition":
        if condition is None:
            raise ValueError("select='this-condition' requires a condition.")
        df = df[df["cooling_type"] == condition]
        if len(df) == 0:
            raise ValueError(f"No rows for cooling_type='{condition}' in {tau_csv_path}.")

    taus = []
    for _, row in df.iterrows():
        if row.get("chosen_model") == "two_exponential" and not pd.isna(row.get("tau2_two_exp_s")):
            taus.append(float(row["tau2_two_exp_s"]))
        elif not pd.isna(row.get("tau_nonlinear_s")):
            taus.append(float(row["tau_nonlinear_s"]))
    if not taus:
        raise ValueError(f"No usable tau values found in {tau_csv_path}.")
    return max(taus)


def run_inference(llm, prompt, max_tokens):
    """
    Run one inference call, streaming tokens so we can capture time to
    first token. llama-cpp-python's high-level API doesn't expose native
    prompt-eval timing without parsing stderr, so prompt_eval_time is
    approximated as TTFT (the wall-clock delay before the first token
    appears is dominated by prompt evaluation for these short prompts).
    """
    start = time.time()
    first_token_time = None
    tokens_generated = 0

    for _chunk in llm(prompt, max_tokens=max_tokens, temperature=0.7, echo=False, stream=True):
        if first_token_time is None:
            first_token_time = time.time()
        tokens_generated += 1

    end = time.time()
    total_time = end - start
    ttft = (first_token_time - start) if first_token_time else None
    prompt_eval_time_approx = ttft
    generation_time = (end - first_token_time) if first_token_time else total_time
    tps = (tokens_generated / generation_time) if generation_time > 0 else None

    return tokens_generated, tps, total_time, ttft, prompt_eval_time_approx


# CSV logging

def get_csv_path(condition):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"benchmark_{condition}_{timestamp}.csv"


CSV_HEADERS = [
    "timestamp",
    "elapsed_sec",
    "condition",
    "cpu_temp_c",
    "cpu_clock_mhz",
    "throttled",
    "freq_capped",
    "soft_temp_limit",
    "t_in_c",
    "t_out_c",
    "pump_duty_pct",
    "flow_rate_lph",
    "heat_removed_w",
    "pump_power_w",
    "pi_power_w",
    "cop",
    "pue",
    "tokens_per_sec",
    "tokens_generated",
    "inference_time_sec",
    "time_to_first_token_sec",
    "prompt_eval_time_approx_sec",
]


def write_row(writer, elapsed, condition, temp, clock, throttle,
              t_in, t_out, pump_duty, flow_rate, heat_removed,
              pump_power, pi_power, cop, pue,
              tps, tokens, inf_time, ttft, pe_time):
    writer.writerow({
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "condition": condition,
        "cpu_temp_c": temp,
        "cpu_clock_mhz": clock,
        "throttled": int(throttle["throttled"]),
        "freq_capped": int(throttle["freq_capped"]),
        "soft_temp_limit": int(throttle["soft_temp_limit"]),
        "t_in_c": round(t_in, 2) if t_in is not None else "",
        "t_out_c": round(t_out, 2) if t_out is not None else "",
        "pump_duty_pct": pump_duty,
        "flow_rate_lph": round(flow_rate, 2) if flow_rate is not None else "",
        "heat_removed_w": heat_removed if heat_removed is not None else "",
        "pump_power_w": pump_power if pump_power is not None else "",
        "pi_power_w": pi_power if pi_power is not None else "",
        "cop": cop if cop is not None else "",
        "pue": pue if pue is not None else "",
        "tokens_per_sec": round(tps, 2) if tps is not None else "",
        "tokens_generated": tokens if tokens is not None else "",
        "inference_time_sec": round(inf_time, 2) if inf_time is not None else "",
        "time_to_first_token_sec": round(ttft, 3) if ttft is not None else "",
        "prompt_eval_time_approx_sec": round(pe_time, 3) if pe_time is not None else "",
    })


# Live terminal display

def print_status(elapsed, duration_min, condition, temp, clock, throttle,
                  t_in, t_out, cop, pue, tps, ttft):
    throttle_str = "YES" if throttle["throttled"] else "no"
    tps_str = f"{tps:.2f}" if tps is not None else "running..."
    ttft_str = f"{ttft:.3f}s" if ttft is not None else "n/a"
    t_in_str = f"{t_in:.1f}" if t_in is not None else "n/a"
    t_out_str = f"{t_out:.1f}" if t_out is not None else "n/a"
    cop_str = f"{cop:.2f}" if cop is not None else "n/a"
    pue_str = f"{pue:.2f}" if pue is not None else "n/a"
    print(
        f"\r[{elapsed:>5.0f}s / {duration_min * 60}s] "
        f"{condition:<12} | CPU {temp}C {clock}MHz throt:{throttle_str:<3} | "
        f"Tin {t_in_str}C Tout {t_out_str}C | COP {cop_str} PUE {pue_str} | "
        f"Tok/s {tps_str:<8} | TTFT {ttft_str:<8}",
        end="", flush=True
    )


# Post-run graphs

def generate_graphs(csv_path, output_dir):
    """Produce the three standard benchmark graphs from the completed CSV:
    CPU temp, tokens/sec, and time-to-first-token, each vs elapsed time."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not installed, skipping graphs. pip install matplotlib")
        return []

    elapsed, temps, tps_list, ttft_list = [], [], [], []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            elapsed.append(float(row["elapsed_sec"]))
            temps.append(float(row["cpu_temp_c"]) if row["cpu_temp_c"] else None)
            tps_list.append(float(row["tokens_per_sec"]) if row["tokens_per_sec"] else None)
            ttft_list.append(float(row["time_to_first_token_sec"]) if row["time_to_first_token_sec"] else None)

    base = os.path.splitext(os.path.basename(csv_path))[0]
    saved = []

    def _plot(values, ylabel, title, suffix):
        xs = [e for e, v in zip(elapsed, values) if v is not None]
        ys = [v for v in values if v is not None]
        if not ys:
            return
        plt.figure(figsize=(9, 4.5))
        plt.plot(xs, ys, marker="o", markersize=2, linewidth=1)
        plt.xlabel("Elapsed time (s)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        out_path = os.path.join(output_dir, f"{base}_{suffix}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        saved.append(out_path)

    _plot(temps, "CPU Temp (C)", "CPU Temperature over Time", "cpu_temp")
    _plot(tps_list, "Tokens/sec", "Inference Throughput over Time", "tokens_per_sec")
    _plot(ttft_list, "Time to First Token (s)", "TTFT over Time", "ttft")

    return saved


# Main benchmark loop

def run_benchmark(model_path, condition, duration_min, output_dir, args):
    duration_sec = duration_min * 60
    csv_path = os.path.join(output_dir, get_csv_path(condition))
    status_path = os.path.join(output_dir, "dashboard_status.json")
    command_path = os.path.join(output_dir, "pump_command.json")
    current_pump_duty = args.pump_duty
    last_cmd_mtime = None

    print(f"\n{'='*70}")
    print(f"  Geothermal Cooling Demo - AI Performance Benchmark"
          + ("  [EXTREME STRESS MODE]" if args.extreme else ""))
    print(f"{'='*70}")
    print(f"  Condition   : {condition}")
    print(f"  Duration    : {duration_min:.1f} minutes"
          + (f"  (derived from tau, see below)" if args.extreme else ""))
    print(f"  Model       : {model_path if model_path else 'CPU stress mode (no model)'}")
    print(f"  Threads     : {args.threads}")
    print(f"  Pump duty   : {args.pump_duty}%")
    print(f"  Safety limit: {gc.MAX_SAFE_TEMP_F}F ({gc.MAX_SAFE_TEMP_C:.1f}C)")
    print(f"  Output      : {csv_path}")
    print(f"{'='*70}\n")

    if args.lock_clock:
        gc.lock_cpu_clock()

    if not args.t_in_sensor or not args.t_out_sensor:
        found = gc.list_available_temp_sensors()
        if found:
            print(f"1-wire sensors detected: {found}")
            print("Pass --t-in-sensor and --t-out-sensor with these IDs to log loop temps.\n")
        else:
            print("No 1-wire DS18B20 sensors detected. Loop temps will be logged blank.\n")

    pump_ready = gc.init_pump(args.pump_duty)
    if not pump_ready:
        print("GPIO/pump control unavailable, pump_duty_pct will be logged but not actuated.\n")

    llm = None
    if LLAMA_AVAILABLE and model_path and os.path.exists(model_path):
        print("Loading model... (this may take 30-60 seconds on Pi 5)")
        llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=args.threads,
            verbose=False,
        )
        print("Model loaded. Starting benchmark.\n")
    elif model_path:
        print(f"Model file not found at: {model_path}")
        print("Running in CPU stress mode instead.\n")

    start_time = time.time()
    inference_count = 0

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()

            print("Running... (Ctrl+C to stop early)\n")

            try:
                while True:
                    elapsed = time.time() - start_time
                    if elapsed >= duration_sec:
                        break

                    new_duty, last_cmd_mtime = gc.read_pump_command(command_path, last_cmd_mtime)
                    if new_duty is not None:
                        current_pump_duty = gc.set_pump_speed(new_duty)
                        print(f"\nPump duty updated to {current_pump_duty}% via dashboard.")

                    temp = gc.get_cpu_temp()
                    clock = gc.get_cpu_clock()
                    throttle = gc.get_throttle_status()
                    t_in, t_out = gc.get_water_temps(args.t_in_sensor, args.t_out_sensor)

                    if temp is not None and temp >= gc.MAX_SAFE_TEMP_C:
                        print(f"\n\nSAFETY STOP: CPU temp {temp}C reached the "
                              f"{gc.MAX_SAFE_TEMP_F}F limit. Ending run early.")
                        break

                    flow_rate = gc.duty_to_flow_rate_lph(current_pump_duty)
                    heat_removed = gc.compute_heat_removed_w(flow_rate, t_in, t_out)
                    pump_power = gc.compute_pump_power_w(current_a=args.pump_current)
                    pi_power = gc.get_pi_power_w()
                    cop = gc.compute_cop(heat_removed, pump_power)
                    pue = gc.compute_pue(pi_power, pump_power)

                    if llm is not None:
                        prompt = (
                            STRESS_PROMPT_B if (args.extreme and inference_count % 2 == 1)
                            else STRESS_PROMPT
                        )
                        tokens, tps, inf_time, ttft, pe_time = run_inference(
                            llm, prompt, args.max_tokens
                        )
                        inference_count += 1
                    else:
                        stress_start = time.time()
                        while time.time() - stress_start < 5:
                            _ = sum(i * i for i in range(100_000))
                        inf_time = time.time() - stress_start
                        tokens, tps, ttft, pe_time = None, None, None, None

                    write_row(
                        writer, elapsed, condition, temp, clock, throttle,
                        t_in, t_out, current_pump_duty, flow_rate, heat_removed,
                        pump_power, pi_power, cop, pue,
                        tps, tokens, inf_time, ttft, pe_time
                    )
                    f.flush()

                    gc.write_live_status(status_path, {
                        "elapsed_sec": round(elapsed, 1),
                        "duration_sec": duration_sec,
                        "condition": condition,
                        "cpu_temp_c": temp,
                        "cpu_clock_mhz": clock,
                        "throttled": throttle["throttled"],
                        "t_in_c": t_in,
                        "t_out_c": t_out,
                        "pump_duty_pct": current_pump_duty,
                        "flow_rate_lph": flow_rate,
                        "heat_removed_w": heat_removed,
                        "pump_power_w": pump_power,
                        "pi_power_w": pi_power,
                        "cop": cop,
                        "pue": pue,
                        "tokens_per_sec": tps,
                        "time_to_first_token_sec": ttft,
                    })

                    print_status(elapsed, duration_min, condition, temp, clock,
                                 throttle, t_in, t_out, cop, pue, tps, ttft)

            except KeyboardInterrupt:
                print("\n\nStopped early by user.")
    finally:
        gc.stop_pump()

    elapsed_total = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"  Benchmark complete.")
    print(f"  Total time     : {elapsed_total:.0f} seconds")
    print(f"  Inferences run : {inference_count}")
    print(f"  Data saved to  : {csv_path}")
    print(f"{'='*70}\n")

    if args.graphs:
        saved = generate_graphs(csv_path, output_dir)
        for p in saved:
            print(f"Saved graph: {p}")

    if args.extreme:
        snapshot_path = _extreme_stress_snapshot(csv_path, output_dir, condition, args)
        print(f"Extreme-stress snapshot saved to: {snapshot_path}")

    return csv_path


def _extreme_stress_snapshot(csv_path, output_dir, condition, args):
    """
    End-of-run summary for an extreme-stress trial: whether the rig
    actually reached a safe steady state within the duration, plus the
    PUE / COP / heat-rejection-rate snapshot from the tail of the run
    (sustained full load), which is the number that goes in the
    manuscript's PUE comparison. Explicitly labeled as a sustained-
    full-utilization bench-scale figure, not a time-averaged operational
    PUE, per the test plan -- hyperscalers report time-averaged PUE over
    a whole fleet's varying load; this is a single rig pinned at 100%
    load, which is a different (harder) number and shouldn't be quoted
    against theirs without that caveat attached.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["cpu_temp_c"])
    if len(df) < 5:
        print("Not enough samples to build an extreme-stress snapshot.")
        return None

    tail_frac = max(0.05, min(0.2, 300.0 / max(df["elapsed_sec"].max(), 1.0)))
    tail = df[df["elapsed_sec"] >= df["elapsed_sec"].max() * (1 - tail_frac)]

    # Steady-state check: regress temp vs. time over the tail window, see
    # if the slope is below the same 0.05 C/min bar step_load_test.py uses
    # for "stable". This is diagnostic only (extreme mode doesn't stop
    # early on this), so a real trial that didn't converge still finishes
    # its full duration and gets flagged rather than cut short.
    import numpy as np
    t = tail["elapsed_sec"].values.astype(float)
    temp = tail["cpu_temp_c"].values.astype(float)
    slope_c_per_min = np.nan
    reached_steady_state = None
    if len(t) >= 5 and (t[-1] - t[0]) > 0:
        A = np.vstack([t, np.ones_like(t)]).T
        slope_per_s, _ = np.linalg.lstsq(A, temp, rcond=None)[0]
        slope_c_per_min = slope_per_s * 60.0
        reached_steady_state = bool(abs(slope_c_per_min) < args.stabilize_slope_check)

    max_temp = float(df["cpu_temp_c"].max())
    safety_stop_hit = max_temp >= gc.MAX_SAFE_TEMP_C - 0.5  # small margin for the last logged sample

    def _tail_mean(col):
        vals = pd.to_numeric(tail[col], errors="coerce").dropna()
        return float(vals.mean()) if len(vals) else None

    pue_sustained = _tail_mean("pue")
    cop_sustained = _tail_mean("cop")
    heat_rejection_w = _tail_mean("heat_removed_w")
    pi_power_w = _tail_mean("pi_power_w")
    pump_power_w = _tail_mean("pump_power_w")
    tps_sustained = _tail_mean("tokens_per_sec")

    snapshot = {
        "condition": condition,
        "csv_path": csv_path,
        "duration_min": round(df["elapsed_sec"].max() / 60.0, 2),
        "tail_window_fraction": round(tail_frac, 3),
        "max_cpu_temp_c": max_temp,
        "safety_stop_hit": safety_stop_hit,
        "steady_state_slope_c_per_min": round(slope_c_per_min, 4) if not np.isnan(slope_c_per_min) else None,
        "reached_steady_state": reached_steady_state,
        "pue_sustained_full_load": pue_sustained,
        "cop_sustained_full_load": cop_sustained,
        "heat_rejection_rate_w": heat_rejection_w,
        "pi_power_w_sustained": pi_power_w,
        "pump_power_w_sustained": pump_power_w,
        "tokens_per_sec_sustained": tps_sustained,
        "framing_note": (
            "PUE and COP here are computed at sustained full utilization (this "
            "trial's tail window, all cores pinned at 100% inference load, no "
            "idle periods). This is a bench-scale worst-case analog, NOT a "
            "time-averaged operational PUE like hyperscalers report (which "
            "blend load across a fleet with idle/partial-load periods). Report "
            "it as such -- do not compare directly to a published fleet-average "
            "PUE without this caveat."
        ),
    }

    import json
    snapshot_path = os.path.join(
        output_dir, f"extreme_stress_snapshot_{condition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Extreme-stress snapshot ({condition})")
    print(f"{'='*70}")
    print(f"  Duration           : {snapshot['duration_min']:.1f} min")
    print(f"  Max CPU temp       : {max_temp:.2f} C" + ("  *** SAFETY STOP HIT ***" if safety_stop_hit else ""))
    if reached_steady_state is not None:
        state = "YES" if reached_steady_state else "NO -- still drifting at end of run"
        print(f"  Reached steady state in tail window: {state} "
              f"(slope={snapshot['steady_state_slope_c_per_min']} C/min)")
    if pue_sustained is not None:
        print(f"  PUE (sustained full load) : {pue_sustained:.3f}")
    if cop_sustained is not None:
        print(f"  COP (sustained full load) : {cop_sustained:.3f}")
    if heat_rejection_w is not None:
        print(f"  Heat rejection rate       : {heat_rejection_w:.2f} W")
    if tps_sustained is not None:
        print(f"  Tokens/sec (sustained)    : {tps_sustained:.2f}")
    print(f"  NOTE: {snapshot['framing_note']}")
    print(f"{'='*70}\n")

    return snapshot_path


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Geothermal demo AI performance benchmark / extreme stress test"
    )
    parser.add_argument("--model", type=str, default="",
                         help="Path to GGUF model file (e.g. tinyllama.gguf)")
    parser.add_argument("--condition", type=str,
                         choices=["no_cooling", "fan", "geothermal"], required=True,
                         help="Cooling condition being tested")
    parser.add_argument("--duration", type=int, default=20,
                         help="Trial duration in minutes (default: 20). Ignored if --extreme "
                              "is set and a duration can be derived from --tau-source/--tau-seconds.")
    parser.add_argument("--output", type=str, default=".",
                         help="Directory to save CSV/graph output (default: current directory)")
    parser.add_argument("--t-in-sensor", type=str, default="",
                         help="DS18B20 device ID for loop inlet temp (e.g. 28-0000abcd1111)")
    parser.add_argument("--t-out-sensor", type=str, default="",
                         help="DS18B20 device ID for loop outlet temp")
    parser.add_argument("--pump-duty", type=int, default=100,
                         help="Pump PWM duty cycle percent, 0-100 (default: 100)")
    parser.add_argument("--pump-current", type=float, default=None,
                         help="Measured pump current in amps, overrides the "
                              "spec-sheet midpoint estimate")
    parser.add_argument("--lock-clock", action="store_true",
                         help="Pin CPU governor to performance for a stable "
                              "clock speed across conditions")
    parser.add_argument("--max-tokens", type=int, default=150,
                         help="Fixed output token count per inference call (default: 150, "
                              "consider raising for --extreme runs to keep each call heavier)")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4,
                         help="llama.cpp thread count (default: all detected cores, for "
                              "maximum sustained load in extreme mode)")
    parser.add_argument("--graphs", action="store_true",
                         help="Generate CPU temp / tok-s / TTFT graphs after the run")

    parser.add_argument("--extreme", action="store_true",
                         help="Extreme Stress Test mode (validation run #2): sustained full "
                              "load with no breaks, duration derived from measured tau rather "
                              "than a fixed --duration, and a PUE/heat-rejection snapshot "
                              "written at the end.")
    parser.add_argument("--tau-source", type=str, default="",
                         help="Path to a tau_fit_summary.csv (from step_load_test.py + "
                              "thermal_system_id.py) used to derive the extreme-mode duration")
    parser.add_argument("--tau-select", choices=["slowest", "this-condition"], default="slowest",
                         help="'slowest' (default): use the slowest condition's tau across the "
                              "whole tau-source file, so every condition runs the same duration "
                              "and even the slowest reaches steady state. 'this-condition': only "
                              "use --condition's own tau.")
    parser.add_argument("--tau-seconds", type=float, default=None,
                         help="Directly supply a tau in seconds instead of reading --tau-source "
                              "(e.g. if you already know the slowest condition's tau)")
    parser.add_argument("--min-duration-min", type=float, default=60.0,
                         help="Floor for the extreme-mode duration (default: 60, per the test plan)")
    parser.add_argument("--tau-multiplier", type=float, default=5.0,
                         help="Duration = max(--min-duration-min, tau-multiplier * tau / 60) "
                              "(default multiplier: 5, per the test plan)")
    parser.add_argument("--stabilize-slope-check", type=float, default=0.05,
                         help="C/min threshold used only for the end-of-run steady-state "
                              "diagnostic in extreme mode (default: 0.05, same as step_load_test.py)")

    args = parser.parse_args()

    duration_min = args.duration
    if args.extreme:
        tau_s = args.tau_seconds
        if tau_s is None and args.tau_source:
            tau_s = load_max_tau_seconds(args.tau_source, select=args.tau_select, condition=args.condition)
        if tau_s is None:
            print(f"--extreme set but no --tau-seconds or --tau-source given -- "
                  f"falling back to --duration={args.duration} min. Pass one of those "
                  f"for a tau-derived duration per the test plan.")
        else:
            duration_min = max(args.min_duration_min, args.tau_multiplier * tau_s / 60.0)
            print(f"Extreme-mode duration derived from tau={tau_s:.1f}s "
                  f"({args.tau_select}): max({args.min_duration_min}, "
                  f"{args.tau_multiplier}*{tau_s:.1f}/60) = {duration_min:.1f} min")

    os.makedirs(args.output, exist_ok=True)
    run_benchmark(args.model, args.condition, duration_min, args.output, args)