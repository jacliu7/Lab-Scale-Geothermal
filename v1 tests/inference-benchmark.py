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
    default, change PUMP_PWM_PIN below if wired differently). gpiozero
    picks the correct backend automatically, including on Pi 5's RP1 I/O
    chip, which is why this script doesn't use RPi.GPIO directly.
  - Pi power draw is read from the PMIC via "vcgencmd pmic_read_adc",
    which is Pi 5 specific. On other boards this will return None and
    pi_power_w / pue will be logged blank.
"""

import subprocess
import csv
import time
import argparse
import os
import json
import glob
from datetime import datetime

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("Warning: llama-cpp-python not installed. Running in CPU stress-only mode.")
    print("Install with: pip install llama-cpp-python")

try:
    from gpiozero import PWMOutputDevice
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: gpiozero not installed. Pump PWM control unavailable.")
    print("Install with: pip install gpiozero")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Constants (from the build spec)

SPECIFIC_HEAT_WATER = 4186.0   # J/(kg*C)
WATER_DENSITY = 1.0            # kg/L, close enough for distilled water near room temp
MAX_PUMP_FLOW_LPH = 100.0      # L/hr at 100% duty, per pump spec sheet
PUMP_VOLTAGE_NOMINAL = 5.0     # V, mid of the 3-5V spec range
PUMP_CURRENT_MIN_A = 0.10      # A, 100 mA
PUMP_CURRENT_MAX_A = 0.20      # A, 200 mA
GROUND_TEMP_F = 55.0
MAX_SAFE_TEMP_F = 185.0
MAX_SAFE_TEMP_C = (MAX_SAFE_TEMP_F - 32.0) * 5.0 / 9.0   # ~85.0 C, matches the Pi's own throttle point
PUMP_PWM_PIN = 18              # BCM numbering, change to match wiring
PUMP_PWM_FREQ_HZ = 1000


# System metric helpers

def get_cpu_temp():
    """Read CPU temperature from vcgencmd. Returns float in Celsius."""
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True, text=True, timeout=2
        )
        return float(result.stdout.strip().split("=")[1].replace("'C", ""))
    except Exception:
        return None


def get_cpu_clock():
    """Read ARM CPU clock speed in MHz."""
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


def _read_ds18b20(sensor_id):
    """Read a single DS18B20 1-wire temp sensor by its device folder name."""
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


def get_water_temps(t_in_id, t_out_id):
    """Return (t_in_c, t_out_c) from the loop inlet/outlet DS18B20 sensors."""
    t_in = _read_ds18b20(t_in_id) if t_in_id else None
    t_out = _read_ds18b20(t_out_id) if t_out_id else None
    return t_in, t_out


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
    pump's rated max flow. Recalibrate against a measured flow curve if you
    get a chance to clock actual L/hr at a few duty cycle points."""
    return MAX_PUMP_FLOW_LPH * (duty_pct / 100.0)


def compute_heat_removed_w(flow_rate_lph, t_in_c, t_out_c):
    """Heat_removed = flow_rate * specific_heat * (T_in - T_out), in watts."""
    if flow_rate_lph is None or t_in_c is None or t_out_c is None:
        return None
    flow_kg_s = (flow_rate_lph * WATER_DENSITY) / 3600.0
    delta_t = t_in_c - t_out_c
    return round(flow_kg_s * SPECIFIC_HEAT_WATER * delta_t, 2)


def compute_pump_power_w(voltage=PUMP_VOLTAGE_NOMINAL, current_a=None):
    """Pump electrical power, P = V * I. Falls back to the midpoint of the
    spec sheet current range (100-200 mA) if you haven't measured it directly."""
    if current_a is None:
        current_a = (PUMP_CURRENT_MIN_A + PUMP_CURRENT_MAX_A) / 2.0
    return round(voltage * current_a, 3)


def compute_cop(heat_removed_w, work_input_w):
    """COP = heat removed / work input (pump power). Higher means more cooling per watt spent."""
    if not heat_removed_w or not work_input_w:
        return None
    return round(heat_removed_w / work_input_w, 3)


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


# Prompt used for repeated LLM inference (fixed size, per the constants list)

STRESS_PROMPT = (
    "Explain in detail how geothermal heat exchange systems work, "
    "including the thermodynamic principles, the role of the ground loop, "
    "heat pumps, and why underground temperatures remain stable year-round. "
    "Include comparisons with conventional cooling systems."
)


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


def write_live_status(path, status):
    """Write the latest metrics snapshot to a JSON file a dashboard can poll.
    Writes to a temp file then swaps it in so the dashboard never reads a
    half-written file mid-update."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(status, f)
    os.replace(tmp_path, path)


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
    print(f"  Geothermal Cooling Demo - AI Performance Benchmark")
    print(f"{'='*70}")
    print(f"  Condition   : {condition}")
    print(f"  Duration    : {duration_min} minutes")
    print(f"  Model       : {model_path if model_path else 'CPU stress mode (no model)'}")
    print(f"  Pump duty   : {args.pump_duty}%")
    print(f"  Safety limit: {MAX_SAFE_TEMP_F}F ({MAX_SAFE_TEMP_C:.1f}C)")
    print(f"  Output      : {csv_path}")
    print(f"{'='*70}\n")

    if args.lock_clock:
        lock_cpu_clock()

    if not args.t_in_sensor or not args.t_out_sensor:
        found = list_available_temp_sensors()
        if found:
            print(f"1-wire sensors detected: {found}")
            print("Pass --t-in-sensor and --t-out-sensor with these IDs to log loop temps.\n")
        else:
            print("No 1-wire DS18B20 sensors detected. Loop temps will be logged blank.\n")

    pump_ready = init_pump(args.pump_duty)
    if not pump_ready:
        print("GPIO/pump control unavailable, pump_duty_pct will be logged but not actuated.\n")

    llm = None
    if LLAMA_AVAILABLE and model_path and os.path.exists(model_path):
        print("Loading model... (this may take 30-60 seconds on Pi 5)")
        llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=4,
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

                    new_duty, last_cmd_mtime = read_pump_command(command_path, last_cmd_mtime)
                    if new_duty is not None:
                        current_pump_duty = set_pump_speed(new_duty)
                        print(f"\nPump duty updated to {current_pump_duty}% via dashboard.")

                    temp = get_cpu_temp()
                    clock = get_cpu_clock()
                    throttle = get_throttle_status()
                    t_in, t_out = get_water_temps(args.t_in_sensor, args.t_out_sensor)

                    if temp is not None and temp >= MAX_SAFE_TEMP_C:
                        print(f"\n\nSAFETY STOP: CPU temp {temp}C reached the "
                              f"{MAX_SAFE_TEMP_F}F limit. Ending run early.")
                        break

                    flow_rate = duty_to_flow_rate_lph(current_pump_duty)
                    heat_removed = compute_heat_removed_w(flow_rate, t_in, t_out)
                    pump_power = compute_pump_power_w(current_a=args.pump_current)
                    pi_power = get_pi_power_w()
                    cop = compute_cop(heat_removed, pump_power)
                    pue = compute_pue(pi_power, pump_power)

                    if llm is not None:
                        tokens, tps, inf_time, ttft, pe_time = run_inference(
                            llm, STRESS_PROMPT, args.max_tokens
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

                    write_live_status(status_path, {
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
        stop_pump()

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

    return csv_path


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Geothermal demo AI performance benchmark"
    )
    parser.add_argument("--model", type=str, default="",
                         help="Path to GGUF model file (e.g. tinyllama.gguf)")
    parser.add_argument("--condition", type=str,
                         choices=["no_cooling", "fan", "geothermal"], required=True,
                         help="Cooling condition being tested")
    parser.add_argument("--duration", type=int, default=20,
                         help="Trial duration in minutes (default: 20)")
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
                         help="Fixed output token count per inference call (default: 150)")
    parser.add_argument("--graphs", action="store_true",
                         help="Generate CPU temp / tok-s / TTFT graphs after the run")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    run_benchmark(args.model, args.condition, args.duration, args.output, args)