#!/usr/bin/env python3
"""
Geothermal Cooling Demo — AI Performance Benchmark
----------------------------------------------------
Runs a small LLM (via llama.cpp) in a continuous loop while logging:
  - CPU temperature
  - CPU clock speed
  - Throttle status
  - Tokens per second

Saves all data to a timestamped CSV for analysis and dashboard display.

Usage:
  python3 benchmark.py --model path/to/model.gguf --condition geothermal --duration 20

Requirements:
  pip install llama-cpp-python
  A GGUF model file (e.g. TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf)
  Download from: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
"""

import subprocess
import csv
import time
import argparse
import os
from datetime import datetime

# ── Try to import llama-cpp-python ─────────────────────────────────────────
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("Warning: llama-cpp-python not installed. Running in CPU stress-only mode.")
    print("Install with: pip install llama-cpp-python")


# ── System metric helpers ───────────────────────────────────────────────────

def get_cpu_temp():
    """Read CPU temperature from vcgencmd. Returns float in Celsius."""
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True, text=True, timeout=2
        )
        # Output looks like: temp=52.1'C
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
        # Output looks like: frequency(48)=2400000000
        hz = int(result.stdout.strip().split("=")[1])
        return hz // 1_000_000  # Convert to MHz
    except Exception:
        return None


def get_throttle_status():
    """
    Read throttle flags from vcgencmd.
    Returns a dict with human-readable flags.
    Bit 0: currently under-voltage
    Bit 1: currently frequency capped (throttled)
    Bit 2: currently throttled
    Bit 3: currently soft temperature limit active
    """
    try:
        result = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True, text=True, timeout=2
        )
        # Output looks like: throttled=0x50000
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


# ── Prompt used for repeated LLM inference ─────────────────────────────────

STRESS_PROMPT = (
    "Explain in detail how geothermal heat exchange systems work, "
    "including the thermodynamic principles, the role of the ground loop, "
    "heat pumps, and why underground temperatures remain stable year-round. "
    "Include comparisons with conventional cooling systems."
)


# ── CSV logging ─────────────────────────────────────────────────────────────

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
    "tokens_per_sec",
    "tokens_generated",
    "inference_time_sec",
]


def write_row(writer, elapsed, condition, temp, clock, throttle, tps, tokens, inf_time):
    writer.writerow({
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "condition": condition,
        "cpu_temp_c": temp,
        "cpu_clock_mhz": clock,
        "throttled": int(throttle["throttled"]),
        "freq_capped": int(throttle["freq_capped"]),
        "soft_temp_limit": int(throttle["soft_temp_limit"]),
        "tokens_per_sec": round(tps, 2) if tps is not None else "",
        "tokens_generated": tokens if tokens is not None else "",
        "inference_time_sec": round(inf_time, 2) if inf_time is not None else "",
    })


# ── Live terminal display ───────────────────────────────────────────────────

def print_status(elapsed, duration, condition, temp, clock, throttle, tps):
    throttle_str = "YES ⚠" if throttle["throttled"] else "no"
    tps_str = f"{tps:.2f}" if tps is not None else "running..."
    print(
        f"\r[{elapsed:>5.0f}s / {duration * 60}s] "
        f"Condition: {condition:<12} | "
        f"Temp: {temp}°C | "
        f"Clock: {clock} MHz | "
        f"Throttled: {throttle_str:<6} | "
        f"Tok/s: {tps_str:<8}",
        end="", flush=True
    )


# ── Main benchmark loop ─────────────────────────────────────────────────────

def run_benchmark(model_path, condition, duration_min, output_dir):
    duration_sec = duration_min * 60
    csv_path = os.path.join(output_dir, get_csv_path(condition))

    print(f"\n{'='*70}")
    print(f"  Geothermal Cooling Demo — AI Performance Benchmark")
    print(f"{'='*70}")
    print(f"  Condition : {condition}")
    print(f"  Duration  : {duration_min} minutes")
    print(f"  Model     : {model_path if model_path else 'CPU stress mode (no model)'}")
    print(f"  Output    : {csv_path}")
    print(f"{'='*70}\n")

    # Load model
    llm = None
    if LLAMA_AVAILABLE and model_path and os.path.exists(model_path):
        print("Loading model... (this may take 30-60 seconds on Pi 5)")
        llm = Llama(
            model_path=model_path,
            n_ctx=512,        # small context to keep memory usage low
            n_threads=4,      # use all 4 Pi 5 cores
            verbose=False,
        )
        print("Model loaded. Starting benchmark.\n")
    elif model_path:
        print(f"Model file not found at: {model_path}")
        print("Running in CPU stress mode instead.\n")

    start_time = time.time()
    inference_count = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()

        print("Running... (Ctrl+C to stop early)\n")

        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed >= duration_sec:
                    break

                # Read system metrics before inference
                temp = get_cpu_temp()
                clock = get_cpu_clock()
                throttle = get_throttle_status()

                # Run inference
                tps = None
                tokens = None
                inf_time = None

                if llm is not None:
                    inf_start = time.time()
                    response = llm(
                        STRESS_PROMPT,
                        max_tokens=150,
                        temperature=0.7,
                        echo=False,
                    )
                    inf_time = time.time() - inf_start
                    tokens = response["usage"]["completion_tokens"]
                    tps = tokens / inf_time if inf_time > 0 else 0
                    inference_count += 1
                else:
                    # CPU stress fallback: busy loop for 5 seconds
                    stress_start = time.time()
                    while time.time() - stress_start < 5:
                        _ = sum(i * i for i in range(100_000))
                    inf_time = time.time() - stress_start

                # Log to CSV
                write_row(
                    writer, elapsed, condition,
                    temp, clock, throttle,
                    tps, tokens, inf_time
                )
                f.flush()

                # Print live status
                print_status(elapsed, duration_min, condition, temp, clock, throttle, tps)

        except KeyboardInterrupt:
            print("\n\nStopped early by user.")

    elapsed_total = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"  Benchmark complete.")
    print(f"  Total time     : {elapsed_total:.0f} seconds")
    print(f"  Inferences run : {inference_count}")
    print(f"  Data saved to  : {csv_path}")
    print(f"{'='*70}\n")

    return csv_path


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Geothermal demo AI performance benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to GGUF model file (e.g. tinyllama.gguf)"
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["no_cooling", "fan", "geothermal"],
        required=True,
        help="Cooling condition being tested"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=20,
        help="Trial duration in minutes (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Directory to save CSV output (default: current directory)"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    run_benchmark(args.model, args.condition, args.duration, args.output)
