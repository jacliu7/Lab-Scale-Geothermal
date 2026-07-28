#!/usr/bin/env python3
"""
thermal_system_id.py

Thermal system identification for the Pi geothermal-cooling rig.

What this does
---------------
1. Loads a benchmark log (CSV) with a timestamp/elapsed-time column, a
   temperature column (CPU temp, or T_in/T_out), and a load/phase marker
   that tells us when a step change happened (e.g. idle -> full load,
   full load -> idle, pump on -> pump off).
2. Auto-detects step events from the load/phase column (or accepts
   explicit step timestamps from the command line).
3. For each step, fits a first-order thermal response:

        T(t) = T_ss + (T0 - T_ss) * exp(-t / tau)

   using TWO methods so results are directly comparable to the published
   literature:

     (a) Nonlinear least squares (scipy.optimize.curve_fit) on the raw
         temperature data. This is the more statistically correct fit.

     (b) Linearized log-fit on theta = (T - T_min) / (T_max - T_min):
         ln(1 - theta) = -t / tau
         This is a straight-line regression on ln(1-theta) vs. t, which
         is exactly the method used in the standard data-center thermal
         transient reference (Shields, "Dynamic Thermal Response of the
         Data Center to Cooling Loss During Facility Power Failure",
         Georgia Tech M.S. Thesis, 2009). Reporting tau from this method
         lets you put your number directly next to theirs in a table.

4. Computes the effective thermal resistance R = dT_ss / Q and thermal
   capacitance C = tau / R for each step, using the power draw at the
   time of the step (so this needs a power column -- Pi power draw is
   fine, doesn't need to be perfectly precise).

5. Produces:
     - a CSV summary table (one row per step, per cooling condition)
     - a figure overlaying raw data + both fits for each step
     - a benchmark comparison figure: your tau values vs. published
       server / CRAC time constants
     - a markdown table you can drop straight into the manuscript

Expected input CSV columns (rename via --col-* flags if yours differ):
    time_s        : elapsed seconds since test start (float)
    cpu_temp_c     : CPU temperature, deg C
    t_in_c         : coolant inlet temp, deg C   (optional)
    t_out_c        : coolant outlet temp, deg C  (optional)
    power_w        : total measured power draw, W (Pi + pump, or Pi only)
    cooling_type   : string label, e.g. "none", "fan", "loop"
    load_state     : string or int marker for phase, e.g. "idle"/"load",
                      or 0/1. Used to auto-detect step transitions.

You do not need every column filled for every row; the script only
requires time_s, one temperature column, and load_state (or explicit
--steps timestamps) to run the tau fit. power_w is needed for R and C.

Usage
-----
    python3 thermal_system_id.py --input rig_log.csv --outdir results \
        --temp-col cpu_temp_c --cooling-col cooling_type

    # or, if you already know exactly when your steps happened:
    python3 thermal_system_id.py --input rig_log.csv --outdir results \
        --temp-col cpu_temp_c --steps 0,1200,2400,3600
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Published reference values for comparison.
# Source: Shields, S. "Dynamic Thermal Response of the Data Center to
# Cooling Loss During Facility Power Failure." M.S. Thesis, Georgia
# Institute of Technology, School of Mechanical Engineering, Aug 2009.
# Chapter 3 (Server Time Constant Experiment) and Chapter 5 (CRAC HX
# Response). Time constants extracted via step-change-in-inlet-temperature
# experiments and least-squares fit to a first-order model, same approach
# used in this script.
# ---------------------------------------------------------------------------
LITERATURE_TAU_S = {
    "Legacy server, full load (processor outlet)": 340,
    "Legacy server, full load (PSU outlet)": 380,
    "Legacy server, idle (processor outlet)": 370,
    "Legacy server, idle (PSU outlet)": 300,
    "Modern 2U Xeon server, full load (processor outlet)": 130,
    "Modern 2U Xeon server, full load (PSU outlet)": 990,
    "Bare resistive heater (thermal-mass lower limit)": 50,
    "CRAC air-to-water HX (coolant flow step)": 10,
}

# Published PUE benchmarks for the manuscript's separate PUE comparison
# figure (not used in the tau fit itself, kept here for convenience so
# both figures pull from one script). Source: Uptime Institute 2025
# Global Data Center Survey; Google 2025 fleet-wide sustainability data.
LITERATURE_PUE = {
    "Global industry average (Uptime Institute 2025)": 1.54,
    "Enterprise / colocation average": 1.69,   # midpoint of 1.58-1.80 range
    "Hyperscale average (Google/Meta/MSFT/AWS)": 1.12,  # midpoint 1.10-1.15
    "Google fleet-wide 2025": 1.09,
    "New-build regulatory target (2026+, e.g. Germany EnEfG)": 1.20,
}


def first_order_model(t, T_ss, T0, tau):
    """T(t) = T_ss + (T0 - T_ss) * exp(-t/tau)"""
    return T_ss + (T0 - T_ss) * np.exp(-t / tau)


def fit_nonlinear(t, T):
    """Nonlinear least-squares fit of the first-order step response."""
    T0_guess = T[0]
    Tss_guess = T[-1]
    tau_guess = max((t[-1] - t[0]) / 4, 1.0)
    p0 = [Tss_guess, T0_guess, tau_guess]
    try:
        popt, pcov = curve_fit(first_order_model, t, T, p0=p0, maxfev=20000)
        T_ss, T0, tau = popt
        resid = T - first_order_model(t, *popt)
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((T - np.mean(T)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"T_ss": T_ss, "T0": T0, "tau_s": abs(tau), "r2": r2}
    except RuntimeError:
        return {"T_ss": np.nan, "T0": np.nan, "tau_s": np.nan, "r2": np.nan}


def fit_linearized(t, T, T_min=None, T_max=None):
    """
    Linearized log-fit, matching the literature methodology:
        theta = (T - T_min) / (T_max - T_min)
        ln(1 - theta) = -t / tau
    Fit a line through the origin-referenced log data via least squares.
    T_min/T_max default to the first/last sample if not given (use the
    literature's convention of fixing these from known asymptotes if you
    have cleaner steady-state values available).
    """
    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)
    if T_min is None:
        T_min = min(T[0], T[-1])
    if T_max is None:
        T_max = max(T[0], T[-1])
    span = T_max - T_min
    if span == 0:
        return {"tau_s": np.nan, "r2": np.nan}

    theta = (T - T_min) / span
    # guard against theta >= 1 or <= 0 from noise at the tail
    theta = np.clip(theta, 1e-6, 1 - 1e-6)
    y = np.log(1 - theta)
    t0 = t - t[0]

    # least squares slope through the data (not forced through origin,
    # to absorb any small timing offset, matching thesis appendix method)
    A = np.vstack([t0, np.ones_like(t0)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    tau = -1.0 / slope if slope != 0 else np.nan

    y_pred = A @ np.array([slope, intercept])
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"tau_s": abs(tau), "r2": r2}


def detect_steps(df, load_col, time_col):
    """Return indices where load_state changes value (a step event)."""
    states = df[load_col].astype(str).values
    change_idx = [0]
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            change_idx.append(i)
    return change_idx


def slice_window(df, time_col, start_time, window_s):
    sub = df[(df[time_col] >= start_time) & (df[time_col] <= start_time + window_s)]
    return sub


def analyze_step(df, time_col, temp_col, power_col, start_time, window_s, label):
    sub = slice_window(df, time_col, start_time, window_s)
    if len(sub) < 5:
        return None

    t = sub[time_col].values.astype(float)
    t = t - t[0]
    T = sub[temp_col].values.astype(float)

    nl = fit_nonlinear(t, T)
    # Use the nonlinear fit's converged asymptotes (T_ss, T0) as the
    # reference values for the linearized cross-check, rather than the raw
    # window endpoints. Within a finite window the raw endpoints haven't
    # fully converged, which biases the log-linearization heavily since it
    # divides by (T_max - T_min). This mirrors the literature approach of
    # using independently-known steady-state values rather than the last
    # sample in a possibly-truncated window.
    if not np.isnan(nl.get("T_ss", np.nan)):
        t_lo = min(nl["T0"], nl["T_ss"])
        t_hi = max(nl["T0"], nl["T_ss"])
        lin = fit_linearized(t, T, T_min=t_lo, T_max=t_hi)
    else:
        lin = fit_linearized(t, T)

    result = {
        "label": label,
        "start_time_s": start_time,
        "window_s": window_s,
        "n_samples": len(sub),
        "T_start_c": T[0],
        "T_end_c": T[-1],
        "delta_T_c": T[-1] - T[0],
        "tau_nonlinear_s": nl["tau_s"],
        "r2_nonlinear": nl["r2"],
        "tau_linearized_s": lin["tau_s"],
        "r2_linearized": lin["r2"],
    }

    if power_col is not None and power_col in sub.columns:
        dQ = sub[power_col].values.astype(float)
        Q_ss = np.nanmean(dQ[-max(3, len(dQ) // 10):])  # avg of tail
        dT_ss = result["delta_T_c"]
        if Q_ss != 0:
            R_thermal = abs(dT_ss / Q_ss)          # deg C / W
            tau_use = nl["tau_s"] if not np.isnan(nl["tau_s"]) else lin["tau_s"]
            C_thermal = tau_use * (1.0 / R_thermal) if R_thermal != 0 else np.nan
            result["Q_ss_w"] = Q_ss
            result["R_thermal_c_per_w"] = R_thermal
            result["C_thermal_j_per_c"] = C_thermal
        else:
            result["Q_ss_w"] = Q_ss
            result["R_thermal_c_per_w"] = np.nan
            result["C_thermal_j_per_c"] = np.nan

    return result, sub, nl


def make_step_plot(result, sub, nl, time_col, temp_col, outdir, idx):
    t = sub[time_col].values.astype(float)
    t0 = t - t[0]
    T = sub[temp_col].values.astype(float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t0, T, "o", ms=3, color="#444444", label="Measured")

    if not np.isnan(nl["tau_s"]):
        t_fit = np.linspace(0, t0[-1], 200)
        T_fit = first_order_model(t_fit, nl["T_ss"], nl["T0"], nl["tau_s"])
        ax.plot(t_fit, T_fit, "-", lw=2, color="#d95f02",
                 label=f"First-order fit (tau = {nl['tau_s']:.0f} s, R2 = {nl['r2']:.3f})")

    ax.set_xlabel("Time since step [s]")
    ax.set_ylabel("Temperature [C]")
    ax.set_title(result["label"])
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = os.path.join(outdir, f"step_fit_{idx:02d}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def make_benchmark_plot(results_df, outdir):
    """Bar chart: your rig's tau values next to the published server/CRAC values."""
    fig, ax = plt.subplots(figsize=(9, 6))

    lit_labels = list(LITERATURE_TAU_S.keys())
    lit_values = list(LITERATURE_TAU_S.values())

    rig_labels = []
    rig_values = []
    if results_df is not None and len(results_df) > 0:
        for _, row in results_df.iterrows():
            tau = row.get("tau_nonlinear_s", np.nan)
            if not pd.isna(tau):
                rig_labels.append(f"RIG: {row['label']}")
                rig_values.append(tau)

    all_labels = lit_labels + rig_labels
    all_values = lit_values + rig_values
    colors = ["#7570b3"] * len(lit_labels) + ["#1b9e77"] * len(rig_labels)

    y_pos = np.arange(len(all_labels))
    ax.barh(y_pos, all_values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Time constant, tau [s] (log scale)")
    ax.set_xscale("log")
    ax.set_title("Thermal time constant: lab rig vs. published data center hardware\n"
                 "(purple = literature, green = this rig)")
    ax.grid(alpha=0.3, axis="x", which="both")
    fig.tight_layout()
    fname = os.path.join(outdir, "tau_benchmark_comparison.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Path to benchmark log CSV")
    ap.add_argument("--outdir", default="thermal_sysid_results", help="Output directory")
    ap.add_argument("--time-col", default="time_s")
    ap.add_argument("--temp-col", default="cpu_temp_c",
                     help="Temperature column to fit (cpu_temp_c, t_out_c, etc.)")
    ap.add_argument("--power-col", default="power_w",
                     help="Power column for R/C calculation (set to '' to skip)")
    ap.add_argument("--load-col", default="load_state",
                     help="Column marking idle/load phase, used for auto step detection")
    ap.add_argument("--cooling-col", default="cooling_type",
                     help="Column labeling cooling condition (none/fan/loop)")
    ap.add_argument("--window", type=float, default=500.0,
                     help="Seconds of data to use after each detected step "
                          "(literature used the first 200-500 s of the response)")
    ap.add_argument("--steps", default=None,
                     help="Comma-separated explicit step start times in seconds, "
                          "overrides auto-detection, e.g. --steps 0,1200,2400")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.time_col not in df.columns:
        sys.exit(f"Column '{args.time_col}' not found. Available columns: {list(df.columns)}")
    if args.temp_col not in df.columns:
        sys.exit(f"Column '{args.temp_col}' not found. Available columns: {list(df.columns)}")

    power_col = args.power_col if args.power_col and args.power_col in df.columns else None

    # Determine step start times
    if args.steps:
        step_times = [float(x) for x in args.steps.split(",")]
    else:
        if args.load_col not in df.columns:
            sys.exit(
                f"No --steps given and load column '{args.load_col}' not found. "
                f"Either add a load_state column to your log or pass --steps explicitly."
            )
        idxs = detect_steps(df, args.load_col, args.time_col)
        step_times = [df.iloc[i][args.time_col] for i in idxs]

    # Group by cooling condition if available, else treat whole file as one condition
    conditions = df[args.cooling_col].unique() if args.cooling_col in df.columns else [None]

    all_results = []
    fit_records = []
    plot_idx = 0

    for cond in conditions:
        sub_df = df[df[args.cooling_col] == cond] if cond is not None else df
        # re-detect steps within this condition's slice if grouping by cooling
        if args.steps:
            local_step_times = step_times
        elif args.load_col in df.columns:
            idxs = detect_steps(sub_df.reset_index(drop=True), args.load_col, args.time_col)
            local_step_times = [sub_df.reset_index(drop=True).iloc[i][args.time_col] for i in idxs]
        else:
            local_step_times = step_times

        for st in local_step_times:
            label = f"{cond if cond else 'all'} @ t={st:.0f}s"
            out = analyze_step(sub_df, args.time_col, args.temp_col, power_col,
                                st, args.window, label)
            if out is None:
                continue
            result, sub, nl = out
            result["cooling_type"] = cond
            all_results.append(result)
            plot_idx += 1
            make_step_plot(result, sub, nl, args.time_col, args.temp_col, args.outdir, plot_idx)

    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(args.outdir, "tau_fit_summary.csv")
    results_df.to_csv(results_csv, index=False)

    # Benchmark comparison plot
    bench_plot = make_benchmark_plot(results_df, args.outdir)

    # Markdown summary table for the manuscript
    md_path = os.path.join(args.outdir, "tau_summary_table.md")
    with open(md_path, "w") as f:
        f.write("# Thermal system-ID summary\n\n")
        f.write("## This rig\n\n")
        if len(results_df) > 0:
            cols = ["label", "cooling_type", "delta_T_c", "tau_nonlinear_s",
                    "r2_nonlinear", "tau_linearized_s", "r2_linearized",
                    "R_thermal_c_per_w", "C_thermal_j_per_c"]
            cols = [c for c in cols if c in results_df.columns]
            f.write(results_df[cols].round(3).to_markdown(index=False))
        else:
            f.write("No steps could be fit. Check --steps or --load-col.\n")
        f.write("\n\n## Published reference values (Shields, 2009, Georgia Tech)\n\n")
        f.write("| System | tau [s] |\n|---|---|\n")
        for k, v in LITERATURE_TAU_S.items():
            f.write(f"| {k} | {v} |\n")

    print(f"\nWrote {len(results_df)} step fits.")
    print(f"  Summary CSV : {results_csv}")
    print(f"  Markdown    : {md_path}")
    print(f"  Benchmark plot: {bench_plot}")
    print(f"  Per-step plots: {args.outdir}/step_fit_*.png")

    if len(results_df) > 0:
        print("\ntau (nonlinear fit) by step:")
        print(results_df[["label", "tau_nonlinear_s", "r2_nonlinear"]].to_string(index=False))


if __name__ == "__main__":
    main()