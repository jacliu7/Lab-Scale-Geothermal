#!/usr/bin/env python3
"""
Geothermal Cooling Demo - Live Dashboard

Reads dashboard_status.json (written every loop iteration by benchmark.py)
and renders it as a live Streamlit page: CPU temp, loop inlet/outlet temp,
elapsed time, COP/PUE, tokens/sec, latency, plus a pump speed slider that
writes commands back to benchmark.py via pump_command.json.

Run this in a separate terminal/session while benchmark.py is running:
  streamlit run streamlit_dashboard.py

Requirements:
  pip install streamlit pandas
  pip install streamlit-autorefresh   (optional but recommended, see below)
"""

import json
import os
import time
import streamlit as st
import pandas as pd

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False


st.set_page_config(page_title="Geothermal Cooling Demo", layout="wide")

# Sidebar: point this at the same --output directory you passed to benchmark.py
st.sidebar.header("Settings")
data_dir = st.sidebar.text_input("Benchmark output directory", value=".")
refresh_ms = st.sidebar.slider("Refresh interval (ms)", 500, 5000, 1500, step=250)
status_path = os.path.join(data_dir, "dashboard_status.json")
command_path = os.path.join(data_dir, "pump_command.json")

if AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=refresh_ms, key="refresh")
else:
    st.sidebar.warning(
        "Install streamlit-autorefresh for automatic live updates:\n"
        "pip install streamlit-autorefresh\n\n"
        "Without it, use the Refresh button below."
    )
    if st.sidebar.button("Refresh now"):
        pass

st.title("Geothermal Cooling Demo - Live Dashboard")

# History buffers persist across reruns within this browser session
if "history" not in st.session_state:
    st.session_state.history = []

MAX_HISTORY_POINTS = 1200  # ~20 min at 1s resolution with some headroom


def load_status():
    try:
        with open(status_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def send_pump_command(duty_pct):
    tmp_path = command_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"pump_duty_pct": duty_pct}, f)
    os.replace(tmp_path, command_path)


status = load_status()

if status is None:
    st.info(
        f"Waiting for benchmark.py to write status to:\n\n`{status_path}`\n\n"
        "Start (or check the --output path on) benchmark.py, then this page "
        "will populate automatically."
    )
    st.stop()

# Append to history (only if elapsed time actually advanced, to avoid
# duplicate points when the file hasn't updated between refreshes yet)
last_elapsed = st.session_state.history[-1]["elapsed_sec"] if st.session_state.history else None
if status.get("elapsed_sec") != last_elapsed:
    st.session_state.history.append(status)
    if len(st.session_state.history) > MAX_HISTORY_POINTS:
        st.session_state.history = st.session_state.history[-MAX_HISTORY_POINTS:]

df = pd.DataFrame(st.session_state.history)

# Top: run progress
elapsed = status.get("elapsed_sec", 0) or 0
duration = status.get("duration_sec", 1) or 1
progress = min(1.0, elapsed / duration)
st.progress(progress, text=f"{status.get('condition', '?')} - {elapsed:.0f}s / {duration:.0f}s")

# Pump control
st.subheader("Pump Control")
current_duty = status.get("pump_duty_pct", 0) or 0
new_duty = st.slider("Pump duty cycle (%)", 0, 100, int(current_duty), key="pump_slider")
if new_duty != int(current_duty):
    send_pump_command(new_duty)
    st.caption(f"Command sent: {new_duty}% (benchmark.py applies this on its next loop tick)")

# Live metrics
st.subheader("System Status")
c1, c2, c3, c4 = st.columns(4)
c1.metric("CPU Temp", f"{status.get('cpu_temp_c', '--')} C",
           delta=None if not status.get("throttled") else "THROTTLED")
c2.metric("CPU Clock", f"{status.get('cpu_clock_mhz', '--')} MHz")
c3.metric("Flow Rate", f"{status.get('flow_rate_lph', 0):.1f} L/hr" if status.get("flow_rate_lph") is not None else "--")
c4.metric("Pump Duty", f"{status.get('pump_duty_pct', '--')}%")

st.subheader("Water Loop")
w1, w2, w3, w4 = st.columns(4)
t_in = status.get("t_in_c")
t_out = status.get("t_out_c")
delta_t = (t_in - t_out) if (t_in is not None and t_out is not None) else None
w1.metric("Inlet Temp (T_in)", f"{t_in:.1f} C" if t_in is not None else "n/a")
w2.metric("Outlet Temp (T_out)", f"{t_out:.1f} C" if t_out is not None else "n/a")
w3.metric("Delta T", f"{delta_t:.1f} C" if delta_t is not None else "n/a")
w4.metric("Heat Removed", f"{status.get('heat_removed_w', 0):.2f} W" if status.get("heat_removed_w") is not None else "n/a")

st.subheader("Efficiency Metrics")
e1, e2, e3, e4 = st.columns(4)
e1.metric("COP", f"{status.get('cop'):.2f}" if status.get("cop") is not None else "n/a")
e2.metric("PUE", f"{status.get('pue'):.3f}" if status.get("pue") is not None else "n/a")
e3.metric("Pi Power", f"{status.get('pi_power_w', 0):.2f} W" if status.get("pi_power_w") is not None else "n/a")
e4.metric("Pump Power", f"{status.get('pump_power_w', 0):.3f} W" if status.get("pump_power_w") is not None else "n/a")

st.subheader("AI Performance")
a1, a2 = st.columns(2)
a1.metric("Tokens/sec", f"{status.get('tokens_per_sec'):.2f}" if status.get("tokens_per_sec") is not None else "n/a")
a2.metric("Time to First Token", f"{status.get('time_to_first_token_sec'):.3f} s" if status.get("time_to_first_token_sec") is not None else "n/a")

# Live charts
if len(df) > 1:
    st.subheader("CPU Temperature over Time")
    st.line_chart(df.set_index("elapsed_sec")[["cpu_temp_c"]])

    st.subheader("COP / PUE over Time")
    cop_pue_cols = [c for c in ["cop", "pue"] if c in df.columns]
    if cop_pue_cols:
        st.line_chart(df.set_index("elapsed_sec")[cop_pue_cols])

    if "tokens_per_sec" in df.columns and df["tokens_per_sec"].notna().any():
        st.subheader("Tokens/sec over Time")
        st.line_chart(df.set_index("elapsed_sec")[["tokens_per_sec"]])

    if "time_to_first_token_sec" in df.columns and df["time_to_first_token_sec"].notna().any():
        st.subheader("Time to First Token over Time")
        st.line_chart(df.set_index("elapsed_sec")[["time_to_first_token_sec"]])
else:
    st.caption("Charts will populate once a few data points have come in.")

if not AUTOREFRESH_AVAILABLE:
    time.sleep(0.1)