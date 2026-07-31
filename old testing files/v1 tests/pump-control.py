#!/usr/bin/env python3
"""
Pump Flow Rate Control, Raspberry Pi version
Converted from an Arduino MOSFET pump control sketch.
Drives a MOSFET-gated pump with PWM and gives you a touch/mouse GUI
to set flow rate on the Pi's own screen. Also includes a synthetic
CPU stress test (pure Python, no Ollama needed) with a live temp/load
graph, so you can load the CPU and watch how well the cooling loop
holds the temperature down at a given pump flow rate.

Hardware notes:
- PUMP_PIN below is a BCM GPIO number, not a physical pin number.
- GPIO18 is used because it supports hardware PWM, which gives a
  cleaner, more stable signal than software PWM on other pins.
- Keep a flyback diode across the pump terminals if it doesn't
  already have one built in, since a pump motor is an inductive load.
- Gate resistor (100 to 220 ohm) between this pin and the MOSFET gate,
  and a pulldown resistor (10k) from gate to ground, is good practice
  so the pump doesn't twitch on during boot before the script runs.

Extra dependencies for the stress test and graph:
    pip install matplotlib psutil
"""

import multiprocessing
import os
import time
from collections import deque

import tkinter as tk
from tkinter import ttk

import psutil
from gpiozero import PWMOutputDevice
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- Hardware setup ---
PUMP_PIN = 18       # BCM numbering. Change if you wired it elsewhere.
PWM_FREQ_HZ = 1000  # 1 kHz is a safe default for most MOSFET pump drivers

pump = PWMOutputDevice(PUMP_PIN, frequency=PWM_FREQ_HZ)

THERMAL_ZONE = "/sys/class/thermal/thermal_zone0/temp"
GRAPH_WINDOW_SECONDS = 300   # keep the last 5 minutes on screen
SAMPLE_INTERVAL_MS = 500    # refresh twice a second for a live feel


def read_cpu_temp():
    """Reads the SoC temperature in Celsius straight from the kernel."""
    try:
        with open(THERMAL_ZONE) as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError):
        return float("nan")


def cpu_stress_worker():
    """
    Pure Python busy loop, no external libraries, no Ollama, no GPU.
    One of these gets spawned per CPU core so it actually loads all of
    them rather than just pegging one thread. Runs until the parent
    process terminates it, that's the only exit condition.
    """
    while True:
        x = 0
        for i in range(1_000_000):
            x += i * i


class PumpControlApp:
    def __init__(self, root):
        self.root = root
        root.title("Pump Flow Control")
        root.geometry("460x760")

        self.running = False
        # Your original sketch used analogWrite(pumpPin, 50), which is
        # 50/255, about 19.6%, even though the comment said 25%.
        # Default here is set to a clean 25% instead.
        self.flow_percent = tk.IntVar(value=25)

        tk.Label(root, text="Pump Flow Rate Control",
                 font=("Helvetica", 18, "bold")).pack(pady=10)

        self.status_label = tk.Label(root, text="Status: OFF",
                                      font=("Helvetica", 14), fg="red")
        self.status_label.pack(pady=5)

        self.percent_label = tk.Label(
            root, text=f"Flow Rate: {self.flow_percent.get()}%",
            font=("Helvetica", 14))
        self.percent_label.pack(pady=5)

        self.slider = tk.Scale(
            root, from_=0, to=100, orient=tk.HORIZONTAL, length=320,
            variable=self.flow_percent, command=self.on_slider_change)
        self.slider.pack(pady=10)

        preset_frame = tk.Frame(root)
        preset_frame.pack(pady=5)
        for pct in (25, 50, 75, 100):
            tk.Button(preset_frame, text=f"{pct}%", width=6,
                      command=lambda p=pct: self.set_preset(p)).pack(
                side=tk.LEFT, padx=4)

        self.toggle_btn = tk.Button(
            root, text="Start Pump", width=20, height=2,
            bg="#4CAF50", fg="white", command=self.toggle_pump)
        self.toggle_btn.pack(pady=15)

        # --- CPU stress test section ---
        ttk.Separator(root, orient="horizontal").pack(fill="x", pady=5)

        tk.Label(root, text="CPU Stress Test", font=("Helvetica", 16, "bold")).pack(pady=5)

        self.temp_label = tk.Label(root, text="CPU Temp: -- degC", font=("Helvetica", 14))
        self.temp_label.pack()

        self.load_label = tk.Label(root, text="CPU Load: -- %", font=("Helvetica", 12))
        self.load_label.pack()

        self.stress_status = tk.Label(root, text="Stress: OFF", font=("Helvetica", 11), fg="gray")
        self.stress_status.pack(pady=2)

        self.stress_btn = tk.Button(
            root, text="Start CPU Stress Test", width=22, height=2,
            bg="#2196F3", fg="white", command=self.toggle_stress)
        self.stress_btn.pack(pady=8)

        self.stress_running = False
        self.stress_processes = []

        # Rolling buffers for the graph. Capped so old points fall off
        # rather than the plot growing forever on a long test run.
        self.start_time = time.time()
        max_points = int(GRAPH_WINDOW_SECONDS * 1000 / SAMPLE_INTERVAL_MS)
        self.time_data = deque(maxlen=max_points)
        self.temp_data = deque(maxlen=max_points)
        self.load_data = deque(maxlen=max_points)

        fig = Figure(figsize=(4.6, 3.2), dpi=100)
        self.ax_temp = fig.add_subplot(111)
        self.ax_load = self.ax_temp.twinx()

        self.ax_temp.set_xlabel("Seconds")
        self.ax_temp.set_ylabel("CPU Temp (degC)", color="#d62728")
        self.ax_load.set_ylabel("CPU Load (%)", color="#1f77b4")
        self.ax_load.set_ylim(0, 100)

        (self.temp_line,) = self.ax_temp.plot([], [], color="#d62728", label="Temp")
        (self.load_line,) = self.ax_load.plot([], [], color="#1f77b4", linestyle="--", label="Load")

        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().pack(pady=10, fill="both", expand=True)

        # Prime psutil, its first reading is meaningless without a baseline
        psutil.cpu_percent(interval=None)

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_temp()

    def on_slider_change(self, value):
        pct = int(float(value))
        self.percent_label.config(text=f"Flow Rate: {pct}%")
        if self.running:
            pump.value = pct / 100.0

    def set_preset(self, pct):
        self.flow_percent.set(pct)
        self.percent_label.config(text=f"Flow Rate: {pct}%")
        if self.running:
            pump.value = pct / 100.0

    def toggle_pump(self):
        if self.running:
            pump.off()
            self.running = False
            self.status_label.config(text="Status: OFF", fg="red")
            self.toggle_btn.config(text="Start Pump", bg="#4CAF50")
        else:
            pump.value = self.flow_percent.get() / 100.0
            self.running = True
            self.status_label.config(text="Status: ON", fg="green")
            self.toggle_btn.config(text="Stop Pump", bg="#f44336")

    def toggle_stress(self):
        if self.stress_running:
            for p in self.stress_processes:
                p.terminate()
            for p in self.stress_processes:
                p.join()
            self.stress_processes = []
            self.stress_running = False
            self.stress_btn.config(text="Start CPU Stress Test", bg="#2196F3")
            self.stress_status.config(text="Stress: OFF", fg="gray")
        else:
            core_count = os.cpu_count() or 4
            self.stress_processes = [
                multiprocessing.Process(target=cpu_stress_worker, daemon=True)
                for _ in range(core_count)
            ]
            for p in self.stress_processes:
                p.start()
            self.stress_running = True
            self.stress_btn.config(text="Stop CPU Stress Test", bg="#f44336")
            self.stress_status.config(text=f"Stress: ON ({core_count} cores)", fg="green")

    def update_temp(self):
        temp = read_cpu_temp()
        load = psutil.cpu_percent(interval=None)
        elapsed = time.time() - self.start_time

        self.time_data.append(elapsed)
        self.temp_data.append(temp)
        self.load_data.append(load)

        color = "#4CAF50"
        if temp >= 75:
            color = "#f44336"
        elif temp >= 60:
            color = "#FF9800"
        self.temp_label.config(text=f"CPU Temp: {temp:.1f} degC", fg=color)
        self.load_label.config(text=f"CPU Load: {load:.0f}%")

        self.temp_line.set_data(self.time_data, self.temp_data)
        self.load_line.set_data(self.time_data, self.load_data)
        self.ax_temp.relim()
        self.ax_temp.autoscale_view()
        if len(self.time_data) > 1:
            self.ax_temp.set_xlim(self.time_data[0], self.time_data[-1])
        self.canvas.draw_idle()

        self.root.after(SAMPLE_INTERVAL_MS, self.update_temp)

    def on_close(self):
        for p in self.stress_processes:
            p.terminate()
        for p in self.stress_processes:
            p.join()
        pump.off()
        pump.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PumpControlApp(root)
    root.mainloop()