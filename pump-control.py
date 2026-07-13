#!/usr/bin/env python3
"""
Pump Flow Rate Control, Raspberry Pi version
Converted from an Arduino MOSFET pump control sketch.
Drives a MOSFET-gated pump with PWM and gives you a touch/mouse GUI
to set flow rate on the Pi's own screen.

Hardware notes:
- PUMP_PIN below is a BCM GPIO number, not a physical pin number.
- GPIO18 is used because it supports hardware PWM, which gives a
  cleaner, more stable signal than software PWM on other pins.
- Keep a flyback diode across the pump terminals if it doesn't
  already have one built in, since a pump motor is an inductive load.
- Gate resistor (100 to 220 ohm) between this pin and the MOSFET gate,
  and a pulldown resistor (10k) from gate to ground, is good practice
  so the pump doesn't twitch on during boot before the script runs.
"""

import tkinter as tk
from gpiozero import PWMOutputDevice

# --- Hardware setup ---
PUMP_PIN = 18       # BCM numbering. Change if you wired it elsewhere.
PWM_FREQ_HZ = 1000  # 1 kHz is a safe default for most MOSFET pump drivers

pump = PWMOutputDevice(PUMP_PIN, frequency=PWM_FREQ_HZ)


class PumpControlApp:
    def __init__(self, root):
        self.root = root
        root.title("Pump Flow Control")
        root.geometry("420x320")

        self.running = False
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

        root.protocol("WM_DELETE_WINDOW", self.on_close)

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

    def on_close(self):
        pump.off()
        pump.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PumpControlApp(root)
    root.mainloop()