# Lab-Scale Geothermal Cooling

Testing whether a small ground-loop cooling setup actually helps with the
thermal load of a Raspberry Pi running local LLM inference, compared to no
cooling and a plain fan. This repo is the test scripts and the trial data behind the paper. 

Paper link (manuscript in preparation for journal submission): 
Website for more information: [https://earthheat.org/products/geothermal_lab_demo.html#testing-demo](url)

Basic approach: step a Pi from idle to full inference load under each
cooling condition, fit an RC thermal model to how the temperature responds
(first-order or two-exponential), then check
that fitted model against a more realistic bursty on/off load. tau, R<sub>th</sub>,
C<sub>th</sub>, COP, and PUE from these runs are what end up in the manuscript.

## Layout

```
v3 tests/               current scripts, use these
  geo_common.py           sensor I/O, pump control, COP/PUE/Q math
  thermal_system_id.py    step-response fitting
  step_load_test.py       calibration run (idle -> step up -> hold -> step down -> idle)
  bursty_test.py          validation run (on/off cycles, scored against the fit)

trial_data/              CSVs, fit summaries, plots from actual rig runs
old testing files/       v0/v1/v2 iterations, kept around so old trial data still
                         makes sense, not maintained (don't run these)
```

## Setup

Python 3 with numpy, pandas, matplotlib, scipy. On the Pi you also want
gpiozero for pump control and llama-cpp-python + a GGUF model for the real
inference load. Neither's required just to run the scripts -- see
`--simulate` below.

```bash
pip install numpy pandas matplotlib scipy
```

## Running a trial

One condition at a time (`no_cooling`, `fan`, `geothermal`). Randomize the
order across trials/days yourself, the scripts don't do that for you.

### Step load test (run this first)

Idles until CPU temp is stable, steps the load to full inference, holds
until stable again, steps back down. Fits tau/R<sub>th</sub>/C<sub>th</sub> on both
transitions as soon as it finishes.

```bash
cd "v3 tests"
python3 step_load_test.py --model tinyllama.gguf --condition geothermal \
    --trial 1 --output ../trial_data \
    --ambient-sensor 28-0000ambient1 \
    --pump-duty 100 --lock-clock
```

Off-Pi dry run (no hardware, just to sanity check the state machine / fit
pipeline):

```bash
python3 step_load_test.py --condition no_cooling --trial 1 \
    --output ../trial_data --simulate
```

Output is the raw trial CSV plus a `fit_<condition>_trial<N>/` folder
(tau summary CSV + markdown, per-step fit plots, and a plot comparing your
tau against published data-center numbers). Console output will tell you
if it picked the two-exponential model over single (usually happens on
geothermal, since the loop's a chain: die -> heatsink -> coolant ->
reservoir) and will flag if a fit window got clipped by the next phase
starting early.

### Bursty test (run this second)

Points at the tau_fit_summary.csv the step test made, drives repeated
load/idle cycles (default 2 min / 1 min), and simulates the RC model
forward through the measured power to see how well it predicts what
actually happened.

```bash
python3 bursty_test.py --model tinyllama.gguf --condition geothermal \
    --trial 1 --output ../trial_data \
    --tau-source ../trial_data/fit_geothermal_trial1/tau_fit_summary.csv \
    --ambient-sensor 28-0000ambient1 \
    --pump-duty 100 --lock-clock --total-minutes 25
```

Scores RMSE/MAE/bias and plots predicted vs. measured delta_T. If they
track, the step-test model generalizes to a realistic load pattern; if
they diverge, that's the interesting result -- it's where the lumped
model breaks down.

## Notes to self before running

Q (heat input) is CPU/SoC electrical power, not a fluid-side balance --
there's no reliable coolant inlet sensor on this rig, so q_cpu_w from
vcgencmd stands in for die heat (compute_q_cpu_w in geo_common.py).

Power gets sampled in the background up to 50 Hz and time-weighted per
logged interval, not grabbed once a second, because the die's thermal
time constant (~6s) is fast enough that a single sample per iteration
misses real fluctuations.

Fits use &Delta;T = T<sub>die</sub> - T<sub>ambient</sub> whenever an ambient sensor is
wired up so ambient drift between trials doesn't leak into R<sub>th</sub>.
Without one it falls back to absolute temp.

Set --min-hold-s on step_load_test.py for the geothermal condition,
something like 3-5x the expected slow tau. The default stability check
only looks at a trailing window and will call a phase "stable" before
the slow branch has actually finished decaying.

Safety cutoff is 185F / ~85C, same as the Pi's own throttle point. A
trial that hits it stops immediately -- whatever got captured up to
that point still gets fit, but flag the trial for exclusion or rerun.

## Trial data

`trial_data/` has the runs behind the paper: `*_fit_trial/` folders are
step-load calibration (fit plots, tau summary, benchmark comparison),
`*_bursty_trial/` folders are the validation runs (prediction-vs-measured
plot, RMSE/MAE/bias). One pair per cooling condition.
