# Thermal system-ID summary

## This rig

| label        | cooling_type   | fit_basis          | chosen_model    |   delta_T_c | window_clipped_to_next_step   |   effective_window_s |   tau_nonlinear_s |   r2_nonlinear |   tau_linearized_s |   r2_linearized |   tau1_two_exp_s |   tau2_two_exp_s |   r2_two_exp |   Q_ss_w |   Q_baseline_w |   delta_Q_w |   R_thermal_c_per_w |   C_thermal_j_per_c |   R1_two_exp_c_per_w |   C1_two_exp_j_per_c |   R2_two_exp_c_per_w |   C2_two_exp_j_per_c |
|:-------------|:---------------|:-------------------|:----------------|------------:|:------------------------------|---------------------:|------------------:|---------------:|-------------------:|----------------:|-----------------:|-----------------:|-------------:|---------:|---------------:|------------:|--------------------:|--------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| fan @ t=116s | fan            | delta_T_vs_ambient | two_exponential |      30.113 | True                          |                421.2 |            34.079 |          0.921 |             26.617 |           0.694 |            8.054 |           81.423 |        0.973 |    5.255 |          1.748 |       3.507 |               8.587 |               3.969 |                5.616 |                1.434 |                3.567 |               22.825 |
| fan @ t=540s | fan            | delta_T_vs_ambient | two_exponential |     -28.6   | False                         |                400.7 |            30.117 |          0.921 |            325.573 |           0.072 |            8.768 |           81.092 |        0.979 |    1.764 |          5.284 |      -3.521 |               8.124 |               3.707 |                5.882 |                1.491 |                2.998 |               27.048 |

## Published reference values (Shields, 2009, Georgia Tech)

| System | tau [s] |
|---|---|
| Legacy server, full load (processor outlet) | 340 |
| Legacy server, full load (PSU outlet) | 380 |
| Legacy server, idle (processor outlet) | 370 |
| Legacy server, idle (PSU outlet) | 300 |
| Modern 2U Xeon server, full load (processor outlet) | 130 |
| Modern 2U Xeon server, full load (PSU outlet) | 990 |
| Bare resistive heater (thermal-mass lower limit) | 50 |
| CRAC air-to-water HX (coolant flow step) | 10 |
