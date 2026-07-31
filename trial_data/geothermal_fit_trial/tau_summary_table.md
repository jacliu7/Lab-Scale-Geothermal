# Thermal system-ID summary

## This rig

| label               | cooling_type   | fit_basis          |   delta_T_c | window_clipped_to_next_step   |   effective_window_s |   tau1_two_exp_s |   tau2_two_exp_s |   r2_two_exp |   Q_ss_w |   Q_baseline_w |   delta_Q_w |   R_thermal_c_per_w |   R1_two_exp_c_per_w |   C1_two_exp_j_per_c |   R2_two_exp_c_per_w |   C2_two_exp_j_per_c |
|:--------------------|:---------------|:-------------------|------------:|:------------------------------|---------------------:|-----------------:|-----------------:|-------------:|---------:|---------------:|------------:|--------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| geothermal @ t=254s | geothermal     | delta_T_vs_ambient |        9.35 | True                          |                170.1 |            0.508 |           27.989 |        0.848 |    5.972 |          2.448 |       3.524 |               2.803 |                2.075 |                0.245 |                0.728 |               38.438 |
| geothermal @ t=429s | geothermal     | delta_T_vs_ambient |       -7.7  | False                         |                152.2 |            4.199 |          140.201 |        0.752 |    2.409 |          6.011 |      -3.602 |               2.497 |                2.442 |                1.719 |                0.054 |             2577.74  |

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
