# GPU round: test results (120 test_fresh instances, every choice fixed on validation)

Gap = (C − DB)/C to the tightest proven bound; served = no shedding or reserve shortfall; speed-up = full MILP at 0.1 % / method, timed in the same process (mean / geometric mean).

| method | served | gap mean | median | max | speed-up mean / geo |
|---|---|---|---|---|---|
| w2_rloop_rl|milp01 | 96.7 % | 0.244 % | 0.096 % | 4.28 % | 1.0× / 1.0× |
| w2_rloop_rl|milp05 | 96.7 % | 0.285 % | 0.105 % | 3.08 % | 3.6× / 2.1× |
| w1_rloop_nofb_rl @24 + safety net | 100.0 % | 0.343 % | 0.195 % | 2.02 % | 3.8× / 2.9× |
| w2_rloop_rl @24 + safety net | 100.0 % | 0.366 % | 0.196 % | 2.92 % | 3.9× / 2.9× |
| w2_rloop_rl @24 steps | 99.2 % | 0.786 % | 0.235 % | 5.74 % | 5.4× / 3.8× |
| LP-checked local search (no learning) | 99.2 % | 1.245 % | 0.820 % | 8.05 % | 16.6× / 11.8× |
| w2_rloop_rl @halting head | 97.5 % | 1.276 % | 0.550 % | 24.58 % | 25.0× / 18.9× |
| w1_rloop_nofb_rl @24 steps | 94.2 % | 1.339 % | 0.241 % | 35.05 % | 5.4× / 3.8× |
| w1_rloop_nofb_rl @halting head | 94.2 % | 1.631 % | 0.608 % | 35.05 % | 25.1× / 17.8× |
| lpround (no learning, 5 LPs) | 98.3 % | 1.766 % | 0.983 % | 21.18 % | 52.0× / 37.7× |

Paired differences (positive = the first method is better / faster; 95 % bootstrap CI). Δ log speed-up compares each method's speed-up over the full MILP solved in its own process (within an arm: the log time ratio).

| first | second | n | Δ gap (pp) | Δ log speed-up | n both served | Δ gap, both served |
|---|---|---|---|---|---|---|
| w2_rloop_rl @24 steps | w1_rloop_nofb_rl @24 steps | 120 | +0.553 [+0.006, +1.288] | -0.00 [-0.01, +0.01] | 112 | -0.050 [-0.212, +0.118] |
| w2_rloop_rl @halting head | w1_rloop_nofb_rl @halting head | 120 | +0.356 [-0.382, +1.115] | +0.06 [-0.05, +0.17] | 111 | -0.001 [-0.211, +0.213] |
| w2_rloop_rl @24 + safety net | w1_rloop_nofb_rl @24 + safety net | 120 | -0.023 [-0.061, +0.013] | -0.01 [-0.07, +0.05] | 120 | -0.023 [-0.061, +0.013] |
| w2_rloop_rl @24 steps | lpround (no learning, 5 LPs) | 120 | +0.980 [+0.635, +1.391] | -2.30 [-2.34, -2.26] | 117 | +0.907 [+0.573, +1.310] |
| w2_rloop_rl @24 steps | LP-checked local search (no learning) | 120 | +0.459 [+0.250, +0.668] | -1.14 [-1.17, -1.10] | 118 | +0.457 [+0.243, +0.666] |
| w2_rloop_rl @24 + safety net | w2_rloop_rl|milp05 | 120 | -0.081 [-0.144, -0.020] | +0.30 [+0.15, +0.46] | 116 | -0.079 [-0.140, -0.017] |
