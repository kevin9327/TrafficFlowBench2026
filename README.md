# 2026 IEEE Big Data — Traffic Flow Bench (`LWR/FD Physics + Shockwave Queue + IPF ODME`)

Official open-source repository for **IEEE Big Data Cup 2026 (Challenge 06): 2026 IEEE Big Data-Traffic Flow Bench**.

## Coupled 4-Task Architecture
1. **Task 1 & Task 3 (`S_state` + `S_physics`):** Bidirectional spatio-temporal kinematic wave (`LWR`) interpolation projected onto each link's triangular Fundamental Diagram (`fd_parameters.csv`) across all 10 freeway corridors (`D7_I10_E..D12_I405_S`).
2. **Task 2 (`S_queue`):** Shockwave propagation velocity (`w = \Delta q / \Delta k`) + boundary bottleneck congestion detector for 30-minute queue forecasting.
3. **Task 4 (`S_ODME`):** Multiplicative Iterative Proportional Fitting (`IPF`) anchored to the weak prior `b` and observed link counts `c`.

## Reproducibility
```bash
pip install -r requirements.txt
python trafficflowbench_solver.py
```
