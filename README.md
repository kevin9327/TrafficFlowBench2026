# TrafficFlowBench 2026 — 4-Architecture Multi-Task Spatio-Temporal Ensemble (IEEE Big Data Cup 2026)

Official solution repository for the **2026 IEEE Big Data-Traffic Flow Bench** (`https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench`), uniting three interdependent freeway network tasks across `6,980,503` evaluation rows:

1. **Task 1 — Spatio-Temporal State Estimation (`state`, `6,735,795` rows)**:
   - Joint estimation of freeway link `speed_kmh` and `flow_vph` across diverse sensor mask regimes (`spatial`, `temporal`, `random`).
   - Convex multi-architecture blend combining Spatio-Temporal Graph GBDT (`yeashu`, weight `0.50`), Cross-Station Lagged LightGBM/CatBoost (`sujan`, weight `0.30`), Bidirectional Temporal Kinematic Reconstructor (`souke`, weight `0.13`), and Shockwave-Aware Regressor (`yuni`, weight `0.07`).
2. **Task 2 — Bottleneck Queue Detection (`queue`, `174,000` rows)**:
   - Binary congestion shockwave classification (`queue_pred` $\in \{0, 1\}$) via calibrated weighted majority voting (`0.45 / 0.30 / 0.15 / 0.10 >= 0.50`) across all 4 diverse architectures to directly optimize classification F1/Accuracy without probability blurring.
3. **Task 3 — Origin-Destination Matrix Estimation (`odme`, `70,708` rows)**:
   - Non-negative path flow reconstruction (`path_flow >= 0`) preserving conservation constraints across origin-destination corridors.

## Reproducibility
```bash
python build_and_submit_traffic.py
```
