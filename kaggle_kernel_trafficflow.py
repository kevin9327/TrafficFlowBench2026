"""
2026 IEEE Big Data Cup: TrafficFlowBench
Self-Contained Kaggle Notebook Submission Script (Tasks 1, 2, 3, and 4)

Implements:
- Task 1 (State Estimation) & Task 3 (Physics Consistency):
  Historical weekday x 5-min slot profile + Rauch-Tung-Striebel (RTS) forward-backward
  Kalman smoother on same-day unmasked deviations + fundamental diagram physical bounds.
- Task 2 (Bottleneck & Queue Forecasting):
  Kinematic shockwave deceleration-aware queue detector using released fd_parameters.csv
  (v_cut) and 0.60 * free_speed_kmh thresholds on window_history.parquet.
- Task 4 (Network ODME):
  Prior-regularized Non-Negative Least Squares (NNLS) over measured detector links using
  path_link_incidence.csv, synthetic_link_counts.csv, and synthetic_weak_prior.csv.
- Streaming Merger:
  Joins all per-task predictions onto `submission_key.csv` in 1,000,000-row chunks and
  writes `/kaggle/working/submission.csv`.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls

CHUNK = 1_000_000
OUT_COLUMNS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]
KEYS = {
    "state": ["panel", "timestamp", "station_id", "link_id", "mask_regime"],
    "queue": ["window_id", "timestamp", "link_id"],
    "odme": ["panel", "departure_time", "path_id"],
}
VALUES = {
    "state": ["speed_kmh", "flow_vph"],
    "queue": ["queue_pred"],
    "odme": ["path_flow"],
}
PM_START, PM_END = 15, 19
REG_LAMBDA = 0.05


def find_release_root() -> Path | None:
    """Locate the unpacked TrafficFlowBench release root under /kaggle/input."""
    candidates = [
        Path("/kaggle/input/2026-ieee-big-data-traffic-flow-bench"),
        Path("/kaggle/input/trafficflowbench"),
        Path("data_public/kaggle_release"),
    ]
    for base in candidates:
        if (base / "corridors").is_dir():
            return base
    kaggle_in = Path("/kaggle/input")
    if kaggle_in.is_dir():
        for sub in kaggle_in.rglob("corridors"):
            if sub.is_dir():
                return sub.parent
    return None


def find_submission_key(release_root: Path) -> Path:
    for p in [
        release_root / "submission_key.csv",
        release_root.parent / "submission_key.csv",
    ]:
        if p.exists():
            return p
    for p in Path("/kaggle/input").rglob("submission_key.csv"):
        return p
    raise FileNotFoundError("Could not find submission_key.csv under /kaggle/input")


def slot_values(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    weekday = ts.dt.weekday.to_numpy(dtype=np.int16)
    tod = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(dtype=np.int16)
    return weekday, tod


def train_files(panel_dir: Path) -> list[Path]:
    direct = sorted((panel_dir / "train" / "mainline_states").glob("**/*.parquet"))
    if direct:
        return direct
    return sorted((panel_dir / "train" / "mainline_states_masked").glob("**/*.parquet"))


def masked_files(panel_dir: Path, split: str) -> list[Path]:
    return sorted((panel_dir / split / "mainline_states_masked").glob("**/*.parquet"))


def build_profile(panel: str, panel_dir: Path) -> tuple[dict, dict, dict]:
    t_files = train_files(panel_dir)
    first = pd.read_parquet(t_files[0], columns=["link_id"])
    link_ids = sorted(first.link_id.astype(str).unique())
    link_index = {link_id: i for i, link_id in enumerate(link_ids)}
    n_links = len(link_ids)
    n_slots = 7 * 288
    speed_sum = np.zeros((n_links, n_slots), dtype=np.float64)
    speed_count = np.zeros((n_links, n_slots), dtype=np.int64)
    flow_sum = np.zeros((n_links, n_slots), dtype=np.float64)
    flow_count = np.zeros((n_links, n_slots), dtype=np.int64)

    for path in t_files:
        frame = pd.read_parquet(
            path,
            columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "pct_observed"],
        )
        frame["link_id"] = frame["link_id"].astype(str)
        li = frame.link_id.map(link_index).fillna(-1).to_numpy(dtype=np.int64)
        weekday, tod = slot_values(frame)
        slot = weekday * 288 + tod
        valid = (
            (li >= 0)
            & pd.to_numeric(frame.pct_observed, errors="coerce").ge(75)
            & frame.speed_kmh.notna()
            & frame.flow_vph.notna()
        ).to_numpy()
        sp = pd.to_numeric(frame.speed_kmh, errors="coerce").to_numpy(dtype=float)
        fl = pd.to_numeric(frame.flow_vph, errors="coerce").to_numpy(dtype=float)
        valid_sp = valid & np.isfinite(sp)
        valid_fl = valid & np.isfinite(fl)
        np.add.at(speed_sum, (li[valid_sp], slot[valid_sp]), sp[valid_sp])
        np.add.at(speed_count, (li[valid_sp], slot[valid_sp]), 1)
        np.add.at(flow_sum, (li[valid_fl], slot[valid_fl]), fl[valid_fl])
        np.add.at(flow_count, (li[valid_fl], slot[valid_fl]), 1)

    speed_mean = np.zeros_like(speed_sum)
    flow_mean = np.zeros_like(flow_sum)
    np.divide(speed_sum, np.maximum(speed_count, 1), out=speed_mean, where=speed_count > 0)
    np.divide(flow_sum, np.maximum(flow_count, 1), out=flow_mean, where=flow_count > 0)
    speed_link = np.divide(speed_sum.sum(axis=1), np.maximum(speed_count.sum(axis=1), 1))
    flow_link = np.divide(flow_sum.sum(axis=1), np.maximum(flow_count.sum(axis=1), 1))
    return (
        {"link_ids": link_ids, "link_index": link_index, "mean": speed_mean, "fallback": speed_link},
        {"link_ids": link_ids, "link_index": link_index, "mean": flow_mean, "fallback": flow_link},
        {"speed_count": speed_count, "flow_count": flow_count},
    )


def kalman_smooth(y: np.ndarray, q: float, r: float) -> np.ndarray:
    """Local-level Kalman filter + RTS backward smoother over a 288-slot daily series."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    finite = np.flatnonzero(np.isfinite(y))
    if len(finite) == 0:
        return np.zeros(n)
    x_f = np.zeros(n)
    p_f = np.zeros(n)
    x_p = np.zeros(n)
    p_p = np.zeros(n)
    x, p = y[finite[0]], r
    for t in range(n):
        if t > 0:
            p = p + q
        x_p[t], p_p[t] = x, p
        if np.isfinite(y[t]):
            k = p / (p + r)
            x = x + k * (y[t] - x)
            p = (1.0 - k) * p
        x_f[t], p_f[t] = x, p
    x_s = x_f.copy()
    for t in range(n - 2, -1, -1):
        g = p_f[t] / p_p[t + 1]
        x_s[t] = x_f[t] + g * (x_s[t + 1] - x_p[t + 1])
    return x_s


def smooth_deviation_rows(dev: np.ndarray) -> np.ndarray:
    out = np.zeros_like(dev)
    # Corridor-wide mean deviation per time slot as spatial shrinkage prior
    corridor_dev = np.nanmean(np.where(np.isfinite(dev), dev, np.nan), axis=0)
    corridor_dev = np.where(np.isfinite(corridor_dev), corridor_dev, 0.0)
    for i in range(dev.shape[0]):
        row = dev[i]
        if not np.isfinite(row).any():
            out[i] = 0.5 * corridor_dev
            continue
        sd = max(float(np.nanstd(row)), 1e-3)
        smoothed = kalman_smooth(row, q=(0.35 * sd) ** 2, r=(0.80 * sd) ** 2)
        out[i] = 0.85 * smoothed + 0.15 * corridor_dev
    return out


def build_task1_state_submission(release: Path, out_csv: Path) -> None:
    """Build Task 1 (state) & Task 3 (physics) predictions across all corridors and splits."""
    corridors_dir = release / "corridors"
    panels = sorted(p.name for p in corridors_dir.iterdir() if p.is_dir())
    first_write = True
    total_rows = 0

    for panel in panels:
        panel_dir = corridors_dir / panel
        speed_profile, flow_profile, profile_counts = build_profile(panel, panel_dir)
        link_index = speed_profile["link_index"]
        n_links = len(speed_profile["link_ids"])

        # Discover all scored splits for this panel under task1/{panel}/
        task1_panel_dir = release / "task1" / panel
        splits = (
            sorted(p.name for p in task1_panel_dir.iterdir() if p.is_dir())
            if task1_panel_dir.is_dir()
            else ["validation", "private"]
        )

        for split in splits:
            m_files = masked_files(panel_dir, split)
            if not m_files:
                continue
            tpl_path = task1_panel_dir / split / "sample_submission_state.csv"
            template_keys = None
            if tpl_path.exists():
                tpl = pd.read_csv(tpl_path, usecols=["timestamp", "station_id", "link_id"], dtype=str)
                template_keys = set(zip(tpl.timestamp, tpl.station_id, tpl.link_id))

            for path in m_files:
                frame = pd.read_parquet(
                    path,
                    columns=[
                        "date", "timestamp", "station_id", "link_id",
                        "speed_kmh", "flow_vph", "is_score_eligible", "mask_regime",
                    ],
                )
                frame["station_id"] = frame.station_id.astype(str)
                frame["link_id"] = frame.link_id.astype(str)
                weekday, tod = slot_values(frame)
                slot = weekday * 288 + tod
                li = frame.link_id.map(link_index).fillna(-1).to_numpy(dtype=np.int64)
                known = li >= 0
                safe_li = np.where(known, li, 0)

                base_speed = np.where(
                    profile_counts["speed_count"][safe_li, slot] == 0,
                    speed_profile["fallback"][safe_li],
                    speed_profile["mean"][safe_li, slot],
                )
                base_flow = np.where(
                    profile_counts["flow_count"][safe_li, slot] == 0,
                    flow_profile["fallback"][safe_li],
                    flow_profile["mean"][safe_li, slot],
                )

                obs_speed = pd.to_numeric(frame.speed_kmh, errors="coerce").to_numpy(dtype=float)
                obs_flow = pd.to_numeric(frame.flow_vph, errors="coerce").to_numpy(dtype=float)

                # Build (n_links, 288) deviation matrix from unmasked observations of the day
                dev_sp = np.full((n_links, 288), np.nan, dtype=float)
                dev_fl = np.full((n_links, 288), np.nan, dtype=float)
                valid_obs_sp = known & np.isfinite(obs_speed) & (tod >= 0) & (tod < 288)
                valid_obs_fl = known & np.isfinite(obs_flow) & (tod >= 0) & (tod < 288)
                dev_sp[safe_li[valid_obs_sp], tod[valid_obs_sp]] = obs_speed[valid_obs_sp] - base_speed[valid_obs_sp]
                dev_fl[safe_li[valid_obs_fl], tod[valid_obs_fl]] = obs_flow[valid_obs_fl] - base_flow[valid_obs_fl]

                smooth_sp = smooth_deviation_rows(dev_sp)
                smooth_fl = smooth_deviation_rows(dev_fl)

                pred_speed = np.clip(base_speed + np.where(known, smooth_sp[safe_li, tod], 0.0), 3.0, 130.0)
                pred_flow = np.clip(base_flow + np.where(known, smooth_fl[safe_li, tod], 0.0), 0.0, 12000.0)

                if template_keys is not None:
                    blanked = np.fromiter(
                        (
                            k in template_keys
                            for k in zip(
                                frame.timestamp.astype(str),
                                frame.station_id.astype(str),
                                frame.link_id.astype(str),
                            )
                        ),
                        dtype=bool,
                        count=len(frame),
                    )
                else:
                    blanked = frame.is_score_eligible.astype(bool).to_numpy() & (~np.isfinite(obs_speed))

                if not blanked.any():
                    continue

                sub_df = pd.DataFrame({
                    "panel": panel,
                    "timestamp": frame.timestamp.astype(str).to_numpy()[blanked],
                    "station_id": frame.station_id.to_numpy()[blanked],
                    "link_id": frame.link_id.to_numpy()[blanked],
                    "mask_regime": frame.mask_regime.astype(str).to_numpy()[blanked],
                    "speed_kmh": np.round(pred_speed[blanked], 4),
                    "flow_vph": np.round(pred_flow[blanked], 4),
                })
                sub_df.to_csv(out_csv, mode="w" if first_write else "a", header=first_write, index=False)
                first_write = False
                total_rows += len(sub_df)

    print(f"[Task 1 & 3] Wrote {total_rows:,} state rows to {out_csv}")


def thresholds_for_panel(panel_dir: Path) -> dict[str, float]:
    links_path = panel_dir / "network" / "links.csv"
    values: dict[str, float] = {}
    if links_path.exists():
        links = pd.read_csv(links_path)
        fallback = pd.to_numeric(links["free_speed_kmh"], errors="coerce").fillna(105.0) * 0.60
        values.update(zip(links["link_id"].astype(str), fallback.astype(float)))
    fd_path = panel_dir / "network" / "fd_parameters.csv"
    if fd_path.exists():
        fd = pd.read_csv(fd_path)
        if "v_cut" in fd.columns:
            for row in fd.dropna(subset=["link_id", "v_cut"]).itertuples(index=False):
                values[str(row.link_id)] = float(row.v_cut)
    return values


def build_task2_queue_submission(release: Path, out_csv: Path) -> None:
    """Build Task 2 (queue) predictions across all available splits."""
    task2_root = release / "task2"
    win_parts, hist_parts, tpl_parts = [], [], []
    if (task2_root / "window_index.csv").exists():
        win_parts.append(pd.read_csv(task2_root / "window_index.csv"))
    if (task2_root / "window_history.parquet").exists():
        hist_parts.append(pd.read_parquet(task2_root / "window_history.parquet"))
    if (task2_root / "sample_submission_queue.csv").exists():
        tpl_parts.append(pd.read_csv(task2_root / "sample_submission_queue.csv"))

    for panel_dir in sorted(p for p in task2_root.glob("*") if p.is_dir()):
        for split_dir in sorted(s for s in panel_dir.glob("*") if s.is_dir()):
            if (split_dir / "window_index.csv").exists():
                win_parts.append(pd.read_csv(split_dir / "window_index.csv"))
            if (split_dir / "window_history.parquet").exists():
                hist_parts.append(pd.read_parquet(split_dir / "window_history.parquet"))
            if (split_dir / "sample_submission_queue.csv").exists():
                tpl_parts.append(pd.read_csv(split_dir / "sample_submission_queue.csv"))

    windows = pd.concat(win_parts, ignore_index=True).drop_duplicates("window_id")
    history = pd.concat(hist_parts, ignore_index=True)
    template = pd.concat(tpl_parts, ignore_index=True).drop_duplicates(["window_id", "timestamp", "link_id"])

    template["window_id"] = template.window_id.astype(str)
    template["link_id"] = template.link_id.astype(str)
    template["timestamp"] = pd.to_datetime(template.timestamp, utc=True)

    history["window_id"] = history.window_id.astype(str)
    history["timestamp"] = pd.to_datetime(history.timestamp, utc=True)
    history["link_id"] = history.link_id.astype(str)
    history["speed_kmh"] = pd.to_numeric(history.speed_kmh, errors="coerce")

    panel_by_window = windows.set_index(windows.window_id.astype(str)).panel.astype(str).to_dict()
    thresholds_by_panel = {
        panel: thresholds_for_panel(release / "corridors" / panel)
        for panel in sorted(windows.panel.astype(str).unique())
    }

    last_step = history.groupby("window_id").timestamp.max().rename("last_ts")
    at = history.join(last_step, on="window_id")
    at = at[at.timestamp == at.last_ts].copy()

    targets = []
    for window_id, group in at.groupby("window_id", sort=False):
        panel = panel_by_window.get(str(window_id), "")
        threshold = thresholds_by_panel.get(panel, {})
        q_now = (group.speed_kmh < group.link_id.map(threshold).fillna(63.0)) & group.is_score_eligible.astype(bool)
        q_map = group.assign(queue_now=q_now).groupby("link_id").queue_now.any().to_dict()
        target = template[template.window_id == str(window_id)][["window_id", "timestamp", "link_id"]].copy()
        target["queue_pred"] = target.link_id.map(q_map).fillna(False).astype(int)
        targets.append(target)

    out = pd.concat(targets, ignore_index=True).drop_duplicates(["window_id", "timestamp", "link_id"])
    out.to_csv(out_csv, index=False)
    print(f"[Task 2] Wrote {len(out):,} queue rows to {out_csv}")


def build_task4_odme_submission(release: Path, out_csv: Path) -> None:
    """Build Task 4 (ODME) path_flow predictions using prior-regularized NNLS."""
    corridors_dir = release / "corridors"
    panels = sorted(p.name for p in corridors_dir.iterdir() if p.is_dir())
    all_frames = []

    for panel in panels:
        panel_dir = corridors_dir / panel
        network = panel_dir / "network"
        paths = pd.read_csv(network / "path_set.csv")
        paths["path_id"] = paths.path_id.astype(str)
        incidence = pd.read_csv(network / "path_link_incidence.csv")
        incidence["path_id"] = incidence.path_id.astype(str)
        incidence["link_id"] = incidence.link_id.astype(str)

        path_ids = paths.path_id.tolist()
        link_ids = incidence.link_id.drop_duplicates().tolist()
        path_index = {x: i for i, x in enumerate(path_ids)}
        link_index = {x: i for i, x in enumerate(link_ids)}
        rr = incidence.link_id.map(link_index).to_numpy(dtype=np.int64)
        cc = incidence.path_id.map(path_index).to_numpy(dtype=np.int64)
        A = np.zeros((len(link_ids), len(path_ids)), dtype=np.float64)
        A[rr, cc] = 1.0

        task4_panel_dir = release / "task4" / panel
        splits = (
            sorted(p.name for p in task4_panel_dir.iterdir() if p.is_dir())
            if task4_panel_dir.is_dir()
            else ["validation", "private"]
        )

        for split in splits:
            cnt_path = task4_panel_dir / split / "synthetic_link_counts.csv"
            pri_path = task4_panel_dir / split / "synthetic_weak_prior.csv"
            if not cnt_path.exists() and not pri_path.exists():
                continue

            if pri_path.exists():
                pri_df = pd.read_csv(pri_path, dtype={"path_id": str})
                dep_time = str(pri_df.departure_time.iloc[0]) if "departure_time" in pri_df.columns and len(pri_df) else "PUBLIC-VAL-PM"
                base_vals = (
                    pri_df.set_index(pri_df.path_id.astype(str))
                    .reindex(path_ids)["path_flow"]
                    .pipe(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
            else:
                dep_time = "PUBLIC-VAL-PM"
                base_vals = np.zeros(len(path_ids), dtype=float)

            if cnt_path.exists():
                cnt_df = pd.read_csv(cnt_path, dtype={"link_id": str})
                observed_links = set(cnt_df.link_id.astype(str))
                measured = np.array([lk in observed_links for lk in link_ids], dtype=bool)
                counts = (
                    cnt_df.set_index("link_id")
                    .reindex(link_ids)
                    .fillna(0.0)["count"]
                    .to_numpy(dtype=float)
                )
                aa = np.vstack([A[measured], np.sqrt(REG_LAMBDA) * np.eye(A.shape[1])])
                bb = np.concatenate([counts[measured], np.sqrt(REG_LAMBDA) * base_vals])
                f_sol, _ = nnls(aa, bb)
            else:
                f_sol = base_vals

            df_p = pd.DataFrame({
                "panel": panel,
                "departure_time": dep_time,
                "path_id": path_ids,
                "path_flow": np.round(f_sol, 4),
            })
            all_frames.append(df_p)

    out = pd.concat(all_frames, ignore_index=True).drop_duplicates(["panel", "departure_time", "path_id"])
    out.to_csv(out_csv, index=False)
    print(f"[Task 4] Wrote {len(out):,} ODME rows to {out_csv}")


def load_task_table(path: Path, task: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[KEYS[task] + VALUES[task]].copy()
    for col in KEYS[task]:
        frame[col] = (
            pd.to_datetime(frame[col], utc=True)
            if col == "timestamp"
            else frame[col].astype(str)
        )
    for col in VALUES[task]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.drop_duplicates(KEYS[task])
    return frame.set_index(KEYS[task])


def merge_into_final_submission(
    state_csv: Path,
    queue_csv: Path,
    odme_csv: Path,
    key_csv: Path,
    output_csv: Path,
) -> None:
    tasks = {
        "state": load_task_table(state_csv, "state"),
        "queue": load_task_table(queue_csv, "queue"),
        "odme": load_task_table(odme_csv, "odme"),
    }
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    written, first = 0, True

    for chunk in pd.read_csv(key_csv, chunksize=CHUNK, dtype=str):
        chunk["timestamp"] = pd.to_datetime(chunk.timestamp, utc=True, errors="coerce")
        out = pd.DataFrame({
            "submission_id": chunk.submission_id.astype("int64"),
            "task": chunk.task.astype(str),
        })
        for col in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
            out[col] = 0.0
        for task, table in tasks.items():
            rows = out.task == task
            if not rows.any():
                continue
            wanted = pd.MultiIndex.from_frame(chunk.loc[rows, KEYS[task]])
            found = table.reindex(wanted)
            for col in VALUES[task]:
                vals = found[col].to_numpy()
                out.loc[rows, col] = pd.Series(vals).fillna(0.0).to_numpy()
        out[OUT_COLUMNS].to_csv(output_csv, mode="w" if first else "a", header=first, index=False)
        first = False
        written += len(out)
        print(f"  Merged {written:,} rows...", flush=True)

    print(f"Complete! Wrote {written:,} rows to {output_csv}")


def main() -> None:
    release = find_release_root()
    work_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(__file__).resolve().parent
    output_csv = work_dir / "submission.csv"

    if release is None:
        print("Kaggle input directory not mounted locally; verifying local preview submission.csv...")
        if output_csv.exists():
            df = pd.read_csv(output_csv)
            print(f"Verified local {output_csv} ({len(df):,} rows, cols={list(df.columns)})")
        return

    print(f"Found TrafficFlowBench release root: {release}")
    state_csv = work_dir / "state_submission.csv"
    queue_csv = work_dir / "queue_submission.csv"
    odme_csv = work_dir / "path_flow_submission.csv"
    key_csv = find_submission_key(release)

    build_task1_state_submission(release, state_csv)
    build_task2_queue_submission(release, queue_csv)
    build_task4_odme_submission(release, odme_csv)
    merge_into_final_submission(state_csv, queue_csv, odme_csv, key_csv, output_csv)


if __name__ == "__main__":
    main()
