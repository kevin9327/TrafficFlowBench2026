import os
import sys
import time
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")
ARTIFACT_DIR = r"C:\Users\swsz9\.gemini\antigravity\brain\c9e5e770-b6cf-4fe7-a4b3-7e0ca322c2a8"
ENS_DIR = os.path.abspath("TrafficFlowBench2026/ens")
OUT_CSV = os.path.abspath("TrafficFlowBench2026/submission_super_4arch.csv")


def main():
    models = [
        ("tf_yeashu_submission.csv", 0.50),
        ("tf_sujan_submission.csv", 0.30),
        ("tf_souke_submission.csv", 0.13),
        ("tf_yuni_submission.csv", 0.07),
    ]
    print("Loading base ID and task columns from tf_yeashu_submission.csv...")
    base = pd.read_csv(
        os.path.join(ENS_DIR, models[0][0]),
        dtype={"submission_id": str, "task": str, "speed_kmh": "float32", "flow_vph": "float32", "queue_pred": "float32", "path_flow": "float32"},
    )
    n = len(base)
    print(f"Loaded {n:,} rows.")

    speed = base["speed_kmh"].to_numpy(dtype=np.float32) * models[0][1]
    flow = base["flow_vph"].to_numpy(dtype=np.float32) * models[0][1]
    queue = base["queue_pred"].to_numpy(dtype=np.float32) * 0.45
    pflow = base["path_flow"].to_numpy(dtype=np.float32) * models[0][1]

    q_weights = [0.45, 0.30, 0.15, 0.10]
    for idx, (fn, w) in enumerate(models[1:], 1):
        print(f"Blending {fn} (weight={w}, q_weight={q_weights[idx]})...")
        df = pd.read_csv(
            os.path.join(ENS_DIR, fn),
            usecols=["speed_kmh", "flow_vph", "queue_pred", "path_flow"],
            dtype="float32",
        )
        speed += df["speed_kmh"].to_numpy(dtype=np.float32) * w
        flow += df["flow_vph"].to_numpy(dtype=np.float32) * w
        queue += df["queue_pred"].to_numpy(dtype=np.float32) * q_weights[idx]
        pflow += df["path_flow"].to_numpy(dtype=np.float32) * w

    task_arr = base["task"].to_numpy()
    is_state = task_arr == "state"
    is_queue = task_arr == "queue"
    is_odme = task_arr == "odme"

    base["speed_kmh"] = np.where(is_state, np.round(np.clip(speed, 0.0, None), 3), 0.0)
    base["flow_vph"] = np.where(is_state, np.round(np.clip(flow, 0.0, None), 3), 0.0)
    base["queue_pred"] = np.where(is_queue, (queue >= 0.50).astype(np.int32), 0)
    base["path_flow"] = np.where(is_odme, np.round(np.clip(pflow, 0.0, None), 3), 0.0)

    print(f"Writing {OUT_CSV}...")
    base.to_csv(OUT_CSV, index=False, float_format="%.3f")
    print(f"Saved {OUT_CSV} ({os.path.getsize(OUT_CSV):,} bytes)")

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        ctx = browser.contexts[0]
        for pg in ctx.pages:
            if "2026-ieee-big-data-traffic-flow-bench" in pg.url:
                pg.close()

        page = ctx.new_page()
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.goto("https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench", wait_until="domcontentloaded", timeout=60000)
        time.sleep(5)

        sub_btn = page.locator('button:has-text("Submit Prediction"), a:has-text("Submit Prediction")')
        if sub_btn.count() > 0:
            sub_btn.first.click()
            time.sleep(4)

        cdp = ctx.new_cdp_session(page)
        cdp.send("DOM.enable")
        doc = cdp.send("DOM.getDocument", {"depth": -1, "pierce": True})
        q = cdp.send("DOM.querySelectorAll", {"nodeId": doc["root"]["nodeId"], "selector": 'input[type="file"]'})
        node_ids = q.get("nodeIds", [])
        print("Found file input nodeIds:", node_ids)
        for nid in node_ids:
            try:
                cdp.send("DOM.setFileInputFiles", {"files": [OUT_CSV], "nodeId": nid})
                print(f"Set file via CDP on nodeId {nid}")
                break
            except Exception as e:
                print("CDP err:", e)

        print("Waiting for 280MB CSV upload to finish...")
        for step in range(180):
            time.sleep(2)
            sub_submit = page.locator('button:has-text("Submit")')
            ready = False
            for i in range(sub_submit.count()):
                b = sub_submit.nth(i)
                if b.is_visible() and (b.inner_text() or "").strip() == "Submit" and not b.is_disabled():
                    ready = True
                    break
            if ready:
                print(f"Upload complete at {(step + 1) * 2}s!")
                break

        ta = page.locator('textarea[placeholder*="Describe"], textarea[aria-label*="Description"], textarea')
        if ta.count() > 0 and ta.first.is_visible():
            ta.first.fill("4-Architecture Multi-Task Spatio-Temporal Ensemble (State Speed/Flow Convex Blend + Queue Weighted Majority Vote + ODME Path Flow)")
            time.sleep(1)

        sub_submit = page.locator('button:has-text("Submit")')
        for i in range(sub_submit.count()):
            b = sub_submit.nth(i)
            if b.is_visible() and (b.inner_text() or "").strip() == "Submit" and not b.is_disabled():
                b.click()
                print("Clicked Submit on TrafficFlowBench2026!")
                break

        time.sleep(12)
        page.goto("https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/submissions", wait_until="domcontentloaded", timeout=60000)
        for _ in range(12):
            time.sleep(5)
            txt = page.locator("body").inner_text()
            if "Pending" not in txt and ("Complete" in txt or "0." in txt):
                break
            page.reload(wait_until="domcontentloaded")
        time.sleep(3)
        shot = os.path.join(ARTIFACT_DIR, "kaggle_traffic_submitted.png")
        page.screenshot(path=shot, full_page=True)
        print("Saved screenshot:", shot)
        print(page.locator("body").inner_text()[800:2600])

        page.goto("https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/leaderboard", wait_until="domcontentloaded", timeout=60000)
        time.sleep(5)
        shot_lb = os.path.join(ARTIFACT_DIR, "kaggle_traffic_leaderboard.png")
        page.screenshot(path=shot_lb, full_page=True)
        print("Saved leaderboard screenshot:", shot_lb)
        print(page.locator("body").inner_text()[800:2600])
        page.close()


if __name__ == "__main__":
    main()
