"""e7_retro.py — evaluate many E7 structure windows in worker processes (2026-10-30, used by e6e7_link PATH A).

Called once when E4 reaches CONFIDENT_GEOMETRY: the stored windows of the last E7_RETRO_HORIZON_S are re-checked with the
CURRENT leader transform and the COMPLETE E2 rival list (E4 CONFIDENT needs 9/9 exact floors). Each window is the unchanged
e7_rematch.evaluate on (t_end - win, t_end] — the same call as a live window, only earlier t_end. Workers open the LiveStore
read-only by path (frames are cut at t_end, so rows written later never enter a window) and run on the E2 cores (6, 7), which
are idle once E2 has finished. No decision is taken here: the caller applies the FIX rule.
"""
import os
import time
from pathlib import Path


def _init(cores):
    try:
        os.sched_setaffinity(0, set(cores))
    except (AttributeError, OSError, ValueError):
        pass
    try:
        os.nice(10)
    except OSError:
        pass


def _one(job):
    import json
    from pathlib import Path
    import e3_tracker as E3
    import e6e7_link as LK
    t0 = time.time()
    st = E3.Store(job["store"]); st.refresh()
    lights = {}

    def light(p):
        if p not in lights:
            lights[p] = E3.LightMap(p)
        return lights[p]
    e2 = json.loads(Path(job["e2_state"]).read_text())
    extra, pre_s = LK.e2_rivals(e2, st, job["t_end"], job["leader_floor"], light, window_s=job["win"])
    cands = job["cands"] + [{k: x[k] for k in ("candidate_id", "floor", "rec")} for x in extra]
    for c in cands:
        light(c["rec"]["light"])
    try:
        res = LK.E7.evaluate(st, cands, lights, job["t_end"], "STRUCTURE_ONLY", window_s=job["win"], structural=True, semantic_ok=None)
    except Exception as ex:                                                     # noqa: BLE001
        res = dict(final_decision="MORE_OBSERVATION_REQUIRED", why=f"E7 error {type(ex).__name__}: {ex}", results=[])
    keep = ("final_decision", "best", "runner_up", "gap_m", "ambiguity", "structurally_indistinguishable", "n_frames", "why")
    lead = job["cands"][0]["candidate_id"] if job["cands"] else None
    return dict(t_end=job["t_end"], res={k: res.get(k) for k in keep},
                leader_correction=next((r.get("correction") for r in res.get("results") or [] if r.get("candidate_id") == lead), None),
                e2_rivals=[(x["candidate_id"], x["floor"], x["prescreen"]) for x in extra], prescreen_s=pre_s,
                e7_s=round(time.time() - t0, 1))


HOT_C = 75.0      # field runs reached 73-75 C with CPU E2 + SigLIP; above this no extra worker processes (safety guard ends at 80 C)


def max_temp_c():
    temps = []
    for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(int(p.read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            pass
    return max(temps) if temps else None


def evaluate_windows(jobs, procs=2, cores=(6, 7)):
    """-> list of window results (same order as jobs). In-process (sequential, caller's low-priority thread) when the SoC is at
    or above HOT_C or a pool cannot start; otherwise `procs` worker processes on `cores`."""
    if not jobs:
        return []
    t = max_temp_c()
    if t is not None and t >= HOT_C:
        print(f"[e7_retro] SoC {t:.1f} C >= {HOT_C} C -> in-process, no extra workers", flush=True)
        return [_one(j) for j in jobs]
    try:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max(1, min(procs, len(jobs))), mp_context=mp.get_context("spawn"),
                                 initializer=_init, initargs=(tuple(cores),)) as ex:
            return list(ex.map(_one, jobs))
    except Exception as e:                                                      # noqa: BLE001
        print(f"[e7_retro] pool unavailable ({type(e).__name__}: {e}) -> in-process", flush=True)
        return [_one(j) for j in jobs]
