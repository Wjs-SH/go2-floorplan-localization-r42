"""exp_archive.py — LIVE2 per-experiment immutable archive of the VOLATILE run records.

Field run 2026-10-21: the second experiment overwrote the first one's /dev/shm/live2_store (LiDAR store) and
runtime/fastlio/Log (FAST-LIO mat_pre), so EXP1 could not be replayed. These sources are reused by every run.

  archive_run(rd, sources)   after every module of the run has stopped (no writer): COPY each source that provably belongs
                             to this run into <rd>/archive/, hash source and copy, write MANIFEST.json + SHA256SUMS +
                             ARCHIVE_VERIFIED.json. Sources are only read. A source that does not belong to the run (time
                             window) or does not exist is listed as missing, never faked.
  start_gate(exps, sources)  before a new experiment: the latest ended experiment must be archive-verified; if not, archive it
                             now (nothing has run since, so its sources are still intact). Fails -> refuse the start.

Not recorded by LIVE2 at all (stated, not invented): ROS bag / raw /lidar_points (RECORDER_NOT_AVAILABLE — LIVE2 never adds a
raw LiDAR subscriber), hub colour/depth ring (overwritten slot buffer; the E5 run keeps its own event frames).
"""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

ARCHIVE_DIR, VERIFIED = "archive", "ARCHIVE_VERIFIED.json"
SLACK_S = 10.0       # all modules are stopped BEFORE ended_wall is set; 2026-10-21 runs were only 50 s apart, a wide slack
                     # would take the NEXT run's store (found when validating on the real EXP1/EXP2 sources)
NOT_RECORDED = {"ros_bag_raw_lidar": "RECORDER_NOT_AVAILABLE (LIVE2 adds no raw /lidar_points subscriber)",
                "hub_rgb_depth_ring": "NOT_ARCHIVED_BY_POLICY (volatile slot ring; E5 run keeps its event frames)"}


def sha256(p, bs=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def _stamp_of(name):
    """'live1_observe_20261021_221926' / '20261021_221918' -> epoch (local time) or None."""
    import re
    m = re.search(r"(\d{8}_\d{6})", name)
    return time.mktime(time.strptime(m.group(1), "%Y%m%d_%H%M%S")) if m else None


def live_sources(L, store=Path("/dev/shm/live2_store"), hub=Path("/dev/shm/live1_hub")):
    """The run-reused locations of the LIVE2 workspace L (paths only; ownership is decided per run in archive_run)."""
    L = Path(L)
    return dict(store=Path(store), fastlio_log=L / "runtime" / "fastlio" / "Log", field_eval=L / "runtime" / "field_eval",
                observe_runs=L / "candidate" / "experiments", hub_index=Path(hub) / "index.json",
                e6_aim_out=L / "runtime" / "e6_pantilt_out",          # 2026-10-22: student aim console log of the E6 run
                code=[L / "candidate" / "live1", L / "candidate" / "live2"])


def _window(exp, next_t0=None):
    t0 = float(exp.get("created_wall") or 0)
    t1 = float(exp.get("ended_wall") or time.time()) + SLACK_S
    return t0, t1 if next_t0 is None else min(t1, float(next_t0))


def _pick(sources, exp, next_t0=None):
    """-> {name: (src_path, kind)} for sources that belong to this run, and {name: why} for the ones that do not."""
    t0, t1 = _window(exp, next_t0)
    pick, miss = {}, {}
    st = sources["store"]
    try:
        sess = json.loads((st / "session.json").read_text())
        s0 = float(sess.get("started_wall") or 0)
        (pick.__setitem__("live2_store", (st, "dir")) if t0 <= s0 <= t1 else
         miss.__setitem__("live2_store", f"OVERWRITTEN_OR_OTHER_RUN (session started {s0:.0f} outside {t0:.0f}..{t1:.0f})"))
    except (OSError, ValueError):
        miss["live2_store"] = "MISSING"
    fl = sources["fastlio_log"]
    mp = fl / "mat_pre.txt"
    if mp.exists() and t0 <= mp.stat().st_mtime <= t1:
        pick["fastlio_Log"] = (fl, "dir")
    else:
        miss["fastlio_Log"] = "MISSING" if not mp.exists() else "OVERWRITTEN_OR_OTHER_RUN (mat_pre.txt mtime outside the run)"
    for name, root, pre in (("field_eval", sources["field_eval"], ""), ("observe_run", sources["observe_runs"], "live1_observe_")):
        c = sorted(d for d in (root.glob(pre + "2*") if root.exists() else []) if d.is_dir() and _stamp_of(d.name) and t0 <= _stamp_of(d.name) <= t1)
        if c:
            pick[name] = (c[-1], "dir")
        else:
            miss[name] = "MISSING (no run directory in the experiment window)"
    ao = sources.get("e6_aim_out")              # optional: only runs with E6 (approval B) have one; absence is not a failure
    if ao is not None:
        c = sorted(d for d in (ao.glob("2*") if ao.exists() else []) if d.is_dir() and _stamp_of(d.name) and t0 <= _stamp_of(d.name) <= t1)
        if c:
            pick["e6_aim_console"] = (c[-1], "dir")
        else:
            miss["e6_aim_console"] = "NOT_RUN (no E6 aim console in the experiment window)"
    hi = sources["hub_index"]
    if hi.exists() and t0 <= hi.stat().st_mtime <= t1:
        pick["hub_index.json"] = (hi, "file")
    else:
        miss["hub_index.json"] = "MISSING_OR_OTHER_RUN"
    return pick, miss


def _files(src, kind):
    if kind == "file":
        return [(src, Path(src.name))]
    return sorted((p, p.relative_to(src)) for p in src.rglob("*") if p.is_file())


def archive_run(rd, sources, exp=None, clock=time.time, next_t0=None):
    """Copy this run's volatile sources into <rd>/archive and verify. Returns the ARCHIVE_VERIFIED record (also written)."""
    rd = Path(rd)
    exp = exp or json.loads((rd / "experiment.json").read_text())
    ad = rd / ARCHIVE_DIR
    t_start = clock()
    pick, miss = _pick(sources, exp, next_t0)
    rows, bad = [], []
    ad.mkdir(parents=True, exist_ok=True)
    for name, (src, kind) in pick.items():
        for p, rel in _files(src, kind):
            dst = ad / name / rel
            try:
                h0 = sha256(p)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
                h1 = sha256(dst)
            except OSError as e:
                bad.append(dict(file=str(p), why=f"{type(e).__name__}: {e}"))
                continue
            if h0 != h1:
                bad.append(dict(file=str(p), why="HASH_MISMATCH_AFTER_COPY (source changed during copy?)"))
            rows.append(dict(path=str((Path(name) / rel)), src=str(p), bytes=dst.stat().st_size, sha256=h1))
    code = {}
    for d in sources.get("code") or []:
        for p in sorted(Path(d).rglob("*")):
            if p.is_file() and p.suffix in (".py", ".html", ".json", ".yaml", ".sh") and "__pycache__" not in p.parts:
                code[str(p)] = sha256(p)
    cfg = None
    try:
        sess = json.loads((Path(sources["store"]) / "session.json").read_text())
        cfg = dict(path=sess.get("fastlio_cfg"), sha256=sess.get("fastlio_cfg_sha256"))
    except (OSError, ValueError):
        pass
    (ad / "SHA256SUMS").write_text("".join(f"{r['sha256']}  {r['path']}\n" for r in rows))
    manifest = dict(kind="LIVE2_EXPERIMENT_ARCHIVE", exp_id=exp.get("exp_id"), window=_window(exp, next_t0), archived_wall=clock(),
                    files=rows, sources={k: str(v[0]) for k, v in pick.items()}, missing=miss, not_recorded=NOT_RECORDED,
                    copy_errors=bad, code_sha256=code, fastlio_cfg=cfg, note="copies only; sources were read, never moved or modified")
    (ad / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    nbytes = sum(r["bytes"] for r in rows)
    rec = dict(exp_id=exp.get("exp_id"), verified=not bad, n_files=len(rows), bytes=nbytes,
               sha256_of_SHA256SUMS=sha256(ad / "SHA256SUMS"), missing=sorted(miss), copy_errors=len(bad),
               took_s=round(clock() - t_start, 2), free_gb_after=round(shutil.disk_usage(str(ad)).free / 2 ** 30, 1),
               written_wall=clock())
    tmp = ad / (VERIFIED + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, ad / VERIFIED)
    return rec


def verified(rd):
    """True only if ARCHIVE_VERIFIED says so AND the SHA256SUMS file is the one it vouched for (no later edit)."""
    ad = Path(rd) / ARCHIVE_DIR
    try:
        rec = json.loads((ad / VERIFIED).read_text())
        return bool(rec.get("verified")) and sha256(ad / "SHA256SUMS") == rec.get("sha256_of_SHA256SUMS"), rec
    except (OSError, ValueError):
        return False, None


def start_gate(exps, sources, archive_now=True):
    """-> (ok, why, record). The latest ENDED experiment must be archive-verified before a new run reuses the sources."""
    exps = Path(exps)
    last = None
    for d in sorted(exps.glob("X*_*"), reverse=True):
        try:
            x = json.loads((d / "experiment.json").read_text())
        except (OSError, ValueError):
            continue
        if str(x.get("state", "")).startswith("ENDED"):
            last = (d, x)
        break                                   # only the newest experiment directory matters
    if last is None:
        return True, "NO_PREVIOUS_ENDED_EXPERIMENT", None
    d, x = last
    ok, rec = verified(d)
    if ok:
        return True, f"ARCHIVE_VERIFIED {x.get('exp_id')}", rec
    if not archive_now:
        return False, f"ARCHIVE_NOT_VERIFIED {x.get('exp_id')}", rec
    try:
        rec = archive_run(d, sources, x)
    except Exception as e:                       # noqa: BLE001  any failure blocks the start (never silently skipped)
        return False, f"ARCHIVE_FAILED {x.get('exp_id')} {type(e).__name__}: {e}", None
    ok, rec2 = verified(d)
    return (True, f"ARCHIVED_AT_START {x.get('exp_id')}", rec2) if ok else (False, f"ARCHIVE_NOT_VERIFIED {x.get('exp_id')} {rec}", rec)
