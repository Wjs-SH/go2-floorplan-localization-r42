#!/usr/bin/env python3
"""Asynchronous reader: READ_ASYNC -> LOCALIZE_ASYNC for captured observations (decoupled from the stop).

R1 (2026-10-16): job handling moved to r1_jobs.ReaderCore (OCR-first, OCR cache, fair bounded VLM wait, finite terminal
reasons, identity-checked jobs). This module only wires the real engines (created lazily, never at import):
  1. OCR once per input (cache key = image sha256 + crop + preprocessing version + engine)  -> published immediately
  2. VLM only if the OCR result is not confident AND the resource guard admits it (policy unchanged: need 4200 MB,
     reserve 2000 MB, temperature/GPU limits of resource_guard.admit_gpu_job), within a fixed per-job deadline
  3. fusion (no automatic tie-break; conflict -> READING_CONFLICT)
  4. E5 packet always at terminal; E6 only for SEMANTIC_ACCEPTED + CAPTURE_QUALITY_OK + real pan/tilt capture
  5. E7: NOT_WIRED (R1 does not connect E7)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
for p in (HERE, SRC / "sci_scheduler", SRC / "sci_semantic_localization", SRC / "sci_common"):
    sys.path.insert(0, str(p))

from semantic import classify_text, fuse, is_location_type  # noqa: E402,F401  (fuse/is_location_type kept for wrappers)
import r1_jobs as J  # noqa: E402

CALIB = Path(os.environ.get("SCI_PANTILT_CALIB", str(Path.home() / "pantilt_calib" / "configs" / "pantilt_calib_multi_20260928.json")))
R1_READER = True                                  # marker: identity/deadline reader (used by before/after tests)


class Reader:
    def __init__(self, a, cfg=None):
        self.a = a
        self.cfg = cfg or J.load_cfg(getattr(a, "config", None))
        from resource_guard import ResourceMonitor, admit_gpu_job
        from ocr import OcrEngine
        self.mon = ResourceMonitor(period=1.0)
        self.mon.start()
        self.ocr = OcrEngine(use_easy=a.easyocr)       # EasyOCR (Korean) is opt-in: +~2 GB (CUDA context), see bench_ocr.json
        self.vlm = self.make_vlm() if a.vlm != "off" else None
        if self.vlm and a.vlm == "resident":
            self.vlm.start(wait_s=self.cfg["vlm_start_timeout_s"])
        self._admit_gpu_job = admit_gpu_job
        self.core = J.ReaderCore(a.run, self.cfg, ocr_fn=lambda img: self.ocr.read(img, classify_text),
                                 engine_info=self.ocr.describe(), vlm_fn=self.vlm_fn if self.vlm else None,
                                 admit_fn=self.admit, localize_fn=self.localize, once=a.once)

    def make_vlm(self):
        from vlm import VlmServer
        return VlmServer()

    def admit(self, need_mb):
        return self._admit_gpu_job(self.mon.last, 0 if self.vlm.healthy() else need_mb, reserve_mb=self.cfg["vlm_reserve_mb"])

    def vlm_fn(self, img, timeout_s):
        self.vlm.start(wait_s=self.cfg["vlm_start_timeout_s"])
        try:
            return self.vlm.ask(img, max_side=self.cfg["vlm_max_side"], timeout=timeout_s)
        except Exception as e:                       # urllib socket timeout -> VLM_TIMEOUT
            if "timed out" in str(e).lower() or isinstance(e, TimeoutError):
                raise TimeoutError(str(e))
            raise

    def localize(self, st, rec, packet_only=False):
        """Packet always (records the outcome); E6 only when the core allowed it (rec['e6']['status']=='PENDING')."""
        out = {}
        job = st["job"]
        trial = Path(job.get("trial") or "")
        if not job.get("trial") or not (trial / "runner" / "events.jsonl").exists():
            return dict(packet=None, packet_null_reason="no trial/runner/events.jsonl for this capture")
        rd = trial / "reading"
        rd.mkdir(parents=True, exist_ok=True)
        J.atomic_write_json(rd / f"e5_reading_{J.safe(st['capture_id'])}.json", rec)
        from e5_e7 import e5_packet
        pk = e5_packet.build_packet(trial, CALIB, floor_hint=self.a.floor_hint, reading=rec, identity=job)
        J.atomic_write_json(trial / f"E5_OBSERVATION_PACKET_{J.safe(st['capture_id'])}.json", pk)
        out["packet"] = str(trial / f"E5_OBSERVATION_PACKET_{J.safe(st['capture_id'])}.json")
        if not packet_only and self.a.e6_floors and rec["e6"]["status"] == "PENDING":
            from e5_e7 import e6_region
            r6 = e6_region.run_e6(pk, self.a.e6_floors, trial / "e6", self.a.building)
            out["e6"] = dict(status=r6["status"], n_regions=r6["n_regions"],
                             floors=[dict(floor=h["floor"], p=h["p_floor"]) for h in r6["floor_hypotheses"]])
        elif rec["e6"]["status"] == "PENDING":
            out["e6"] = dict(status="SKIPPED", why="no --e6-floors")
        return out

    def loop(self):
        print(f"reader ready (ocr load {self.ocr.load_s:.1f}s, engine {self.ocr.describe()['engine']}), "
              f"watching {self.core.jd / 'pending'}", flush=True)
        while True:
            try:
                r = self.core.step()
                if any(r.values()):
                    print(json.dumps(r), flush=True)
            except Exception:                         # noqa: BLE001  a bad job must not stop the reader
                traceback.print_exc()
            if self.vlm is not None and self.a.vlm == "on_demand" and self.vlm.p is not None \
                    and time.time() - self.vlm.last_use > self.a.vlm_idle_s:
                self.vlm.stop()
            if self.a.once and self.core.idle():
                break
            time.sleep(0.3)
        if self.vlm is not None and self.a.vlm != "resident":
            self.vlm.stop()


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="experiments/<run> folder of e5_node")
    ap.add_argument("--vlm", default="on_demand", choices=["off", "on_demand", "resident"])
    ap.add_argument("--vlm-idle-s", type=float, default=120.0)
    ap.add_argument("--easyocr", action="store_true", help="enable EasyOCR ko+en (GPU, ~2 GB RSS); default PP-OCRv4 only")
    ap.add_argument("--e6-floors", nargs="*", default=None)
    ap.add_argument("--building", default="kci")
    ap.add_argument("--floor-hint", default=None, help="E8 rule: must stay unset in blinded evaluation")
    ap.add_argument("--once", action="store_true", help="process pending jobs and exit (no VLM retry)")
    ap.add_argument("--config", default=None, help="reader config (default config/r1_reader.json)")
    # legacy options accepted but superseded by the bounded config (kept so old launch lines fail loudly in the log)
    ap.add_argument("--retry-s", type=float, default=None, help="SUPERSEDED by config check_interval_s")
    ap.add_argument("--retry-window-s", type=float, default=None, help="SUPERSEDED by config resource_wait_s")
    return ap


def main(reader_cls=None):
    a = build_argparser().parse_args()
    if a.retry_s is not None or a.retry_window_s is not None:
        print("reader: --retry-s/--retry-window-s are superseded by config/r1_reader.json (bounded per-job deadline)", flush=True)
    import signal
    r = (reader_cls or Reader)(a)

    def _term(*_):                                # llama-server runs in its own session: stop it explicitly
        if r.vlm is not None:
            r.vlm.stop()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    r.loop()


if __name__ == "__main__":
    main()
