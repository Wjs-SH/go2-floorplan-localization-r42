"""R1 reader job core: OCR-first, OCR cache, fair bounded VLM wait, finite terminal states (no model access here).

All engines are injected:
  ocr_fn(img_bgr) -> dict(attempts, merged=[(text, conf)], result=classify_text(...), input_px)
  vlm_fn(img_bgr, timeout_s) -> dict(raw, parsed(SCI fusion form), sec, image_px)      (TimeoutError = VLM_TIMEOUT)
  admit_fn(need_mb) -> (ok: bool, reason: str)                                         (resource guard, unchanged policy)
  localize_fn(state, done_record) -> dict(packet=..., e6=..., e7=...)                  (only after a terminal reading)
  clock() -> monotonic seconds of THIS boot (deadlines);  wall() -> unix seconds (logs only)

Files under <run>/jobs/ (reader-owned except pending/):
  pending/*.json        written by E5 (input; moved to ingested/ after validation, never deleted)
  state/<key>.json      one record per work item (event_key + capture_id), atomic replace
  history/<key>.jsonl   append-only: ingest, ocr, admission checks, inference attempts, terminal, late results
  ocr_cache/<k>.json    OCR result keyed by input sha256 + crop/resize/rotation + preprocessing version + engine revision
  done/<event>__<capture>.json   per-capture result (revision increases until terminal, never changed after terminal)
  done/<event>.json     current result of the event = latest ingested capture of that event
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sci_common"))
sys.path.insert(0, str(HERE))
from r1_contracts import (BOOT_ID, SCHEMA_VERSION, CaptureStatus, GeoReason, ReadReason, SemanticStatus,  # noqa: E402
                          append_jsonl, atomic_write_json, read_json)
from semantic import fuse, label_schema  # noqa: E402

IDENTITY_FIELDS = ("schema_version", "run_id", "event_key", "capture_id")
WAITING, OCR_PENDING, TERMINAL = "WAIT_VLM", "OCR_PENDING", "TERMINAL"
DEFAULT_CFG = Path(__file__).resolve().parents[2] / "config" / "r1_reader.json"


def load_cfg(path=None) -> dict:
    cfg = read_json(path or DEFAULT_CFG)
    if not cfg:
        raise FileNotFoundError(f"reader config missing: {path or DEFAULT_CFG}")
    cfg = dict(cfg)
    cfg["vlm_need_mb"] = int(os.environ.get("SCI_VLM_NEED_MB", cfg["vlm_need_mb"]))          # legacy env override kept
    cfg["vlm_reserve_mb"] = int(os.environ.get("SCI_VLM_RESERVE_MB", cfg["vlm_reserve_mb"]))
    return cfg


def safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s).replace("#", "__"))


def file_sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ocr_cache_key(image_sha256, crop, resize, rotation, preprocess_version, engine_info, options) -> str:
    blob = json.dumps(dict(image_sha256=image_sha256, crop=crop, resize=resize, rotation=rotation,
                           preprocess_version=preprocess_version, engine=engine_info, options=options), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def legacy_reason(done_record: dict) -> str:
    """Analysis-only view of pre-R1 done files: their NO_ROOM_LABEL mixes no-text / parser / capture-quality / resource
    cases, so it is reported as ambiguous instead of being mapped onto an R1 reason."""
    st = (done_record or {}).get("status")
    if done_record and "reason_code" in done_record:
        return done_record["reason_code"]
    if st == "NO_ROOM_LABEL":
        return "LEGACY_NO_ROOM_LABEL_AMBIGUOUS"
    return f"LEGACY_{st}"


class ReaderCore:
    def __init__(self, run_dir, cfg, ocr_fn, engine_info, vlm_fn=None, admit_fn=None, localize_fn=None,
                 clock=time.monotonic, wall=time.time, boot_id=BOOT_ID, once=False, load_image=None):
        self.run = Path(run_dir)
        self.jd = self.run / "jobs"
        for sub in ("pending", "ingested", "state", "history", "ocr_cache", "done"):
            (self.jd / sub).mkdir(parents=True, exist_ok=True)
        self.cfg, self.ocr_fn, self.engine_info = cfg, ocr_fn, dict(engine_info)
        self.vlm_fn, self.admit_fn, self.localize_fn = vlm_fn, admit_fn, localize_fn
        self.clock, self.wall, self.boot_id, self.once = clock, wall, boot_id, once
        self.load_image = load_image or _cv2_imread
        self.counters = dict(ocr_runs=0, ocr_cache_hits=0, admission_checks=0, inference_attempts=0, ocr_errors=0,
                             vlm_async_started=0, vlm_logical_timeouts=0, vlm_late_results=0)
        self.states = {}
        # E5RT2: optional asynchronous VLM (one inference in flight on a worker thread). The reader loop keeps ingesting,
        # running OCR and applying every deadline while the VLM runs; a logical timeout ends the job, the running call is
        # NOT assumed stopped and its late result is kept as history only (late_result).
        self.vlm_async = bool(self.cfg.get("vlm_async", False))
        self._vlm_exec, self._vlm_running = None, None    # (work_key, future, deadline_mono)
        self._restore()

    # ------------------------------------------------------------------------------------------- persistence
    def _key(self, event_key, capture_id):
        return f"{safe(event_key)}__{safe(capture_id)}"

    def _save(self, st):
        atomic_write_json(self.jd / "state" / f"{st['work_key']}.json", st)

    def _hist(self, st_or_key, kind, **kw):
        key = st_or_key["work_key"] if isinstance(st_or_key, dict) else st_or_key
        append_jsonl(self.jd / "history" / f"{key}.jsonl",
                     dict(kind=kind, boot_id=self.boot_id, mono_s=self.clock(), wall=self.wall(), **kw))

    def _now(self):
        return dict(boot_id=self.boot_id, mono_s=self.clock(), wall=self.wall())

    def _restore(self):
        """Reader restart: reload state; on a different boot the monotonic deadline is meaningless -> reissue ONE new
        bounded deadline (max_reboot_reissues); further reboots end the wait (SKIPPED_RESOURCE). Never infinite."""
        for p in sorted((self.jd / "state").glob("*.json")):
            st = read_json(p)
            if not st:
                continue
            self.states[st["work_key"]] = st
            if st["status"] != TERMINAL and st.get("deadline_boot") != self.boot_id:
                st["reboot_reissues"] = st.get("reboot_reissues", 0) + 1
                if st["reboot_reissues"] > self.cfg["max_reboot_reissues"]:
                    self._hist(st, "reboot_reissue_refused", n=st["reboot_reissues"])
                    self._terminal(st, ReadReason.SKIPPED_RESOURCE, detail="deadline lost across reboots (bounded)")
                    continue
                st["deadline_mono"] = self.clock() + self.cfg["resource_wait_s"]
                st["deadline_boot"] = self.boot_id
                st["next_check_mono"] = self.clock()
                self._hist(st, "deadline_reissued_after_reboot", new_deadline_mono=st["deadline_mono"])
                self._save(st)

    # ------------------------------------------------------------------------------------------- ingest
    def ingest(self):
        n = 0
        for p in sorted((self.jd / "pending").glob("*.json")):
            job = read_json(p)
            dest = self.jd / "ingested" / p.name
            if dest.exists():
                dest = self.jd / "ingested" / f"{p.stem}.dup{int(self.wall() * 1000)}.json"
            os.replace(p, dest)
            missing = [f for f in IDENTITY_FIELDS if not (job or {}).get(f)]
            if missing:
                key = f"INVALID__{safe(p.stem)}__{int(self.wall() * 1000)}"
                st = self._new_state(key, job or {}, event_key=(job or {}).get("event_key"), capture_id=(job or {}).get("capture_id"))
                self._hist(st, "ingest", file=str(dest))
                self._terminal(st, ReadReason.INPUT_INVALID, detail=f"missing identity fields {missing} (legacy job without identity is refused)")
                n += 1
                continue
            key = self._key(job["event_key"], job["capture_id"])
            if key in self.states:                                   # duplicate file / restart re-delivery: idempotent
                self._hist(key, "duplicate_ignored", file=str(dest))
                continue
            st = self._new_state(key, job, event_key=job["event_key"], capture_id=job["capture_id"])
            self._hist(st, "ingest", file=str(dest), job_created_wall=job.get("created"))
            self._save(st)
            n += 1
        return n

    def _new_state(self, key, job, event_key, capture_id):
        now = self._now()
        st = dict(schema_version=SCHEMA_VERSION, work_key=key, event_key=event_key, capture_id=capture_id,
                  run_id=job.get("run_id"), request_id=job.get("request_id"), job=job, status=OCR_PENDING,
                  first_seen=now, deadline_mono=now["mono_s"] + self.cfg["resource_wait_s"], deadline_boot=self.boot_id,
                  reboot_reissues=0, admission_checks=0, inference_attempts=0, last_vlm_try_mono=None,
                  next_check_mono=now["mono_s"], ocr=None, vlm=dict(status="NOT_STARTED", attempts=[]),
                  reason_code=None, terminal=False, result_revision=0, late_results=0)
        self.states[key] = st
        return st

    # ------------------------------------------------------------------------------------------- OCR
    def _ocr_input(self, st, img):
        job, h, w = st["job"], img.shape[0], img.shape[1]
        roi = job.get("target_roi")
        if roi and job.get("target_roi_verified"):
            x0, y0, x1, y1 = roi
            mx, my = (x1 - x0) * self.cfg["roi_margin_frac"], (y1 - y0) * self.cfg["roi_margin_frac"]
            box = [int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(w, x1 + mx)), int(min(h, y1 + my))]
            if box[2] > box[0] and box[3] > box[1]:
                return img[box[1]:box[3], box[0]:box[2]], dict(kind="verified_target_roi+margin", box=box, roi=roi,
                                                              margin_frac=self.cfg["roi_margin_frac"])
        why = "no target_roi in job" if not roi else ("target_roi not verified" if not job.get("target_roi_verified") else "empty roi")
        return img, dict(kind="full_image", box=[0, 0, w, h], roi_null_reason=why)

    def _do_ocr(self, st):
        job = st["job"]
        cs = job.get("capture_status")
        if cs == CaptureStatus.CAPTURE_QUALITY_FAIL:
            return self._terminal(st, ReadReason.CAPTURE_QUALITY_FAIL,
                                  detail="capture quality gate failed upstream; no OCR success claimed, not sent to E6")
        if cs in (CaptureStatus.NO_CAPTURE, CaptureStatus.FRESH_CAPTURE_UNAVAILABLE) or not job.get("image"):
            return self._terminal(st, ReadReason.INPUT_INVALID, detail=f"no usable capture (capture_status={cs})")
        try:
            sha = file_sha256(job["image"])
        except OSError as e:
            return self._terminal(st, ReadReason.INPUT_INVALID, detail=f"capture image unreadable: {type(e).__name__}")
        if job.get("image_sha256") and job["image_sha256"] != sha:
            return self._terminal(st, ReadReason.INPUT_INVALID, detail="image sha256 differs from the job manifest")
        img = self.load_image(job["image"])
        if img is None:
            return self._terminal(st, ReadReason.INPUT_INVALID, detail="capture image could not be decoded")
        if self.clock() < st.get("ocr_retry_mono", 0.0):
            return st                                               # bounded back-off after an OCR engine error
        crop, prov = self._ocr_input(st, img)
        options = dict(classify="semantic.classify_text")
        key = ocr_cache_key(sha, prov["box"], None, 0, self.cfg["preprocess_version"], self.engine_info, options)
        cpath = self.jd / "ocr_cache" / f"{key}.json"
        cached = read_json(cpath)
        crop_path = self._save_crop(st, crop)
        if cached:
            self.counters["ocr_cache_hits"] += 1
            o, t0, t1, hit = cached["ocr"], cached["t_start"], cached["t_end"], True
        else:
            t0 = self._now()
            try:
                o = self.ocr_fn(crop)
            except Exception as e:                                  # noqa: BLE001  E5RT2: one bad job never blocks others
                self.counters["ocr_errors"] += 1
                st["ocr_errors"] = st.get("ocr_errors", 0) + 1
                self._hist(st, "ocr_error", n=st["ocr_errors"], error=f"{type(e).__name__}: {e}")
                if st["ocr_errors"] >= int(self.cfg.get("max_ocr_errors", 2)):
                    return self._terminal(st, ReadReason.OCR_ENGINE_ERROR,
                                          detail=f"OCR raised {st['ocr_errors']}x ({type(e).__name__}: {e}); bounded, no further retry")
                st["ocr_retry_mono"] = self.clock() + float(self.cfg.get("ocr_retry_s", 2.0))
                st["first_seen_ocr_rotated"] = True
                self._save(st)
                return st
            t1 = self._now()
            self.counters["ocr_runs"] += 1
            atomic_write_json(cpath, dict(cache_key=key, ocr=o, t_start=t0, t_end=t1, engine=self.engine_info))
            hit = False
        lines = [(t, c) for t, c in (o.get("merged") or []) if str(t).strip()]
        fz = fuse(o["result"], None)
        if not lines:
            ocr_reason = ReadReason.NO_TEXT_DETECTED
        elif fz["label_alternatives"]:
            ocr_reason = ReadReason.OCR_ONLY
        else:
            ocr_reason = ReadReason.PARSE_REJECTED
        st["ocr"] = dict(engine=self.engine_info, cache_key=key, cache_hit=hit, t_start=t0, t_end=t1,
                         input=dict(image=job["image"], image_sha256=sha, crop=prov, crop_file=crop_path,
                                    crop_sha256=file_sha256(crop_path) if crop_path else None, input_px=o.get("input_px")),
                         lines=lines, raw_text=[t for t, _ in lines], attempts=o.get("attempts"), result=o["result"],
                         fusion_ocr_only=fz, reason=ocr_reason)
        self._hist(st, "ocr", cache_hit=hit, cache_key=key, reason=ocr_reason, n_lines=len(lines))
        need, why = self.want_vlm(o, fz)
        if not need:
            st["vlm"]["status"] = ReadReason.VLM_NOT_NEEDED
            return self._terminal(st, ocr_reason, detail=why, fusion=fz)
        if self.vlm_fn is None:
            st["vlm"]["status"] = "DISABLED"
            return self._terminal(st, ocr_reason, detail="vlm disabled (--vlm off)", fusion=fz)
        st["status"], st["vlm"]["status"], st["vlm"]["why_needed"] = WAITING, ReadReason.DEFERRED_RESOURCE, why
        self._save(st)
        self._publish(st, fusion=fz)                  # OCR result is visible now, independent of the VLM
        return st

    def _save_crop(self, st, crop):
        try:
            import cv2
        except ImportError:
            return None
        p = Path(st["job"].get("trial") or (self.run / "trials" / safe(st["event_key"]))) / "reading" / f"ocr_input_{safe(st['capture_id'])}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            return None
        with open(p, "wb") as f:
            f.write(buf.tobytes())
        return str(p)

    @staticmethod
    def want_vlm(ocr_res, fz):
        """SCI OCR-first rule (restored; the pan/tilt 'always VLM' override is removed)."""
        r = ocr_res["result"]
        if not r["rooms"]:
            return True, "ocr_no_room_number"
        if max(c for _, c in r["rooms"]) < 0.85:
            return True, "ocr_low_confidence"
        if r["sign_type"] not in ("room_number_plate", "lab_or_office_name_sign"):
            return True, f"ocr_type_{r['sign_type']}"
        return False, "ocr_confident"

    # ------------------------------------------------------------------------------------------- VLM wait
    def expire(self):
        """Applied to EVERY waiting job each loop, independent of which job is picked."""
        now, n = self.clock(), 0
        for st in list(self.states.values()):
            if st["status"] != WAITING:
                continue
            if now >= st["deadline_mono"]:
                self._terminal(st, ReadReason.SKIPPED_RESOURCE, detail=f"resource wait deadline {self.cfg['resource_wait_s']}s "
                                                                       f"from first enqueue reached (not renewed on retry)")
                n += 1
            elif st["admission_checks"] >= self.cfg["max_admission_checks"]:
                self._terminal(st, ReadReason.SKIPPED_RESOURCE, detail=f"max admission checks {self.cfg['max_admission_checks']} reached")
                n += 1
        return n

    def ready_waiting(self):
        """Fair order: never-tried first, then the longest since its last try; ties by first enqueue."""
        now = self.clock()
        w = [s for s in self.states.values() if s["status"] == WAITING and now >= s["next_check_mono"]]
        return sorted(w, key=lambda s: (s["last_vlm_try_mono"] is not None, s["last_vlm_try_mono"] or 0.0,
                                        s["first_seen"]["mono_s"]))

    def _vlm_step(self, st):
        now = self.clock()
        st["last_vlm_try_mono"], st["next_check_mono"] = now, now + self.cfg["check_interval_s"]
        side = self.cfg["vlm_max_side"]
        if side not in self.cfg.get("vlm_profiled_max_sides", []):
            ok, reason = False, f"resource_profile_unverified_{side}px"
        else:
            ok, reason = self.admit_fn(self.cfg["vlm_need_mb"]) if self.admit_fn else (False, "no_admission_function")
        st["admission_checks"] += 1
        self.counters["admission_checks"] += 1
        rec = dict(kind="admission_check", n=st["admission_checks"], ok=ok, reason=reason, mono_s=now)
        st["vlm"]["attempts"].append(rec)
        self._hist(st, "admission_check", n=st["admission_checks"], ok=ok, reason=reason)
        if not ok:
            if self.once:
                return self._terminal(st, ReadReason.SKIPPED_RESOURCE, detail=f"--once: no retry ({reason})")
            self._save(st)
            self._publish(st)
            return st
        st["inference_attempts"] += 1
        self.counters["inference_attempts"] += 1
        img = self.load_image(st["job"]["image"])
        t0 = self._now()
        if self.vlm_async:
            return self._vlm_submit(st, img, t0)
        try:
            v = self.vlm_fn(img, self.cfg["inference_timeout_s"])
        except TimeoutError as e:
            self._hist(st, "inference_attempt", n=st["inference_attempts"], ok=False, error="timeout")
            return self._terminal(st, ReadReason.VLM_TIMEOUT, detail=f"inference timeout {self.cfg['inference_timeout_s']}s: {e}")
        except Exception as e:                                      # noqa: BLE001  error is recorded, job ends
            self._hist(st, "inference_attempt", n=st["inference_attempts"], ok=False, error=f"{type(e).__name__}: {e}")
            st["vlm"]["status"] = "ERROR"
            return self._terminal(st, ReadReason.SKIPPED_RESOURCE, detail=f"vlm error {type(e).__name__}: {e}")
        return self._vlm_finish(st, v, t0, side)

    # ------------------------------------------------------------------------------------------- E5RT2 async VLM
    def _vlm_submit(self, st, img, t0):
        import concurrent.futures as cf
        if self._vlm_exec is None:
            self._vlm_exec = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="reader-vlm")
        fut = self._vlm_exec.submit(self.vlm_fn, img, self.cfg["inference_timeout_s"])
        dl = self.clock() + float(self.cfg["inference_timeout_s"])
        self._vlm_running = (st["work_key"], fut, dl, t0)
        st["status"], st["vlm"]["status"] = "VLM_RUNNING", "RUNNING"
        st["vlm"]["logical_deadline_mono"] = dl
        self.counters["vlm_async_started"] += 1
        self._hist(st, "inference_started_async", n=st["inference_attempts"], logical_deadline_mono=dl)
        self._save(st)
        self._publish(st)
        return st

    def _vlm_poll(self):
        """Owner side: finish / time out the single running VLM job. A timed-out call keeps running in its thread (a
        Future cannot stop it); when it ends, its result goes to late_result() history only."""
        if self._vlm_running is None:
            return 0
        key, fut, dl, t0 = self._vlm_running
        st = self.states.get(key)
        if fut.done():
            self._vlm_running = None
            if st is None or st["terminal"]:
                self.counters["vlm_late_results"] += 1
                try:
                    payload = dict(ok=True, sec=(fut.result() or {}).get("sec"))
                except Exception as e:                              # noqa: BLE001
                    payload = dict(ok=False, error=f"{type(e).__name__}: {e}")
                self.late_result(key, payload, source="vlm_async")
                return 1
            try:
                v = fut.result()
            except TimeoutError as e:
                self._hist(st, "inference_attempt", n=st["inference_attempts"], ok=False, error="timeout")
                self._terminal(st, ReadReason.VLM_TIMEOUT, detail=f"inference timeout {self.cfg['inference_timeout_s']}s: {e}")
                return 1
            except Exception as e:                                  # noqa: BLE001
                self._hist(st, "inference_attempt", n=st["inference_attempts"], ok=False, error=f"{type(e).__name__}: {e}")
                st["vlm"]["status"] = "ERROR"
                self._terminal(st, ReadReason.SKIPPED_RESOURCE, detail=f"vlm error {type(e).__name__}: {e}")
                return 1
            self._vlm_finish(st, v, t0, self.cfg["vlm_max_side"])
            return 1
        if st is not None and not st["terminal"] and self.clock() >= dl:
            self.counters["vlm_logical_timeouts"] += 1
            self._hist(st, "inference_logical_timeout", n=st["inference_attempts"], note="worker thread NOT assumed stopped")
            self._terminal(st, ReadReason.VLM_TIMEOUT, detail=f"logical deadline {self.cfg['inference_timeout_s']}s reached; "
                                                              f"a late result is history only")
            return 1
        return 0

    def _vlm_finish(self, st, v, t0, side):
        t1 = self._now()
        self._hist(st, "inference_attempt", n=st["inference_attempts"], ok=True, sec=v.get("sec"))
        st["vlm"].update(status="DONE", t_start=t0, t_end=t1, raw=v.get("raw"), parsed=v.get("parsed"),
                         image_px=v.get("image_px"), input=dict(image=st["job"]["image"], kind="full_image",
                                                                max_side=side, image_sha256=st["ocr"]["input"]["image_sha256"]))
        fz = fuse(st["ocr"]["result"], v.get("parsed"))
        if fz["status"] == "CONFLICT":
            reason = ReadReason.READING_CONFLICT
        elif fz["label_alternatives"]:
            reason = None                                           # accepted reading: no failure reason
        else:
            reason = st["ocr"]["reason"]
        return self._terminal(st, reason, detail=f"vlm fusion {fz['status']}", fusion=fz)

    # ------------------------------------------------------------------------------------------- results
    def _terminal(self, st, reason, detail="", fusion=None):
        if st.get("terminal"):
            return st
        if fusion is None and st.get("ocr"):
            fusion = st["ocr"]["fusion_ocr_only"]
        st.update(status=TERMINAL, terminal=True, reason_code=reason, detail=detail, t_terminal=self._now())
        if st["vlm"]["status"] == ReadReason.DEFERRED_RESOURCE and reason in (ReadReason.SKIPPED_RESOURCE,):
            st["vlm"]["status"] = ReadReason.SKIPPED_RESOURCE
        self._hist(st, "terminal", reason=reason, detail=detail)
        self._publish(st, fusion=fusion, final=True)
        self._save(st)
        return st

    def semantic_status(self, st, fusion):
        if not fusion or not fusion.get("label_alternatives") or st["reason_code"] == ReadReason.READING_CONFLICT:
            return SemanticStatus.SEMANTIC_REJECTED
        return SemanticStatus.SEMANTIC_ACCEPTED

    def _publish(self, st, fusion=None, final=False):
        job = st["job"]
        fusion = fusion or (st["ocr"] or {}).get("fusion_ocr_only")
        st["result_revision"] += 1
        sem = self.semantic_status(st, fusion) if st["terminal"] else None
        pantilt_ok = bool(job.get("pantilt_ok")) and job.get("capture_status") != CaptureStatus.CAPTURE_PROXY
        e6_allowed = bool(st["terminal"] and sem == SemanticStatus.SEMANTIC_ACCEPTED and pantilt_ok
                          and job.get("capture_status") == CaptureStatus.CAPTURE_QUALITY_OK)
        rec = dict(schema_version=SCHEMA_VERSION, event_key=st["event_key"], capture_id=st["capture_id"], run_id=st["run_id"],
                   request_id=st["request_id"], work_key=st["work_key"], terminal=st["terminal"], reason_code=st["reason_code"],
                   status=("TERMINAL" if st["terminal"] else "OCR_PUBLISHED_VLM_PENDING"),
                   capture_status=job.get("capture_status"), pantilt_ok=pantilt_ok,
                   counted_as_pantilt_success=pantilt_ok and job.get("capture_status") == CaptureStatus.CAPTURE_QUALITY_OK,
                   semantic_status=sem, labels=(fusion or {}).get("label_alternatives", []), fusion=fusion,
                   label_schema=label_schema((st["ocr"] or {}).get("lines"), fusion),
                   ocr=None if not st["ocr"] else {k: st["ocr"][k] for k in ("engine", "lines", "raw_text", "reason", "t_start",
                                                                             "t_end", "input", "cache_hit")},
                   vlm=dict(status=st["vlm"]["status"], admission_checks=st["admission_checks"],
                            inference_attempts=st["inference_attempts"], parsed=st["vlm"].get("parsed"),
                            input=st["vlm"].get("input")),
                   first_seen=st["first_seen"], deadline=dict(mono_s=st["deadline_mono"], boot_id=st["deadline_boot"],
                                                              reissues=st["reboot_reissues"]),
                   trial_id=job.get("trial_id"), generation_id=job.get("generation_id"),
                   observation_id=job.get("observation_id"),            # E5RT2: same hand-off id as the aim request
                   payload_sha256=job.get("payload_sha256"), payload_revision=job.get("payload_revision"),   # SEM1 P02
                   result_revision=st["result_revision"], detail=st.get("detail"), published=self._now(),
                   e6=dict(status="PENDING" if e6_allowed else "SKIPPED",
                           why=None if e6_allowed else self._e6_skip_why(st, sem, pantilt_ok)),
                   e7=dict(status=GeoReason.NOT_WIRED, why="E7 is not connected in R1 (no automatic E7 wiring)"))
        if final and e6_allowed and self.localize_fn is not None:
            try:
                rec.update(self.localize_fn(st, rec))
            except Exception as e:                                  # noqa: BLE001
                rec["localize_error"] = f"{type(e).__name__}: {e}"
        elif final and self.localize_fn is not None and st["reason_code"] != ReadReason.INPUT_INVALID:
            try:                                                    # packet only (records the failure), never E6
                rec.update(self.localize_fn(st, rec, packet_only=True))
            except Exception as e:                                  # noqa: BLE001
                rec["localize_error"] = f"{type(e).__name__}: {e}"
        base = safe(st["event_key"]) if st["event_key"] else st["work_key"]
        atomic_write_json(self.jd / "done" / f"{st['work_key']}.json", rec)
        latest = self._latest_capture(st["event_key"])
        if st["event_key"] and latest == st["work_key"]:
            atomic_write_json(self.jd / "done" / f"{base}.json", rec)
        st["last_published"] = rec["published"]
        return rec

    @staticmethod
    def _e6_skip_why(st, sem, pantilt_ok):
        cs = st["job"].get("capture_status")
        if not st["terminal"]:
            return "reading not terminal"
        if cs == CaptureStatus.CAPTURE_PROXY or not pantilt_ok:
            return "proxy / non-pantilt capture is diagnostic only (never consumed by E6)"
        if cs != CaptureStatus.CAPTURE_QUALITY_OK:
            return f"capture_status={cs}"
        return f"semantic_status={sem}"

    def _latest_capture(self, event_key):
        c = [s for s in self.states.values() if s["event_key"] == event_key]
        return max(c, key=lambda s: s["first_seen"]["mono_s"])["work_key"] if c else None

    def late_result(self, work_key, payload, source="vlm"):
        """A result arriving after the job became terminal is kept as history only (never changes the published result)."""
        st = self.states.get(work_key)
        if st is None:
            self._hist(work_key, "late_result_unknown_job", source=source, payload=payload)
            return False
        if st["terminal"]:
            st["late_results"] += 1
            self._hist(st, "late_result", source=source, payload=payload)
            self._save(st)
            return False
        return True

    def cancel(self, work_key, why="cancelled"):
        st = self.states.get(work_key)
        if st and not st["terminal"]:
            return self._terminal(st, ReadReason.CANCELLED, detail=why)
        return st

    # ------------------------------------------------------------------------------------------- loop
    def step(self):
        """One fair round: ingest all pending, expire all waiting, <=N new OCR, <=M VLM steps (fair order)."""
        out = dict(ingested=self.ingest(), expired=self.expire(), ocr=0, vlm=0, vlm_done=self._vlm_poll())
        now = self.clock()
        ocr_q = sorted((s for s in self.states.values() if s["status"] == OCR_PENDING and now >= s.get("ocr_retry_mono", 0.0)),
                       key=lambda s: (s.get("ocr_errors", 0), s["first_seen"]["mono_s"]))    # E5RT2: errored jobs go last
        for st in ocr_q[:self.cfg["max_ocr_per_loop"]]:
            self._do_ocr(st)
            out["ocr"] += 1
        if self._vlm_running is None:                                # at most one VLM inference in flight
            for st in self.ready_waiting()[:self.cfg["max_vlm_steps_per_loop"]]:
                self._vlm_step(st)
                out["vlm"] += 1
        out["expired"] += self.expire()
        atomic_write_json(self.jd / "reader_status.json",
                          dict(schema_version=SCHEMA_VERSION, now=self._now(), counters=self.counters, engine=self.engine_info,
                               jobs={k: dict(status=s["status"], reason_code=s["reason_code"], vlm=s["vlm"]["status"],
                                             admission_checks=s["admission_checks"], inference_attempts=s["inference_attempts"],
                                             result_revision=s["result_revision"]) for k, s in self.states.items()}))
        return out

    def idle(self):
        return not any(s["status"] != TERMINAL for s in self.states.values()) and not any((self.jd / "pending").glob("*.json"))


def _cv2_imread(p):
    import cv2
    return cv2.imread(str(p))
