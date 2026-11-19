"""R1-C reader tests (SIMULATED: fake OCR/VLM/admission/clock; no model, no HTTP, no device).

Tree under test = $SCI_WS (candidate). IDs: T10(reader side) T12(reader side) T13 T14 T15 T16 T17 T18.
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

WS = Path(os.environ["SCI_WS"])
for p in (WS / "src" / "sci_text_reading", WS / "src" / "sci_common", WS / "src" / "sci_semantic_localization",
          WS / "external" / "sci_event" / "e5_onboard"):
    sys.path.insert(0, str(p))

import r1_jobs as J  # noqa: E402
from semantic import classify_text  # noqa: E402
from r1_contracts import ReadReason, CaptureStatus, SemanticStatus  # noqa: E402

CFG = dict(J.load_cfg(WS / "config" / "r1_reader.json"), vlm_async=False)   # E5RT2: these tests pin the R1 synchronous VLM path; the async path (config default) is tested in tests/e5rt2


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeOcr:
    """Returns configured lines per image sha (or default); counts calls."""

    def __init__(self, lines=None):
        self.lines, self.calls, self.inputs = lines if lines is not None else [], 0, []

    def __call__(self, img):
        self.calls += 1
        self.inputs.append(img.shape[:2])
        return dict(attempts=[dict(engine="fake", lines=self.lines)], merged=list(self.lines),
                    result=classify_text(self.lines), input_px=list(img.shape[:2]))


class FakeVlm:
    def __init__(self, parsed=None, exc=None):
        self.parsed, self.exc, self.calls = parsed, exc, 0

    def __call__(self, img, timeout_s):
        self.calls += 1
        if self.exc:
            raise self.exc
        return dict(raw=json.dumps(self.parsed), parsed=self.parsed, sec=0.1, image_px=list(img.shape[:2]))


ENGINE = dict(engine="FAKE_OCR", rev="1")


def make_image(path, seed=0, size=(720, 1280)):
    rng = np.random.RandomState(seed)
    img = (rng.rand(size[0], size[1], 3) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), img)
    return path


def make_job(run, ev, cap="cap1", capture_status=CaptureStatus.CAPTURE_QUALITY_OK, pantilt_ok=True, seed=0, **kw):
    img = make_image(run / "captures" / f"ev{ev:03d}_{cap}.jpg", seed=seed)
    job = dict(schema_version="r1.1", run_id=run.name, event_id=ev, event_key=f"{run.name}#ev{ev:03d}", capture_id=cap,
               request_id=f"req-{ev}-{cap}", image=str(img), image_sha256=J.file_sha256(img), capture_status=capture_status,
               pantilt_ok=pantilt_ok, trial=str(run / "trials" / f"ev{ev:03d}"), created=1.0, trial_id="T1", generation_id=1)
    job.update(kw)
    p = run / "jobs" / "pending" / f"ev{ev:03d}_{cap}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(job))
    return job


def core(run, ocr=None, vlm=None, admit=None, clock=None, boot="BOOT-A", cfg=None, localize=None, once=False):
    return J.ReaderCore(run, cfg or CFG, ocr_fn=ocr or FakeOcr(), engine_info=ENGINE, vlm_fn=vlm,
                        admit_fn=admit or (lambda need: (False, "memory: fake avail 1100 MB")), localize_fn=localize,
                        clock=clock or Clock(), wall=lambda: 1.7e9, boot_id=boot, once=once)


def done_current(run, ev):
    return json.loads((run / "jobs" / "done" / f"{J.safe(f'{run.name}#ev{ev:03d}')}.json").read_text())


@pytest.fixture
def run(tmp_path):
    r = tmp_path / "kci_e5_test"
    (r / "jobs" / "pending").mkdir(parents=True)
    return r


# ---------------------------------------------------------------------------------------------------------- T13
def test_T13_fair_deferred_and_all_terminal_within_deadline(run):
    clk = Clock()
    c = core(run, ocr=FakeOcr([]), vlm=FakeVlm(), clock=clk)
    for ev in (1, 2, 3):
        make_job(run, ev)
    tried_order, t_start = [], clk.t
    for _ in range(200):
        before = {k: s["admission_checks"] for k, s in c.states.items()}
        c.step()
        tried_order += [k for k, s in c.states.items() if s["admission_checks"] > before.get(k, 0)]
        clk.t += 1.0
        if all(s["terminal"] for s in c.states.values()):
            break
    assert len(c.states) == 3 and all(s["terminal"] for s in c.states.values())
    # fairness: the first three admission checks went to three DIFFERENT jobs (no head-of-line monopoly)
    assert len(set(tried_order[:3])) == 3, tried_order
    counts = [s["admission_checks"] for s in c.states.values()]
    assert max(counts) - min(counts) <= 1, counts
    for s in c.states.values():
        assert s["reason_code"] == ReadReason.SKIPPED_RESOURCE
        assert s["t_terminal"]["mono_s"] <= s["first_seen"]["mono_s"] + CFG["resource_wait_s"] + 1.0
        assert s["inference_attempts"] == 0


def test_T13_pending_not_starved_by_deferred(run):
    clk = Clock()
    c = core(run, ocr=FakeOcr([]), vlm=FakeVlm(), clock=clk)
    make_job(run, 1)
    for _ in range(3):
        c.step(); clk.t += 1.0
    make_job(run, 2, seed=1)
    c.step()
    k2 = [k for k in c.states if k.endswith("ev002__cap1")][0]
    assert c.states[k2]["ocr"] is not None, "new pending job got its OCR in the very next round"


# ---------------------------------------------------------------------------------------------------------- T14
def test_T14_admission_denials_do_not_rerun_ocr_or_reset_times(run):
    clk, ocr = Clock(), FakeOcr([])
    c = core(run, ocr=ocr, vlm=FakeVlm(), clock=clk)
    make_job(run, 1)
    c.step()
    st = next(iter(c.states.values()))
    t_ocr0, first0, deadline0 = dict(st["ocr"]["t_start"]), dict(st["first_seen"]), st["deadline_mono"]
    for _ in range(40):
        clk.t += 1.0
        c.step()
    assert ocr.calls == 1
    assert st["ocr"]["t_start"] == t_ocr0 and st["first_seen"] == first0 and st["deadline_mono"] == deadline0
    assert st["admission_checks"] >= 2 and st["inference_attempts"] == 0
    hist = [json.loads(l) for l in (run / "jobs" / "history" / f"{st['work_key']}.jsonl").read_text().splitlines()]
    assert sum(1 for h in hist if h["kind"] == "admission_check") == st["admission_checks"]
    assert not any(h["kind"] == "inference_attempt" for h in hist)


# ---------------------------------------------------------------------------------------------------------- T15
def test_T15_ocr_published_immediately_vlm_state_separate_and_consistent(run):
    clk = Clock()
    c = core(run, ocr=FakeOcr([("418", 0.6)]), vlm=FakeVlm(), clock=clk)
    make_job(run, 1)
    c.step()
    d = done_current(run, 1)
    assert d["terminal"] is False and d["ocr"]["raw_text"] == ["418"]
    assert d["vlm"]["status"] == ReadReason.DEFERRED_RESOURCE and d["status"] == "OCR_PUBLISHED_VLM_PENDING"
    rs = json.loads((run / "jobs" / "reader_status.json").read_text())
    st = rs["jobs"][d["work_key"]]
    assert st["vlm"] == d["vlm"]["status"] and st["result_revision"] == d["result_revision"]
    for _ in range(60):
        clk.t += 1.0
        c.step()
    d2 = done_current(run, 1)
    assert d2["terminal"] is True and d2["reason_code"] == ReadReason.SKIPPED_RESOURCE
    assert d2["vlm"]["status"] == ReadReason.SKIPPED_RESOURCE and d2["ocr"]["raw_text"] == ["418"]
    rs = json.loads((run / "jobs" / "reader_status.json").read_text())
    assert rs["jobs"][d2["work_key"]]["reason_code"] == d2["reason_code"]


def test_T15_ocr_first_confident_pantilt_never_calls_vlm(run):
    vlm = FakeVlm(parsed=dict(sign_type="room_number_plate", room_number="999"))
    c = core(run, ocr=FakeOcr([("418", 0.97)]), vlm=vlm, admit=lambda n: (True, "ok"))
    make_job(run, 1, capture_source="IMX477_PANTILT")
    c.step()
    d = done_current(run, 1)
    assert vlm.calls == 0 and d["terminal"] and d["reason_code"] == ReadReason.OCR_ONLY
    assert d["vlm"]["status"] == ReadReason.VLM_NOT_NEEDED and d["semantic_status"] == SemanticStatus.SEMANTIC_ACCEPTED


def test_T15_vlm_timeout_and_conflict_terminal(run):
    c = core(run, ocr=FakeOcr([("418", 0.6)]), vlm=FakeVlm(exc=TimeoutError("read timed out")), admit=lambda n: (True, "ok"))
    make_job(run, 1)
    c.step()
    assert done_current(run, 1)["reason_code"] == ReadReason.VLM_TIMEOUT
    r2 = run.parent / "kci_e5_conf"
    (r2 / "jobs" / "pending").mkdir(parents=True)
    c2 = core(r2, ocr=FakeOcr([("418", 0.6)]), admit=lambda n: (True, "ok"),
              vlm=FakeVlm(parsed=dict(sign_type="room_number_plate", room_number="419")))
    make_job(r2, 1)
    c2.step()
    d = done_current(r2, 1)
    assert d["terminal"] and d["reason_code"] == ReadReason.READING_CONFLICT
    assert d["semantic_status"] == SemanticStatus.SEMANTIC_REJECTED and len(d["labels"]) == 2, "both alternatives kept"
    assert d["vlm"]["inference_attempts"] == 1 and d["vlm"]["admission_checks"] == 1


def test_T15_unprofiled_vlm_resolution_is_refused(run):
    cfg = dict(CFG, vlm_max_side=1280)
    vlm = FakeVlm(parsed=dict(sign_type="room_number_plate", room_number="418"))
    c = core(run, ocr=FakeOcr([]), vlm=vlm, admit=lambda n: (True, "ok"), cfg=cfg)
    make_job(run, 1)
    c.step()
    st = next(iter(c.states.values()))
    assert vlm.calls == 0 and "resource_profile_unverified_1280px" in st["vlm"]["attempts"][-1]["reason"]


# ---------------------------------------------------------------------------------------------------------- T16
def test_T16_ocr_cache_key_includes_input_crop_preprocess_engine(run):
    base = ("a" * 64, [0, 0, 10, 10], None, 0, "pre-1", ENGINE, {})
    k = J.ocr_cache_key(*base)
    assert J.ocr_cache_key(*base) == k
    assert J.ocr_cache_key("b" * 64, *base[1:]) != k
    assert J.ocr_cache_key(base[0], [0, 0, 11, 10], *base[2:]) != k
    assert J.ocr_cache_key(*base[:4], "pre-2", *base[5:]) != k
    assert J.ocr_cache_key(*base[:5], dict(ENGINE, rev="2"), {}) != k
    assert J.ocr_cache_key(*base[:3], 90, *base[4:]) != k


def test_T16_cache_hit_same_input_miss_on_changed_preprocess_or_crop(run):
    o1 = FakeOcr([("418", 0.97)])
    c1 = core(run, ocr=o1)
    make_job(run, 1)
    c1.step()
    assert o1.calls == 1
    o2 = FakeOcr([("418", 0.97)])                     # same image, different capture id -> same cache key -> hit
    c2 = core(run, ocr=o2)
    make_job(run, 1, cap="cap2")
    c2.step()
    assert o2.calls == 0 and c2.counters["ocr_cache_hits"] == 1
    o3 = FakeOcr([("418", 0.97)])
    c3 = core(run, ocr=o3, cfg=dict(CFG, preprocess_version="r1-ocr-pre-2"))
    make_job(run, 1, cap="cap3")
    c3.step()
    assert o3.calls == 1, "changed preprocessing version must not reuse the cache"
    o4 = FakeOcr([("418", 0.97)])
    c4 = core(run, ocr=o4)
    make_job(run, 1, cap="cap4", target_roi=[100, 100, 300, 200], target_roi_verified=True)
    c4.step()
    assert o4.calls == 1 and o4.inputs[0] == (150, 300), "verified ROI + 25% margin is a different input"


def test_T16_no_fixed_center_crop_full_image_when_no_verified_roi(run):
    o = FakeOcr([("418", 0.97)])
    c = core(run, ocr=o)
    make_job(run, 1, target_roi=[100, 100, 300, 200], target_roi_verified=False)
    c.step()
    d = done_current(run, 1)
    assert o.inputs[0] == (720, 1280) and d["ocr"]["input"]["crop"]["kind"] == "full_image"
    assert d["ocr"]["input"]["crop_file"] and Path(d["ocr"]["input"]["crop_file"]).exists()
    assert d["ocr"]["engine"] == ENGINE


# ---------------------------------------------------------------------------------------------------------- T17
def test_T17_duplicates_restart_late_results_reboot(run):
    clk, ocr = Clock(), FakeOcr([("418", 0.97)])
    c = core(run, ocr=ocr, clock=clk)
    job = make_job(run, 1)
    c.step()
    d0 = done_current(run, 1)
    assert d0["terminal"]
    # duplicate JSON re-delivered
    (run / "jobs" / "pending" / "dup.json").write_text(json.dumps(job))
    c.step()
    assert len(c.states) == 1 and ocr.calls == 1
    # reader restart on the same directory
    ocr2 = FakeOcr([("418", 0.97)])
    c2 = core(run, ocr=ocr2, clock=clk)
    (run / "jobs" / "pending" / "dup2.json").write_text(json.dumps(job))
    c2.step()
    assert ocr2.calls == 0 and len(c2.states) == 1
    # late result after terminal: history only
    wk = d0["work_key"]
    assert c2.late_result(wk, dict(parsed=dict(room_number="419"))) is False
    assert done_current(run, 1)["result_revision"] == d0["result_revision"]
    hist = (run / "jobs" / "history" / f"{wk}.jsonl").read_text()
    assert '"late_result"' in hist


def test_T17_reboot_gives_one_bounded_new_deadline(run):
    clk = Clock()
    c = core(run, ocr=FakeOcr([]), vlm=FakeVlm(), clock=clk, boot="BOOT-A")
    make_job(run, 1)
    c.step()
    clk2 = Clock(5.0)                                   # new boot: monotonic restarted
    c2 = core(run, ocr=FakeOcr([]), vlm=FakeVlm(), clock=clk2, boot="BOOT-B")
    st = next(iter(c2.states.values()))
    assert not st["terminal"] and st["reboot_reissues"] == 1 and st["deadline_mono"] == 5.0 + CFG["resource_wait_s"]
    c3 = core(run, ocr=FakeOcr([]), vlm=FakeVlm(), clock=Clock(1.0), boot="BOOT-C")
    st3 = next(iter(c3.states.values()))
    assert st3["terminal"] and st3["reason_code"] == ReadReason.SKIPPED_RESOURCE


def test_T17_cancel_is_terminal_and_input_invalid_without_identity(run):
    c = core(run, ocr=FakeOcr([]), vlm=FakeVlm())
    make_job(run, 1)
    c.step()
    wk = next(iter(c.states))
    assert c.cancel(wk)["reason_code"] == ReadReason.CANCELLED
    (run / "jobs" / "pending" / "legacy.json").write_text(json.dumps(dict(ev=7, image="x.jpg", created=1.0)))
    c.step()
    bad = [s for s in c.states.values() if s["work_key"].startswith("INVALID")]
    assert bad and bad[0]["reason_code"] == ReadReason.INPUT_INVALID


# ---------------------------------------------------------------------------------------------------------- T18
def test_T18_no_text_vs_parse_rejected_vs_legacy(run):
    c = core(run, ocr=FakeOcr([]), vlm=None)
    make_job(run, 1)
    c.step()
    assert done_current(run, 1)["reason_code"] == ReadReason.NO_TEXT_DETECTED
    r2 = run.parent / "kci_e5_parse"
    (r2 / "jobs" / "pending").mkdir(parents=True)
    c2 = core(r2, ocr=FakeOcr([("세미나 모집 안내", 0.9), ("2026.10.15", 0.9)]), vlm=None)
    make_job(r2, 1)
    c2.step()
    assert done_current(r2, 1)["reason_code"] == ReadReason.PARSE_REJECTED
    r3 = run.parent / "kci_e5_plain"
    (r3 / "jobs" / "pending").mkdir(parents=True)
    c3 = core(r3, ocr=FakeOcr([("화장실", 0.9)]), vlm=None)
    make_job(r3, 1)
    c3.step()
    assert done_current(r3, 1)["reason_code"] == ReadReason.PARSE_REJECTED, "plain non-room text is a parser reject, not no-text"
    produced = {s["reason_code"] for s in list(c.states.values()) + list(c2.states.values())}
    assert ReadReason.LEGACY_NO_ROOM_LABEL not in produced
    assert J.legacy_reason(dict(status="NO_ROOM_LABEL")) == "LEGACY_NO_ROOM_LABEL_AMBIGUOUS"


# ---------------------------------------------------------------------------------------------------------- T10 / T12
def test_T10_quality_fail_is_terminal_no_ocr_success_no_e6(run):
    ocr, calls = FakeOcr([("418", 0.99)]), []
    c = core(run, ocr=ocr, localize=lambda st, rec, packet_only=False: calls.append(packet_only) or {})
    make_job(run, 1, capture_status=CaptureStatus.CAPTURE_QUALITY_FAIL)
    c.step()
    d = done_current(run, 1)
    assert ocr.calls == 0 and d["reason_code"] == ReadReason.CAPTURE_QUALITY_FAIL
    assert d["semantic_status"] == SemanticStatus.SEMANTIC_REJECTED and d["e6"]["status"] == "SKIPPED"
    assert calls == [True], "packet-only localisation, never E6"


def test_T12_proxy_capture_never_reaches_e6_or_pantilt_success(run):
    calls = []
    c = core(run, ocr=FakeOcr([("418", 0.99)]), localize=lambda st, rec, packet_only=False: calls.append(packet_only) or {})
    make_job(run, 1, capture_status=CaptureStatus.CAPTURE_PROXY, pantilt_ok=False, capture_source="REALSENSE_PROXY")
    c.step()
    d = done_current(run, 1)
    assert d["pantilt_ok"] is False and d["counted_as_pantilt_success"] is False
    assert d["e6"]["status"] == "SKIPPED" and "proxy" in d["e6"]["why"] and calls == [True]
    assert d["e7"]["status"] == "NOT_WIRED"


def test_accepted_quality_ok_pantilt_capture_allows_e6(run):
    calls = []
    c = core(run, ocr=FakeOcr([("418", 0.99)]), localize=lambda st, rec, packet_only=False: calls.append(packet_only) or {})
    make_job(run, 1)
    c.step()
    d = done_current(run, 1)
    assert d["e6"]["status"] == "PENDING" and calls == [False] and d["counted_as_pantilt_success"] is True


# ---------------------------------------------------------------------------------------------------------- wiring
def test_wrapper_reader_onboard_has_no_vlm_force_and_uses_core():
    import reader as R
    import reader_onboard as RO
    assert getattr(R, "R1_READER", False)
    assert "want_vlm" not in RO.OurReader.__dict__, "pan/tilt always-VLM override removed"
    assert Path(R.__file__).resolve().is_relative_to(WS) if hasattr(Path, "is_relative_to") else str(WS) in str(Path(R.__file__).resolve())
    assert str(WS) in str(Path(RO.__file__).resolve()) and str(WS) in str(Path(J.__file__).resolve())
    assert CFG["vlm_need_mb"] == 4200 and CFG["vlm_reserve_mb"] == 2000, "protection values unchanged"


def test_packet_identity_status_and_clock_domains(tmp_path):
    from e5_e7 import e5_packet
    trial = tmp_path / "trial"
    (trial / "runner").mkdir(parents=True)
    (trial / "runner" / "events.jsonl").write_text(json.dumps(dict(t=557.2, stamp=1.76e9, picks=[[300, 100, 340, 130]], z=[3.0],
                                                                   pose=[0, 0, 0], xyz=[3, 1, 1.5])) + "\n")
    rec = dict(labels=[], reason_code=ReadReason.NO_TEXT_DETECTED, semantic_status=SemanticStatus.SEMANTIC_REJECTED,
               terminal=True, first_seen=dict(wall=1.7e9), published=dict(wall=1.7e9 + 5), ocr=dict(engine=ENGINE, raw_text=[]),
               vlm=dict(parsed=None))
    job = dict(schema_version="r1.1", run_id="r", event_key="r#ev001", capture_id="c1", capture_status="CAPTURE_PROXY",
               pantilt_ok=False)
    pk = e5_packet.build_packet(trial, Path(os.environ["SCI_PANTILT_CALIB"]), reading=rec, identity=job)
    assert pk["schema"] == e5_packet.R1_SCHEMA and pk["identity"]["capture_id"] == "c1"
    assert pk["status"]["proxy"] is True and pk["status"]["e7"] == "NOT_WIRED"
    assert "scene_rel_s" in pk["timestamps_clock_domain"]["stop_t"] and "wall" in pk["timestamps_clock_domain"]["reading_request"]
    assert pk["odom"]["frame"] == "UNKNOWN" and pk["odom"]["frame_null_reason"]
    assert pk["failure_reason"] == ReadReason.NO_TEXT_DETECTED
