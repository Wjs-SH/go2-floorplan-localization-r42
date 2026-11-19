"""e6_ocr.py — RapidOCR (PP-OCRv4, CPU) for E6 captures, independent of which python runs the caller.

The OCR packages live in the persistent student venv ~/sci_ws/.venv (rapidocr_onnxruntime 1.3.24, onnxruntime 1.16.3,
CPU provider) — not in system python3 and not under /tmp. In-process when importable, otherwise one subprocess in
that venv (image passed as a PNG under the LIVE2 runtime/tmp, result as JSON on stdout). Nothing is installed here.

read(img_bgr) -> (lines [[text, conf]], meta{engine, python, ocr_s})
read_many([img, ...]) -> ([lines, ...], meta{engine, load_s, per_image_s, ocr_s}) : ONE model load for all images (2026-11-01:
                                      the field read of X20261031_231940 took 67 / 67 / 118.6 s because sign_read started two
                                      subprocesses — two RapidOCR model loads — per photo, against a 120 s timeout each)
sign_read(img_bgr, roi=None) -> (lines, meta) : OCR on the sign ROI (operator-selected ROI when given, else
                                      e6_focus_v2.sign_roi; full resolution) + the full frame, in ONE read_many call,
                                      merged (best confidence per text); meta carries the student-formula sharpness on the
                                      sign ROI (gate) and on the centre ROI (reference).
"""
import json
import os
import pwd
import subprocess
import sys
import time
from pathlib import Path

HOME = Path(pwd.getpwuid(os.getuid()).pw_dir)
OCR_PY = HOME / "sci_ws" / ".venv" / "bin" / "python"
TMP = Path(__file__).resolve().parents[2] / "runtime" / "tmp"
_ENGINE = {}

_SUB = r"""
import json, sys, time, cv2
from rapidocr_onnxruntime import RapidOCR
img = cv2.imread(sys.argv[1]); t = time.time(); res, _ = RapidOCR()(img)
print(json.dumps(dict(lines=[[r[1], round(float(r[2]), 3)] for r in (res or [])], ocr_s=round(time.time() - t, 2))))
"""
# same engine, same model, same defaults as _SUB (RapidOCR() on the decoded BGR image) — the model is loaded once for all images,
# and the onnxruntime thread count is set explicitly (argv[1]) so that its threads stay on the caller's cores (see ocr_threads)
_SUB_MANY = r"""
import json, sys, time, cv2
t0 = time.time()
from rapidocr_onnxruntime import RapidOCR
n = int(sys.argv.pop(1))
eng = RapidOCR(intra_op_num_threads=n, inter_op_num_threads=1) if n > 0 else RapidOCR(); load_s = time.time() - t0
out = []
for f in sys.argv[1:]:
    img = cv2.imread(f); t = time.time(); res, _ = eng(img)
    out.append(dict(lines=[[r[1], round(float(r[2]), 3)] for r in (res or [])], s=round(time.time() - t, 2)))
print(json.dumps(dict(results=out, load_s=round(load_s, 2))))
"""


def env_check():
    """Where OCR can run -> dict (no model load in-process)."""
    out = dict(ocr_python=str(OCR_PY), ocr_python_exists=OCR_PY.exists(), in_process=False)
    try:
        import rapidocr_onnxruntime  # noqa: F401
        out["in_process"] = True
    except ImportError:
        pass
    if OCR_PY.exists():
        r = subprocess.run([str(OCR_PY), "-c", "import rapidocr_onnxruntime as r, onnxruntime as o; "
                            "print(r.__file__, o.__version__, o.get_available_providers())"], capture_output=True, text=True, timeout=60)
        out.update(venv_ok=r.returncode == 0, venv_detail=(r.stdout or r.stderr).strip()[-300:])
    return out


def read(img_bgr):
    t = time.time()
    try:
        from rapidocr_onnxruntime import RapidOCR
        if "rapid" not in _ENGINE:
            _ENGINE["rapid"] = RapidOCR()
        res, _ = _ENGINE["rapid"](img_bgr)
        return [[r[1], round(float(r[2]), 3)] for r in (res or [])], dict(engine="rapidocr in-process", python=sys.executable,
                                                                          ocr_s=round(time.time() - t, 2))
    except ImportError:
        pass
    import cv2
    TMP.mkdir(parents=True, exist_ok=True)
    f = TMP / f"e6_ocr_{os.getpid()}_{int(t * 1000)}.png"
    cv2.imwrite(str(f), img_bgr)
    try:
        r = subprocess.run([str(OCR_PY), "-c", _SUB, str(f)], capture_output=True, text=True, timeout=120)
    finally:
        f.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"OCR subprocess failed ({OCR_PY}): {r.stderr.strip()[-300:]}")
    d = json.loads(r.stdout.strip().splitlines()[-1])
    return d["lines"], dict(engine="rapidocr subprocess", python=str(OCR_PY), ocr_s=round(time.time() - t, 2))


MAX_THREADS = 4


def ocr_threads():
    """onnxruntime threads = the cores this process may use (1..MAX_THREADS).
    2026-11-01 (measured): with the default (-1) onnxruntime 1.16.3 starts one intra-op thread per core and PINS each thread to
    its own core, ignoring the inherited taskset — e34 runs on core 5 only, but its OCR ran threads on cores 0-3, 6, 7 (SigLIP,
    hub, E2 at 100 %), and every parallel step waited for the starved ones (field read 67 / 67 / 118.6 s vs 5 s idle). With an
    explicit count the threads keep the inherited affinity (checked: all threads Cpus_allowed 5)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    return max(1, min(MAX_THREADS, n))


def read_many(imgs, timeout=120):
    """Several images, ONE OCR model load (in-process engine or one subprocess). -> ([lines per image], meta)."""
    t = time.time()
    nth = ocr_threads()
    try:
        from rapidocr_onnxruntime import RapidOCR
        new = "rapid" not in _ENGINE
        if new:
            _ENGINE["rapid"] = RapidOCR(intra_op_num_threads=nth, inter_op_num_threads=1)
        load_s = round(time.time() - t, 2) if new else 0.0
        out, per = [], []
        for im in imgs:
            t1 = time.time()
            res, _ = _ENGINE["rapid"](im)
            out.append([[r[1], round(float(r[2]), 3)] for r in (res or [])])
            per.append(round(time.time() - t1, 2))
        return out, dict(engine="rapidocr in-process", python=sys.executable, load_s=load_s, per_image_s=per, ocr_s=round(time.time() - t, 2),
                         threads=nth)
    except ImportError:
        pass
    import cv2
    TMP.mkdir(parents=True, exist_ok=True)
    fs = [TMP / f"e6_ocr_{os.getpid()}_{int(t * 1000)}_{i}.png" for i in range(len(imgs))]
    for f, im in zip(fs, imgs):
        cv2.imwrite(str(f), im)
    try:
        r = subprocess.run([str(OCR_PY), "-c", _SUB_MANY, str(nth)] + [str(f) for f in fs], capture_output=True, text=True, timeout=timeout)
    finally:
        for f in fs:
            f.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"OCR subprocess failed ({OCR_PY}): {r.stderr.strip()[-300:]}")
    d = json.loads(r.stdout.strip().splitlines()[-1])
    return [x["lines"] for x in d["results"]], dict(engine="rapidocr subprocess (one model load)", python=str(OCR_PY), load_s=d["load_s"],
                                                     per_image_s=[x["s"] for x in d["results"]], ocr_s=round(time.time() - t, 2),
                                                     threads=nth)


ROI_RETRY_CONF, ROI_EXPAND = 0.85, 0.35


def _useful_text(t):
    """A room number or a facility word: what the localization can use."""
    s = "".join(str(t).split()).upper()
    return (s[:3].isdigit() and len(s) >= 3) or s in ("TOILET", "TOILETS", "RESTROOM", "WC", "화장실", "남자화장실", "여자화장실")


def _expand(x0, y0, x1, y1, w, h):
    dx, dy = int((x1 - x0) * ROI_EXPAND), int((y1 - y0) * ROI_EXPAND)
    return max(0, x0 - dx), max(0, y0 - dy), min(w, x1 + dx), min(h, y1 + dy)


def sign_read(img_bgr, roi=None):
    """roi: operator-selected sign ROI in 1280x720 coordinates (e6_manual), used instead of the automatic one."""
    import e6_focus_v2 as F2
    h, w = img_bgr.shape[:2]
    if roi is not None:
        roi, src = [int(v) for v in roi], "OPERATOR_SELECTED"
    else:
        roi, src = F2.sign_roi(img_bgr)
    sharp, _ = F2.roi_metric(img_bgr, roi)
    centre, _ = F2.roi_metric(img_bgr, (320, 180, 960, 540))
    sx, sy = w / 1280.0, h / 720.0
    x0, y0, x1, y1 = int(roi[0] * sx), int(roi[1] * sy), int(roi[2] * sx), int(roi[3] * sy)
    # timeout 240 s = the bound of the two former 120 s subprocesses together (field O005 took 118.6 s for both)
    (l_roi, l_full), m1 = read_many([img_bgr[y0:y1, x0:x1], img_bgr], timeout=240)
    best = {}
    for txt, c in l_roi + l_full:
        best[txt] = max(c, best.get(txt, 0.0))
    # R34 (11-01 O002: the automatic ROI cut plate 405 at its right edge -> 'B' 0.52). No confident room / facility text -> ONE read of a wider ROI, as a real reading of real pixels.
    # Nothing is invented: if the wider ROI reads nothing either, the observation stays an honest failure.
    exp_meta = dict(roi_expanded=None, ocr_expanded=None, roi_cut_warning=False)
    if not any(c >= ROI_RETRY_CONF and _useful_text(t) for t, c in best.items()):
        ex0, ey0, ex1, ey1 = _expand(x0, y0, x1, y1, w, h)
        if (ex0, ey0, ex1, ey1) != (0, 0, w, h):
            (l_exp,), m2 = read_many([img_bgr[ey0:ey1, ex0:ex1]], timeout=120)
            for txt, c in l_exp:
                best[txt] = max(c, best.get(txt, 0.0))
            exp_meta = dict(roi_expanded=[ex0, ey0, ex1, ey1], ocr_expanded=l_exp, roi_expanded_s=m2.get("ocr_s"),
                            roi_cut_warning=any(c >= ROI_RETRY_CONF and _useful_text(t) for t, c in l_exp))
    lines = [[k, v] for k, v in sorted(best.items(), key=lambda kv: -kv[1])]
    return lines, dict(exp_meta, sign_roi_1280=list(roi), sign_roi_px=[x0, y0, x1, y1], roi_source=src, roi_laplacian_var=round(sharp, 1),
                       centre_laplacian_var=round(centre, 1), ocr_roi=l_roi, ocr_full=l_full, engine=m1["engine"],
                       load_s=m1.get("load_s"), roi_s=m1["per_image_s"][0], full_s=m1["per_image_s"][1], ocr_s=m1["ocr_s"],
                       threads=m1.get("threads"))
