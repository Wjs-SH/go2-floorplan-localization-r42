#!/usr/bin/env python3
"""read_shots.py — 명판 재촬영 사이트로 찍은 사진을 로봇의 실제 판독 경로로 읽혀 본다 (스테이징, 2026-11-13).
  OCR  : e6_ocr.sign_read (표지 영역을 찾아 RapidOCR, 현장과 같은 함수)
  VLM  : Qwen3-VL-2B GGUF (llama-server, 자체 포트 8098) — 현장과 같은 질문(vlm.PROMPT)과 같은 자르기(e8_pair._crop_for_vlm, 표지 영역 + 여백)
  고르기: plate_choice.choose — 번호 끝 두 자리가 같은 지도 방 이름 중에서 고르게 한다(두 번 물어 일치할 때만)
어두운 사진은 밝게 보정한 것도 따로 읽혀 본다(현장 경로에는 없는 처리 — 조명이 있었다면 어땠을지 보는 용도).
운영 폴더는 읽기만 한다. 실험이 도는 중이면 시작하지 않는다.
   python3 read_shots.py <OUT 폴더> <실험ID:사진파일:기대번호> ..."""
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("SCI_VLM_PORT", "8098")
S = Path.home() / "sci_ws_student_staging_20261109"; O = Path.home() / "sci_ws/repairs/livefield2_20261018_151606"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
st = json.loads((O / "runtime/ctl/experiment.json").read_text()).get("state")
if st in ("RUNNING", "PARTIAL", "PAUSED", "STARTING", "STOPPING"):
    sys.exit(f"experiment {st}: not started")
sys.path[:0] = [str(S / "candidate/live2"), str(S / "candidate/src/sci_text_reading")]
import cv2
import numpy as np
import vlm as V
import e8_pair as EP
import e6_ocr
import plate_choice as PC


def bright(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB); l = lab[:, :, 0].astype(np.float32)
    l = np.clip(l * (235.0 / max(float(np.percentile(l, 99.5)), 8.0)), 0, 255).astype(np.uint8)
    lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    return cv2.fastNlMeansDenoisingColored(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), None, 6, 6, 5, 15)


def read_one(srv, img, want):
    t0 = time.time()
    lines, meta = e6_ocr.sign_read(img)
    ocr = [(str(l.get("text") if isinstance(l, dict) else l[0]), round(float((l.get("conf") if isinstance(l, dict) else l[1]) or 0), 2)) for l in (lines or [])]
    t_ocr = time.time() - t0
    crop = EP._crop_for_vlm(img, meta.get("sign_roi_1280") if meta.get("roi_source") == "SIGN_COMPONENT" else None)
    a = srv.ask(crop)
    txt = " ".join(t for t, _ in ocr)
    num = want if want in txt.replace(" ", "") else (PC.plate_number(dict(ocr_lines=lines, vlm=dict(raw=a.get("raw")))) if hasattr(PC, "plate_number") else None)
    ch = None
    if num:
        secs = []
        def ask(c, p):
            r = srv.ask(c, p, max_tokens=8); secs.append(round(r["sec"], 1)); return r["raw"]
        try:
            r = PC.choose(ask, crop, str(num))
            ch = dict(num=str(num), state=r.get("state"), floor=r.get("floor"), picks=r.get("picks"), raw=r.get("raw"), sec=secs)
        except Exception as ex:                                  # noqa: BLE001
            ch = dict(num=str(num), error=f"{type(ex).__name__}: {ex}")
    return dict(ocr=ocr, ocr_roi=meta.get("roi_source"), ocr_sharp=meta.get("roi_laplacian_var"), ocr_s=round(t_ocr, 1), vlm_raw=a.get("raw"), vlm_s=round(a.get("sec", 0), 1),
                crop=list(crop.shape[:2]), choice=ch, mean=round(float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))), 1))


srv = V.VlmServer(log=str(OUT / "llama_server.log")); srv.start(); print("server up", flush=True)
res = []
try:
    for spec in sys.argv[2:]:
        X, name, want = spec.split(":")
        f = S / "_runs/plate_site" / X / "photos" / name
        img = cv2.imread(str(f))
        if img is None:
            print(spec, "사진 없음", flush=True); continue
        row = dict(run=X, photo=name, want=want, orig=read_one(srv, img, want))
        if row["orig"]["mean"] < 60:
            row["bright"] = read_one(srv, bright(img), want)
        res.append(row)
        for k in ("orig", "bright"):
            if k in row:
                r = row[k]
                print(f"{X[-6:]} {name} [{'원본' if k == 'orig' else '밝게'} 밝기 {r['mean']}] OCR({r['ocr_roi']}, {r['ocr_s']}s): {r['ocr'][:6]} | VLM({r['vlm_s']}s): {str(r['vlm_raw'])[:170]} | 고르기: {r['choice']}", flush=True)
        (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
finally:
    srv.stop(); print("server stopped", flush=True)
