#!/usr/bin/env python3
"""check_use_on_run.py — 지난 실험의 촬영 기록으로 "사진 -> 운영판 판독·위치추정" 줄을 만들어 보고 두 가지를 잰다 (로봇에서, 읽기 전용).
  (1) 운영판 OCR(e6_ocr.sign_read, 현장과 같은 함수)이 그 사진에서 무엇을 읽는가
  (2) 줄에 적히는 표지 위치(target_odom_link_xy)를 한 후보의 변환으로 그 층 지도에 올렸을 때, 기대한 번호의 방에서 몇 m 떨어지는가
      (운영판 쌍 판정은 3 m 안이어야 그 방으로 본다: e6_semantic.PLACE_TOL_M)
운영 폴더는 읽기만 하고, 아무 줄도 쓰지 않는다. 실험이 도는 중이면 시작하지 않는다.
   python3 check_use_on_run.py <실험ID> <후보ID 예: H027_F4> [기대 번호 …(표지 순서대로)] [--no-ocr]"""
import bisect, hashlib, json, math, sys, time
from pathlib import Path

S = Path.home() / "sci_ws_student_staging_20261109"; O = Path.home() / "sci_ws/repairs/livefield2_20261018_151606"
st = json.loads((O / "runtime/ctl/experiment.json").read_text()).get("state")
if st in ("RUNNING", "PARTIAL", "PAUSED", "STARTING", "STOPPING"):
    sys.exit("실험 %s: 시작하지 않음" % st)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
X, HID, WANT = args[0], args[1], args[2:]
sys.path[:0] = [str(S / "candidate/live2"), str(S / "tools/plate_site")]
import cv2
import numpy as np
import e6_core as G
import e6_semantic as SEM
import plate_site as PS


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


ev = jl(S / "_runs/plate_site" / X / "events.jsonl")
first = {}
for e in ev:                                               # 표지마다 기억 위치만으로 조준한 첫 사진
    if e.get("kind") == "shot" and e.get("tag") == "aim" and (e.get("target") or {}).get("id") not in first:
        first[e["target"]["id"]] = e
h = next(q for q in json.loads((O / "experiments" / X / "state/e3_state.json").read_text())["hypotheses"] if q["id"] == HID)
tr = jl(O / "experiments" / X / "archive/field_eval/odom_trace.jsonl"); ks = [r["stamp"] for r in tr]
i = min(len(tr) - 1, max(0, bisect.bisect_left(ks, h["global_pose"]["odom_t"])))
rec_h = dict(pos=h["pose_px"], anchor=[-tr[i]["x"], tr[i]["y"]], phi=h["phi"], ppm=h["ppm"])
sf = SEM.SemanticFloors(); spaces = [s for s in sf.maps[h["floor"]]["spaces"] if s.get("polygon_px")]
print("%s · 후보 %s (%d층, %.1f px/m) · 방 허용 거리 %.1f m · 표지 %d개" % (X, HID, h["floor"], h["ppm"], SEM.PLACE_TOL_M, len(first)))
ok_all = True
for k, (tid, e) in enumerate(sorted(first.items(), key=lambda kv: kv[1]["wall"])):
    f = S / "_runs/plate_site" / X / "photos" / e["photo"]; data = f.read_bytes(); img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    x, y, yaw = G.planar_to_raw(*e["pose"])
    ph = dict(name=e["photo"], wall=e["wall"], stopped_wall=e["wall"] - float(e.get("sec") or 0), stop=dict(t=e["wall"], x=x, y=y, yaw=yaw), sign=e.get("sign"),
              pan=(e.get("pt_cmd") or [None, None])[0], tilt=(e.get("pt_cmd") or [None, None])[1], basis=e.get("basis"), capture_id=(e.get("capture") or {}).get("capture_id"))
    tg = dict(e["target"]); tg.setdefault("n_refine", 0)
    r = PS.make_record("plate_%s_P%03d" % (X, k + 1), X, ph, tg, str(f), hashlib.sha256(data).hexdigest(), [int(img.shape[1]), int(img.shape[0])])
    stored = r["status"] in ("CAPTURE_STORED", "CAPTURED") and bool(r["image_quality"]["ok"]) and r.get("home_confirmed") is not False    # e8_pair.on_precision 의 조건
    p = np.array(G.map_projection(rec_h, *r["target_odom_link_xy"])); ps = np.array(G.map_projection(rec_h, -x, y))
    near = min(((SEM._poly_dist_px(s["polygon_px"], p) / h["ppm"], s) for s in spaces), key=lambda q: q[0])
    want = WANT[k] if k < len(WANT) else None
    dw = None
    if want:
        cand = [s for s in spaces if str(s.get("id", "")) == "room_" + want or str(s.get("name") or "").split()[:1] == [want]]
        dw = min((SEM._poly_dist_px(s["polygon_px"], p) / h["ppm"] for s in cand), default=None)
    line = "  표지 %s · %s · 줄 받아들임 %s · 품질 표시 %s · 표지↔촬영 자리 %.1f m · 가장 가까운 방 %s (%.1f m)" % (
        tid, e["photo"], "예" if stored else "아니오", r["image_quality"]["flags"] or "없음", float(np.linalg.norm(p - ps)) / h["ppm"], near[1].get("name") or near[1].get("id"), near[0])
    if want:
        good = dw is not None and dw <= SEM.PLACE_TOL_M; ok_all &= good
        line += " · 기대 %s호까지 %s → %s" % (want, "방 없음" if dw is None else "%.1f m" % dw, "허용 안" if good else "허용 밖")
    print(line, flush=True)
    if "--no-ocr" not in sys.argv:
        import e6_ocr
        t0 = time.time(); lines, meta = e6_ocr.sign_read(img)
        txt = [(str(l.get("text") if isinstance(l, dict) else l[0]), round(float((l.get("conf") if isinstance(l, dict) else l[1]) or 0), 2)) for l in (lines or [])]
        hit = want is not None and any(want in t.replace(" ", "") for t, _ in txt); ok_all &= (hit or want is None)
        print("      OCR %.0f초 (%s): %s%s" % (time.time() - t0, meta.get("roi_source"), txt[:5], "" if want is None else (" → 번호 읽음" if hit else " → 번호 못 읽음")), flush=True)
print("기대한 방·번호와 모두 맞음" if ok_all else "맞지 않는 것이 있음")
