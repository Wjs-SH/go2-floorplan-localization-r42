#!/usr/bin/env python3
"""fused_floor.py (2026-11-16, 사용자 지시 "지금 거 운영으로 올려서 메인 알고리즘으로 확정") — 층 확정 = 기하 + 명판 종합 점수.

설계 근거: 인수인계 원문 1.4절 9단계 "의미지도와 LiDAR·Odometry 를 융합하여 위치 후보를 갱신한다". 지금까지의 E7 은 다른 층과의 차를
LiDAR 정합 점수(기하)로만 쟀다(0.05 m). 이 건물처럼 층 구조가 같으면 기하 차가 0.00~0.03 m 라 명판을 읽어도 층이 확정되지 않았다.

언제 도는가: e6e7_link._run 이 E8 쌍 판정(REGION_CANDIDATE_READY, 서로 다른 명판 2 장 이상)으로 E7 구간 정합을 마친 직후. E7 결과(후보별 정합 점수·
보정·수렴)와 그 쌍의 명판(번호·저장 위치)으로 판정한다. E7 후보 수·임계값·계산은 바꾸지 않는다(추가 계산은 후보별 명판 거리와 구간 경로 벽 통과뿐).
종합 점수(낮을수록 좋음, nat):
    C(후보) = (정합 점수 − 이 E7 의 최소 정합 점수) / SIG_G  +  Σ_명판 c_i
    c_i = min(d_i² / (2 SIG_P²), ln(1/EPS))   그 층 의미지도에 그 번호 방이 있고, 보정 뒤 후보 변환으로 올린 명판 저장 위치가 방에서 d_i ≤ TOL
        = ln(1/EPS)            그 층에 그 번호 방이 없거나(방 번호 첫 자리 = 층) TOL 보다 멀다(오독이거나 다른 자리)
  긴 경로 벽 뚫기(2026-11-17, 사용자 지시 "둘 다 넣는 게 맞는 듯"): 90 초 구간 검사에 더해, 최근 PASS_PATH_M(30 m, E3 의 WIN_PATH_M 과 같음)의
     이동 경로를 같은 변환으로 올려 벽을 뚫고 반대편으로 넘어가면(e3_tracker.wall_pass_any — E3 운영 기준, esy v2) 뺀다.
     11-17 X20261117_114930: 확정 후보 H067_F6 은 90 초 구간 벽 0 m 였지만 약 19° 돈 변환이라 출발점이 13.4 m 어긋났다 → 이 검사로 빠진다.
     출발부터 전체 경로에 적용하면 누적 오차로 위치가 거의 맞는 변환도 걸려(213541 H081, 출발점 2.1 m: 맞는 8F 확정을 잃음) 30 m 로 둔다.
     전체 경로에서 벽을 지난 길이(full_path_cross_m)는 기록만 한다.
  층 안 위치(2026-11-17, 같은 지시): 층 점수·층 확정은 그대로. 확정 위치로 쓸 후보는 이긴 층의 쓸 수 있는 후보 중 명판이 모두 TOL 안인 것 가운데
     정합 점수(기하)가 가장 좋은 것(없으면 종합 점수 1위). 같은 층 안에서는 명판 항이 기하를 눌러 기하가 나쁜 후보가 뽑혔다(위 H067: 명판 0.0/0.11 m, 정합 0.043 m vs 0.022 m).
  상한(2026-11-17, 사용자 지시 "맞는 번호를 읽었는데 맞는 층에 불리한 근거가 되지 않게"): 상한이 없으면 d > √(2 ln 20)·SIG_P = 2.45 m 에서
     맞는 층 벌점이 '방 없음' 벌점보다 커져 맞는 번호 명판이 맞는 층에 불리했고, TOL 에서 벌점이 4.5 → 3.0 으로 뚝 떨어졌다.
     상한을 두면 멀리 저장된 명판은 근거가 0(모든 층과 같은 벌점)이 될 뿐 맞는 층을 깎지 않는다. 명판 2 장이고 기하 차가 0.08 m 보다 작으면 확정 여부는 바뀌지 않는다(바뀌는 것은 명판 3 장 이상이거나 기하 차가 큰 경우).
  벽 통과(경로 기준, 사용자 설계): 후보 변환 + 보정으로 올린 정합 구간 경로(프레임 위치)가 벽(wall_tr)을 지난 길이 > XMAX 인 후보는 뺀다.
     (E7 안의 LiDAR 빛줄기 기준 거부권·명판 규칙의 절대값 0.35 는 이 판정에 쓰지 않는다.)
  층 점수 = 그 층에서 정합이 수렴하고(탐색 경계 아님) 경로 벽 통과가 없는 후보의 최소 C. 그런 후보가 없는 층(E7 이 정합하지 않았거나 수렴하지 않은 층)은
     기하가 최선(기하 항 0)이고 그 층에 같은 번호 방이 있으면 그 명판 항도 0 이라고 보수적으로 둔다(다른 층에 유리한 쪽).
  층 확정 = 2위 층 점수 − 1위 ≥ LAM(ln 100) + 1위 층에 실제 후보가 있음 + FAST-LIO 저하 없음 + 같은 주행거리 세션.
  확정하면 e7_state 의 global_decision = GLOBAL_CONFIRMED, path = FUSED_FLOOR, semantic_hypothesis = 1위 후보(위치·방향은 그 후보 + E7 보정).
매개변수(현장 결과를 보기 전에 정함, 정답 미사용): SIG_G 0.05 m(E7 GAP_M), SIG_P 1.0 m(같은 명판 두 번 잰 차 0.87 m), EPS 0.05(번호 오독), LAM ln 100, TOL 3 m(쌍 판정 허용 거리), XMAX 0.3 m(E2·E3 경로 벽 통과).
노트북 검증(같은 식, 다시 만든 지도, 모든 층 정합): 11-16 현장 21 건 중 10 건 확정·전부 정답 층, 11-12~15 번호 읽은 5 건 중 3 건 확정·전부 정답, 틀린 확정 0
  (~/sci_r42_work/wallcross_path_20261116/data/rejudge_X20261116_summary.csv, fused_decisions.csv).
명판 한 장(2026-11-18 B3, 사용자 결정 "오독으로 판단하고 빼는 알고리즘이 들어가면 넣을 수 있다"): e6e7_link 가 확실한 방 번호 한 장만 읽은 상태에서
  사전 검사(값싼 정합 비용)로 기하가 층을 가를 수 있을 때만 이 E7 을 부른다(r['single_plate']). 식·기준은 그대로(같은 ln 100) — 명판 한 장의 최대 근거는 ln 20 이므로
  나머지 1.6 nat 를 기하가 채워야 한다: 명판 층 후보의 정합이 '모든' 다른 층보다 (ln100 − ln20)·SIG_G = 0.08 m 이상 좋아야 확정된다(정합 안 된 층은 기하 최선으로 봄).
  오독 의심: 다른 층의 기하가 명판 층보다 ln20 nat(= 명판 한 장의 최대 무게, 0.15 m) 이상 좋다 — 기하가 명판을 이긴다. 의심이 난 판독은 이 E7 에서 확정하지 않고,
  서로 다른 구간 두 번 의심이면 e6e7_link 가 그 판독을 오독으로 빼낸다(순위·쌍 판정에서 제외, 기록은 남김). 닮은 층(기하 차 < 0.08 m)은 확정 보류일 뿐 오독 근거가 아니다
  (11-16 전 층 정합 34 번: 맞는 명판에서 닮은 층 중앙 6 개 — 오독 근거로 쓰면 맞는 명판이 버려진다).
끄기: 환경변수 LIVE2_FUSED_FLOOR=0 (다음 실험부터). 기록: <state>/e7_fused_floor.jsonl (E7 한 번마다 한 줄, 확정 여부와 무관).
정답 좌표는 입력이 아니다."""
import json
import math
import os
import time

import numpy as np

SIG_G, SIG_P, EPS, LAM, TOL, XMAX = 0.05, 1.0, 0.05, math.log(100.0), 3.0, 0.3
PASS_PATH_M, JUMP_M = 30.0, 3.0   # 벽 뚫기 검사 경로 길이(E3 WIN_PATH_M), 이 거리보다 크게 튄 주행 기록 사이는 잇지 않음 (2026-11-17)
MISS = math.log(1.0 / EPS)          # '방 없음' 벌점 = 맞는 층 명판 벌점의 상한 (2026-11-17)


def enabled():
    return os.environ.get("LIVE2_FUSED_FLOOR", "1") == "1"


def _A(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [-s, -c]])


def placer(rec, fr, d):
    """e7_rematch.project 와 같은 변환(가장 최신 프레임에서 다시 놓고 보정 d 적용). -> (저장소 좌표 -> 지도 화소 함수, ppm)"""
    xn, yn = fr[-1][1], fr[-1][2]
    phi0 = math.radians(rec["phi"])
    pn = np.asarray(rec["pos"], float) + rec["ppm"] * (_A(phi0) @ (np.array([xn, yn]) - np.asarray(rec["anchor"], float)))
    phi1, ppm1 = phi0 + d[2], rec["ppm"] * d[3]
    base = pn + np.array([d[0], -d[1]]) * rec["ppm"]
    A1 = _A(phi1)
    return (lambda q: base + ppm1 * (A1 @ (np.asarray(q, float) - np.array([xn, yn])))), ppm1


def plate_readings(link, r):
    """이 E7 을 부른 쌍의 명판: [(observation_id, 번호, 신뢰도, 저장소 좌표 위치)]. E8 마지막 판정의 ROOM 판독(usable)만."""
    sp = r.get("single_plate")
    if sp:                                                             # B3: 명판 한 장 — e6e7_link 가 판독·저장 위치를 넘긴다
        return [(sp.get("observation_id"), str(sp.get("label")), float(sp.get("conf") or 0.0), np.asarray(sp.get("target"), float))] if sp.get("target") else []
    pair = getattr(link, "pair", None)
    if pair is None:
        return []
    ev = getattr(pair, "last_eval", None) or {}
    ids = set(r.get("pair_observation_ids") or [])
    pos = {g.get("observation_id"): g.get("target_odom_link_xy") for g in (getattr(pair, "group", None) or [])}
    out = []
    for o in ev.get("per_observation") or []:
        oid = o.get("observation_id")
        if o.get("reading_kind") != "ROOM" or not o.get("usable") or not o.get("labels") or (ids and oid not in ids) or not pos.get(oid):
            continue
        out.append((oid, str(o["labels"][0][0]), float(o["labels"][0][1]), np.asarray(pos[oid], float)))
    return out


def evaluate(link, r, cands, res, E7):
    """-> 판정 기록(dict). 순수 계산 + 저장소·지도 읽기만. E7 = e6e7_link 가 파일 경로로 불러온 운영 E7 엔진."""
    import e6_semantic as S
    plates = plate_readings(link, r)
    doc = dict(kind="LIVE2_FUSED_FLOOR", wall=time.time(), observation_id=r.get("observation_id"), params=dict(SIG_G=SIG_G, SIG_P=SIG_P, EPS=EPS, LAM=round(LAM, 3), TOL=TOL, XMAX=XMAX, PLATE_CAP=round(MISS, 3), PASS_PATH_M=PASS_PATH_M, JUMP_M=JUMP_M),
               plates=[dict(observation_id=o, label=l, conf=c) for o, l, c, _ in plates], confirmed=False)
    rs = [q for q in (res or {}).get("results") or [] if q.get("alignment_score") is not None]
    if not rs or not plates or res.get("t_end") is None:
        doc["why"] = "no E7 result" if not rs else ("no room plate in this pair" if not plates else "E7 result without time window")
        return doc
    fr = E7.frames(link.tr.store, float(res["t_end"]), float(res.get("window_s") or 90.0), structural=True)
    if len(fr) < 2:
        doc["why"] = f"only {len(fr)} frames of the E7 window left in the store"
        return doc
    try:                                                               # 출발부터 E7 끝까지의 주행 기록(저장소 행), 튄 곳에서 끊음
        rows_all = [(float(x[2]), float(x[3])) for x in (getattr(link.tr.store, "rows", None) or []) if float(x[1]) <= float(res["t_end"]) + 1e-6]
    except (TypeError, ValueError, IndexError):
        rows_all = []
    segs, cur = [], []
    for p_ in rows_all:
        if cur and math.hypot(p_[0] - cur[-1][0], p_[1] - cur[-1][1]) > JUMP_M:
            segs.append(cur); cur = []
        cur.append(p_)
    if cur:
        segs.append(cur)
    path_m = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for s_ in segs for a, b in zip(s_, s_[1:]))
    recent, acc = [], 0.0                                              # 최근 PASS_PATH_M 의 이어진 경로(마지막 조각 안에서)
    if segs:
        last = segs[-1]; i = len(last) - 1; recent = [last[i]]
        while i > 0 and acc < PASS_PATH_M:
            acc += math.hypot(last[i][0] - last[i - 1][0], last[i][1] - last[i - 1][1]); i -= 1; recent.append(last[i])
        recent.reverse()
    doc.update(full_path_m=round(path_m, 1), full_path_segments=len(segs), pass_path_m=round(acc, 1))
    try:
        import e3_tracker as E3T
        wall_pass_any = E3T.wall_pass_any
    except Exception as ex:                                           # noqa: BLE001  no check available -> fail closed below
        wall_pass_any = None
        doc["pass_check_error"] = f"{type(ex).__name__}: {ex}"
    byid = {c["candidate_id"]: c for c in cands}
    smin = min(float(q["alignment_score"]) for q in rs)
    rows = []
    for q in rs:
        c = byid.get(q.get("candidate_id"))
        if c is None:
            continue
        rec, fl, k = c["rec"], int(c["floor"]), q.get("correction") or {}
        d = (float(k.get("dx_m", 0.0)), float(k.get("dy_m", 0.0)), math.radians(float(k.get("dphi_deg", 0.0))), float(k.get("scale", 1.0)))
        f, ppm1 = placer(rec, fr, d)
        pd, pc = [], 0.0
        for _, lab, _, xy in plates:
            sp = link.sf.spaces(fl, lab)
            dm = None if not sp else min(S._poly_dist_px(s_["polygon_px"], f(xy)) for s_ in sp) / ppm1
            pd.append(None if dm is None else round(dm, 2))
            pc += min(dm * dm / (2 * SIG_P ** 2), MISS) if (dm is not None and dm <= TOL) else MISS
        try:
            xw = float(link.tr.light(rec["light"]).crossing_m(np.array([f([t[1], t[2]]) for t in fr]), ppm1))
        except Exception as ex:                                       # noqa: BLE001
            xw = None
            doc.setdefault("crossing_errors", []).append(f"{q.get('candidate_id')}: {type(ex).__name__}: {ex}")
        try:
            L_ = link.tr.light(rec["light"])
            xf = float(sum(L_.crossing_m(np.array([f(p_) for p_ in s_]), ppm1) for s_ in segs if len(s_) >= 2)) if segs else None
            wp = bool(wall_pass_any(L_.wall_tr, np.array([f(p_) for p_ in recent]), ppm1)) if (wall_pass_any is not None and len(recent) >= 2) else (None if wall_pass_any is None else False)
        except Exception as ex:                                       # noqa: BLE001
            xf, wp = None, None
            doc.setdefault("crossing_errors", []).append(f"{q.get('candidate_id')} long path: {type(ex).__name__}: {ex}")
        geo = (float(q["alignment_score"]) - smin) / SIG_G
        rows.append(dict(candidate_id=q["candidate_id"], floor=fl, score_m=q["alignment_score"], converged=bool(q.get("convergence")) and not q.get("on_bound"),
                         path_cross_m=None if xw is None else round(xw, 2), wall_pass_recent=wp, full_path_cross_m=None if xf is None else round(xf, 2),
                         plate_dist_m=pd, geo=round(geo, 3), plate=round(pc, 3), fused=round(geo + pc, 3)))
    ok = [x for x in rows if x["path_cross_m"] is not None and x["path_cross_m"] <= XMAX and x["converged"]
          and x["wall_pass_recent"] is False]     # 확정 위치로 쓸 수 있는 후보만(수렴 · 90 초 구간 벽 통과 없음 · 최근 30 m 벽 뚫기 없음; 검사 못 하면 뺌)
    per = {}
    for x in ok:
        if x["floor"] not in per or x["fused"] < per[x["floor"]]["fused"]:
            per[x["floor"]] = x
    floors = sorted(getattr(link.sf, "maps", {}) or range(1, 10))
    fl_score = {f: per[f]["fused"] for f in per}
    for f in floors:                                                   # E7 이 정합하지 않은 층: 기하 최선(0), 같은 번호 방이 있으면 그 명판 0 (보수적)
        if f not in fl_score:
            fl_score[f] = round(sum(0.0 if link.sf.spaces(f, lab) else math.log(1.0 / EPS) for _, lab, _, _ in plates), 3)
    rk = sorted(fl_score.items(), key=lambda kv: kv[1])
    doc.update(candidates=rows, floor_scores={str(k): v for k, v in rk}, n_removed_by_path=sum(1 for x in rows if x["path_cross_m"] is None or x["path_cross_m"] > XMAX),
               n_removed_by_wall_pass=sum(1 for x in rows if (x["path_cross_m"] is not None and x["path_cross_m"] <= XMAX) and x["wall_pass_recent"] is not False),
               n_not_converged=sum(1 for x in rows if not x["converged"]), t_end=res.get("t_end"), window_s=res.get("window_s"))
    if not per:
        doc["why"] = "no registered candidate is converged, free of wall crossing on the window path and not passing through a wall on the last 30 m"
        return doc
    fb, sb = rk[0]
    f2, s2 = (rk[1] if len(rk) > 1 else (None, None))
    best = per.get(fb)
    if best is not None:                                               # 층 안 위치: 명판이 모두 TOL 안인 후보 중 정합(기하)이 가장 좋은 것
        same = [x for x in ok if x["floor"] == fb and x["plate_dist_m"] and all(v is not None and v <= TOL for v in x["plate_dist_m"])]
        pos = min(same, key=lambda x: float(x["score_m"])) if same else best
        doc["position"] = dict(candidate_id=pos["candidate_id"], floor_score_candidate=best["candidate_id"], n_plate_ok=len(same),
                               why="best E7 fit among the plate-consistent candidates of the winning floor" if same else "no plate-consistent candidate: lowest fused score")
        best = pos
    gap = None if s2 is None else round(s2 - sb, 3)
    doc.update(best_floor=fb, second_floor=f2, gap=gap, best=None if best is None else best["candidate_id"])
    sp = r.get("single_plate")
    if sp:                                                             # B3: 오독 의심·닮은 층 (기하만으로 판단, 명판 항 없이)
        fp = int(sp.get("floor"))
        mine = per.get(fp)
        need = (LAM - MISS) * SIG_G
        oth = {f: x for f, x in per.items() if f != fp}
        contra = sorted(f for f, x in oth.items() if mine is not None and float(mine["geo"]) - float(x["geo"]) >= MISS)
        alike = sorted(f for f, x in oth.items() if mine is not None and float(x["score_m"]) - float(mine["score_m"]) < need)
        doc["single_plate"] = dict(observation_id=sp.get("observation_id"), label=sp.get("label"), floor=fp, candidate=None if mine is None else mine["candidate_id"],
                                   misread_suspect=bool(contra), contradicting_floors=contra, lookalike_floors=alike, need_geometry_m=round(need, 3),
                                   unregistered_floors=[f for f in floors if f != fp and f not in per], prior_suspects=int(sp.get("suspects") or 0),
                                   why=("no usable candidate on the plate floor: no judgement" if mine is None else
                                        f"geometry of {contra}F beats the plate floor by more than one plate weighs (>= {MISS * SIG_G:.2f} m): misread suspected" if contra else
                                        f"{len(alike)} floor(s) fit within {need:.2f} m: geometry cannot back one plate" if alike else "geometry backs the plate floor"))
    if best is None:
        doc["why"] = f"{fb}F leads only as an unregistered floor"
    elif gap is None or gap < LAM:
        doc["why"] = f"{fb}F leads {f2}F by {gap} < {LAM:.2f}"
    else:
        doc.update(confirmed=True, why=f"{fb}F by {gap:.2f} over {f2}F (>= {LAM:.2f}); plates {[l for _, l, _, _ in plates]} at {best['plate_dist_m']} m; path through walls {best['path_cross_m']} m, no wall pass on the last {doc.get('pass_path_m')} m; {best['candidate_id']} converged, E7 fit {best['score_m']} m")
    sd = doc.get("single_plate")
    if doc.get("confirmed") and sd:                                    # B3: 한 장 확정의 추가 조건 — 명판 층이 이기고, 의심 없고(이번·이전), 확정 위치가 명판과 맞을 것
        bad = ("the plate floor did not win" if fb != sd["floor"] else "misread suspected on this look" if sd["misread_suspect"] else
               f"misread suspected on {sd['prior_suspects']} earlier look(s)" if sd["prior_suspects"] else
               "no plate-consistent candidate for the position" if not (doc.get("position") or {}).get("n_plate_ok") else None)
        if bad:
            doc.update(confirmed=False, why=f"single plate: {bad} ({doc['why']})")
        else:
            doc["why"] = "single plate + geometry: " + doc["why"]
    return doc


def apply(link, r, cands, res, out, E7, sess0=None):
    """e6e7_link._run 끝에서 부른다. 꺼져 있거나 확정이 아니면 None(기존 결정 그대로), 확정이면 새 out. 기록은 항상 남긴다."""
    if not enabled() or (out or {}).get("global_decision") == "GLOBAL_CONFIRMED":
        return None
    doc = evaluate(link, r, cands, res, E7)
    link._fused_doc = doc                                              # B3: e6e7_link 가 오독 의심을 센다
    if doc.get("confirmed"):
        bad, why_lio = link.fastlio_degraded()
        if bad:
            doc.update(confirmed=False, why=f"would confirm, but {why_lio}")
        elif sess0 is not None and link._epoch() != sess0:
            doc.update(confirmed=False, why="would confirm, but ODOM_SESSION_CHANGED during E7")
    try:
        with open(link.state / "e7_fused_floor.jsonl", "a") as f:
            f.write(json.dumps(doc, default=str) + "\n")
    except OSError:
        pass
    if not doc.get("confirmed"):
        return None
    hid = doc["best"]
    c = next(x for x in cands if x["candidate_id"] == hid)
    q = next(x for x in res["results"] if x.get("candidate_id") == hid)
    try:                                                               # 확정 뒤 현재 위치(current_pose)와 화면용: 이 E7 보정으로 다시 놓은 변환
        import e7_long as EL_
        fr30 = E7.frames(link.tr.store, float(res["t_end"]), 30.0, structural=True)
        rec_fix = EL_.recentered(dict(c["rec"]), fr30, q["correction"]) if fr30 else dict(c["rec"])
        link._e7_ledger = dict(gid=r.get("observation_id"), passes=[], t_last=res["t_end"], history=[], rec_fix=dict(rec_fix, floor=c["floor"]))
    except Exception as ex:                                            # noqa: BLE001
        doc["ledger_error"] = f"{type(ex).__name__}: {ex}"
    return dict(out, global_decision="GLOBAL_CONFIRMED", path="FUSED_FLOOR", semantic_hypothesis=hid, why="FUSED_FLOOR: " + doc["why"],
                fused_floor={k: doc.get(k) for k in ("best_floor", "second_floor", "gap", "floor_scores", "plates", "params", "single_plate")}, fix_blocked=None,
                used_observation_ids=[p["observation_id"] for p in doc["plates"]], prior_decision=dict(global_decision=out.get("global_decision"), why=out.get("why")))
