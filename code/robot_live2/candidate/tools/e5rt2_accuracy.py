#!/usr/bin/env python3
"""E5RT2 accuracy path on STORED scene logs (CPU only, no model run). Subcommands share one observation loader.

  coldstart   legacy relative-p90 warm-up: time from the first scene to the first finite threshold, positives first seen
              before it, and first-verification delay (legacy p90 vs a fixed threshold) per positive spot.
  hardneg     per-rule L4 ablation (rule off / margin / quality-gated veto) with unique-spot TP/FP/FN and a group
              (spot) bootstrap interval. Rules are recomputed with the CURRENT policy code on the logged evidence.
  calibrate   fit a FIXED threshold on the DEV split only -> calibration artifact (e5rt.calibration format). Built from
              provisional labels it is marked human_verified=false and is NOT activatable at runtime.
  stageloss   per labelled spot: proposal present -> raw/margin -> temporal -> L4 -> (L5) -> verified -> logged STOP.
  reviewkit   human review kit: best crop + source frame per spot, a frame sheet for true sign boxes, CSV templates.

Label semantics: POS_ROOM / POS_NAME = positive; NEG_* = negative; AMBIGUOUS / POS_ROOM_UNREADABLE / unlabelled are
never mapped to negative (reported separately). Precision is None when a policy makes no positive decision.
Spots are legacy world-memory cells, i.e. derived from legacy proposals: a sign that never became a spot is invisible
here (proposal recall needs human sign boxes -> reviewkit). Consecutive frames are not independent: intervals are
bootstrapped over spots.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent
sys.path.insert(0, str(WS / "src" / "sci_visual_observation"))
import policy as PO  # noqa: E402
from e5rt import evidence as EV  # noqa: E402
from e5rt import calibration as CAL  # noqa: E402

POS = ("POS_ROOM", "POS_NAME")
EXCLUDED = ("AMBIGUOUS", "POS_ROOM_UNREADABLE")
LAB = Path("/home/unitree/sci_ws/datasets/labels/v3")
DATASETS = {
    "dev_S1F_v3": dict(run=Path("/home/unitree/sci_ws/experiments/dev_S1F_v3"), labels=LAB / "labels_S-1F_v3.json", floor="S-1F",
                       frames=Path("/home/unitree/sci_ws/datasets/bags/S-1F"), split="DEV", expect_prompts="roomstd"),
    # SEM1 fix: labels_S-6F_v3 belong to experiments/test_S6F_v3 (VALID, roomstd prompts). E5-RT1/E5RT2 used
    # test_S6F_v2 = INVALID run (wrong fallback prompt, different spot ids) -> those S6 numbers are void.
    "test_S6F_v3": dict(run=Path("/home/unitree/sci_ws/experiments/test_S6F_v3"), labels=LAB / "labels_S-6F_v3.json", floor="S-6F",
                        frames=Path("/home/unitree/sci_ws/datasets/bags/S-6F"), split="SEEN_REGRESSION_NOT_CLEAN", expect_prompts="roomstd"),
}


def check_label_run(cfg):
    """Refuse to score a run with labels made for another run (spot ids are run-specific)."""
    lab = json.loads(Path(cfg["labels"]).read_text())
    want = str(lab.get("run", "")).rstrip("/")
    if not want or not str(cfg["run"]).rstrip("/").endswith(want):
        raise SystemExit(f"LABEL_RUN_MISMATCH: labels {cfg['labels']} are for {want!r}, run is {cfg['run']}")
    p = Path(cfg["run"]) / "siglip_prompts.json"
    if not p.exists():
        raise SystemExit(f"PROMPT_PROVENANCE_MISSING: {p} (runs without it may use the wrong fallback prompt)")
    pr = json.loads(p.read_text())
    if cfg.get("expect_prompts") and pr.get("env") != cfg["expect_prompts"]:
        raise SystemExit(f"PROMPT_MISMATCH: run {cfg['run']} used {pr.get('env')}, evaluation expects {cfg['expect_prompts']}")
    fp = lab.get("run_fingerprint")                         # exact identity when the label file records it
    if fp:
        got = dict(scenes_sha256=sha(Path(cfg["run"]) / "scenes.jsonl"), prompts_sha256=sha(p))
        bad = [k for k, v in got.items() if fp.get(k) and fp[k] != v]
        if bad:
            raise SystemExit(f"RUN_FINGERPRINT_MISMATCH: {bad}")
    cfg["identity_status"] = "RUN_FINGERPRINT_VERIFIED" if fp else "RUN_FINGERPRINT_NOT_RECORDED_IN_LABELS(v3): path+prompt checks only"
    return pr


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def label_of(labels, floor, spot):
    return labels.get(f"{floor}|spot{spot}") if labels else None


def kind(lab):
    if lab is None:
        return "UNLABELLED"
    if lab in POS:
        return "POS"
    if lab in EXCLUDED:
        return "EXCLUDED"
    return "NEG" if lab.startswith("NEG") else "EXCLUDED"


# ------------------------------------------------------------------------------------------------ observations
def load_scenes(run):
    for cfg in DATASETS.values():
        if Path(cfg["run"]) == Path(run):
            check_label_run(cfg)
    out = []
    for line in open(Path(run) / "scenes.jsonl"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("diagnostic_only") or (r.get("timing") or {}).get("paused"):
            continue
        out.append(r)
    return out


def _feat(g):
    f = dict(g.get("feat") or {})
    if (f.get("plane") or {}).get("normal") is not None:
        f["plane"] = dict(f["plane"], normal=np.asarray(f["plane"]["normal"], float))
    if f.get("up") is not None:
        f["up"] = np.asarray(f["up"], float)
    return f


def observations(scenes, sp):
    """One record per (scene, region with a spot). L4 recomputed with StagePolicy `sp` (variant config)."""
    obs = []
    for r in scenes:
        for g in r.get("regions") or []:
            if g.get("spot") is None or (g.get("score_status") or "FRESH") != "FRESH":
                continue
            dec = g.get("dec") or {}
            a2 = ((dec.get("A2") or {}).get("reasons") or []) + ((dec.get("P2") or {}).get("reasons") or [])
            keep = sorted({w for w in a2 if w.startswith(("width", "height", "board"))})   # config-independent legacy filters
            f = _feat(g)
            hn = sp.hardneg_gate(g, f) if f.get("glare") and f.get("plane") is not None else []
            neg = g.get("neg") or {}
            conf = [neg.get(c) for c in ("poster", "board", "directory", "light") if neg.get(c) is not None]
            obs.append(dict(t=float(r["t"]), spot=g["spot"], sim=float(g["sim"]), thr=None if r.get("thr") is None else float(r["thr"]),
                            conf=max(conf) if conf else None, margin=(float(g["sim"]) - max(conf)) if conf else None,
                            quality=1.0, speed=r.get("speed"), L4=bool(keep + hn), L4_rules=keep + hn, L5=False,
                            logged_state=((dec.get("P2") or {}).get("state")), box=g.get("box"), tile=g.get("tile"),
                            frame_id=(r.get("frame") or {}).get("frame_id"), stamp=r.get("stamp")))
    obs.sort(key=lambda o: o["t"])
    return obs


def run_policy(obs, fixed_thr=None, veto=lambda o: o["L4"]):
    pol, first, ver = {}, {}, {}
    for o in obs:
        s = o["spot"]
        if s not in pol:
            pol[s], first[s] = EV.Mean3Legacy(), o["t"]
        oo = dict(o, thr=fixed_thr if fixed_thr is not None else o["thr"])
        if oo["thr"] is None:
            oo["thr"] = math.inf
        st = pol[s].update(oo)
        if s not in ver and st == EV.CONFIRMED and not veto(o):
            ver[s] = o["t"]
    return first, ver


def score(first, ver, labels, floor, n_boot=1000, seed=0):
    rows = []
    for s in sorted(set(first)):
        k = kind(label_of(labels, floor, s))
        rows.append((k, s in ver))
    tp = sum(1 for k, v in rows if k == "POS" and v)
    fp = sum(1 for k, v in rows if k == "NEG" and v)
    fn = sum(1 for k, v in rows if k == "POS" and not v)
    excl_v = sum(1 for k, v in rows if k in ("EXCLUDED", "UNLABELLED") and v)
    out = dict(tp=tp, fp=fp, fn=fn, precision=None if tp + fp == 0 else tp / (tp + fp),
               precision_null_reason="NO_POSITIVE_DECISIONS" if tp + fp == 0 else None,
               recall=None if tp + fn == 0 else tp / (tp + fn), verified_excluded_or_unlabelled=excl_v,
               n_pos=tp + fn, n_neg=sum(1 for k, _ in rows if k == "NEG"))
    lab = [(k, v) for k, v in rows if k in ("POS", "NEG")]
    if lab:
        rng = np.random.RandomState(seed)
        P, R = [], []
        for _ in range(n_boot):
            smp = [lab[i] for i in rng.randint(0, len(lab), len(lab))]
            t_ = sum(1 for k, v in smp if k == "POS" and v)
            f_ = sum(1 for k, v in smp if k == "NEG" and v)
            n_ = sum(1 for k, v in smp if k == "POS" and not v)
            if t_ + f_:
                P.append(t_ / (t_ + f_))
            if t_ + n_:
                R.append(t_ / (t_ + n_))
        out["precision_ci90"] = [float(np.percentile(P, 5)), float(np.percentile(P, 95))] if P else None
        out["recall_ci90"] = [float(np.percentile(R, 5)), float(np.percentile(R, 95))] if R else None
        out["bootstrap"] = f"{n_boot}x over labelled spots (group = legacy spot; physical sign identity not verified)"
    return out


def variants(base_cfg):
    def v(**chg):
        c = copy.deepcopy(base_cfg)
        for k, val in chg.items():
            c["A2"][k] = val
        return c
    cls = list(base_cfg["A2"]["neg_classes"])
    return {
        "current": v(),
        "tile_light_off": v(neg_classes=[c for c in cls if c != "light"]),
        "tile_poster_off": v(neg_classes=[c for c in cls if c != "poster"]),
        "tile_light_poster_off": v(neg_classes=[c for c in cls if c not in ("light", "poster")]),
        "tile_neg_all_off": v(neg_classes=[]),
        "neg_margin_0.01": v(neg_margin=0.01),
        "neg_margin_0.02": v(neg_margin=0.02),
        "neg_margin_0.03": v(neg_margin=0.03),
        "ctx_off": v(ctx_classes=[]),
    }


# ------------------------------------------------------------------------------------------------ commands
def cmd_hardneg(a):
    base = PO.load_cfg(str(WS / "config" / "e5_policy.json"))
    rows = []
    for ds, cfg in DATASETS.items():
        labels = json.loads(cfg["labels"].read_text())["labels"]
        sc = load_scenes(cfg["run"])
        for name, c in variants(base).items():
            sp = PO.StagePolicy("abl", c, None, a1=True, a2=True, a3=False, a4=False)
            obs = observations(sc, sp)
            first, ver = run_policy(obs)
            r = dict(dataset=ds, split=cfg["split"], variant=name, n_spots=len(first), n_verified=len(ver), **score(first, ver, labels, cfg["floor"]))
            # which rules fire on positives vs negatives (observation level, this variant)
            fire = {}
            for o in obs:
                k = kind(label_of(labels, cfg["floor"], o["spot"]))
                if k not in ("POS", "NEG"):
                    continue
                for w in {x.split(":")[0] for x in o["L4_rules"]}:
                    fire.setdefault(w, {"POS": 0, "NEG": 0})[k] += 1
            npos = sum(1 for o in obs if kind(label_of(labels, cfg["floor"], o["spot"])) == "POS")
            nneg = sum(1 for o in obs if kind(label_of(labels, cfg["floor"], o["spot"])) == "NEG")
            r["rule_fire_obs"] = {w: dict(pos=f"{d['POS']}/{npos}", neg=f"{d['NEG']}/{nneg}") for w, d in sorted(fire.items())}
            rows.append(r)
    write(a.out, "hardneg_ablation", rows)


def first_finite(sc):
    t0 = sc[0]["t"] if sc else None
    for r in sc:
        if r.get("thr") is not None and math.isfinite(float(r["thr"])):
            return t0, r["t"]
    return t0, None


def cmd_coldstart(a):
    rows = []
    runs = [(ds, cfg["run"], cfg) for ds, cfg in DATASETS.items()]
    runs += [(p.name, p, None) for p in sorted(Path("/home/unitree/sci_ws/deploy/kci_r1_20261017/experiments").glob("kci_e5_20261017_*"))]
    runs += [(p.name, p, None) for p in sorted(Path(a.extra_runs).glob("*")) if (p / "scenes.jsonl").exists()] if a.extra_runs else []
    # FOLLOW-UP: unlabelled / extra runs keep their timing statistics, but are marked when their prompt provenance is
    # missing or different (no silent pass on path suffix).
    base = PO.load_cfg(str(WS / "config" / "e5_policy.json"))
    sp = PO.StagePolicy("cur", base, None, a1=True, a2=True, a3=False, a4=False)
    cal = json.loads(Path(a.artifact).read_text()) if a.artifact else None
    for name, run, cfg in runs:
        sc = load_scenes(run)
        if not sc:
            continue
        t0, tf = first_finite(sc)
        pj = Path(run) / "siglip_prompts.json"
        prov = (json.loads(pj.read_text()).get("env") if pj.exists() else "MISSING")
        r = dict(run=name, split=(cfg or {}).get("split", "UNLABELLED"), prompt_provenance=prov,
                 accuracy_usable=bool(cfg) and prov == "roomstd", n_scenes=len(sc), first_scene_t=t0, first_finite_thr_t=tf,
                 warmup_s=None if tf is None else round(tf - t0, 1),
                 n_sims_before=sum(len(x.get("regions") or []) for x in sc if tf is None or x["t"] < tf))
        if cfg:
            labels = json.loads(cfg["labels"].read_text())["labels"]
            obs = observations(sc, sp)
            first, ver_l = run_policy(obs)
            pos = [s for s in first if kind(label_of(labels, cfg["floor"], s)) == "POS"]
            r["pos_first_seen_before_finite_thr"] = sum(1 for s in pos if tf is None or first[s] < tf)
            r["pos_verify_delay_legacy_s"] = sorted(round(ver_l[s] - first[s], 1) for s in pos if s in ver_l)
            if cal:
                _, ver_c = run_policy(obs, fixed_thr=cal["thresholds"]["global"])
                r["pos_verify_delay_fixed_s"] = sorted(round(ver_c[s] - first[s], 1) for s in pos if s in ver_c)
                r["fixed_thr_status"] = f"{cal['calibration_id']} human_verified={cal['human_verified']} (fitted on DEV)"
        rows.append(r)
    write(a.out, "coldstart", rows)


def cmd_calibrate(a):
    cfg = DATASETS["dev_S1F_v3"]
    lab_doc = json.loads(cfg["labels"].read_text())
    labels = lab_doc["labels"]
    base = PO.load_cfg(str(WS / "config" / "e5_policy.json"))
    sp = PO.StagePolicy("cur", base, None, a1=True, a2=True, a3=False, a4=False)
    obs = observations(load_scenes(cfg["run"]), sp)
    sims = sorted({round(o["sim"], 4) for o in obs})
    grid = [x for x in np.arange(0.078, max(sims) + 0.002, 0.002)]
    best = None
    table = []
    for thr in grid:
        first, ver = run_policy(obs, fixed_thr=float(thr))
        s = score(first, ver, labels, cfg["floor"], n_boot=0)
        p, rc = s["precision"], s["recall"]
        f1 = None if not p or not rc else 2 * p * rc / (p + rc)
        table.append(dict(thr=round(float(thr), 4), tp=s["tp"], fp=s["fp"], fn=s["fn"], precision=p, recall=rc, f1=f1))
        if f1 is not None and (best is None or f1 > best["f1"] or (f1 == best["f1"] and thr > best["thr"])):
            best = dict(thr=float(thr), f1=f1, tp=s["tp"], fp=s["fp"], fn=s["fn"])
    # FOLLOW-UP: human verification is an EXPLICIT boolean of the label file, never inferred from the labeller text
    verified = lab_doc.get("human_verified") is True
    art = dict(method="mean3_legacy_temporal + fixed global threshold on msim, max F1 on DEV spots (L4 current config)",
               split="DEV", score="raw cosine (max over prompts/tiles), msim = mean of last 3 fresh sims",
               thresholds={"global": None if best is None else round(best["thr"], 4)},
               data=dict(run=str(cfg["run"]), scenes_sha256=sha(Path(cfg["run"]) / "scenes.jsonl"), labels=str(cfg["labels"]),
                         labels_sha256=sha(cfg["labels"]), labeller=lab_doc.get("labeller")),
               human_verified=bool(verified), created=time.strftime("%F %T"),
               dev_fit=best, dev_grid=table,
               status="PROVISIONAL_NOT_ACTIVATABLE (labels are not human-verified)" if not verified else "DEV_FIXED")
    art["calibration_id"] = CAL.artifact_id(art)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration_dev_S1F_v3_PROVISIONAL.json").write_text(json.dumps(art, indent=1))
    tm = CAL.ThresholdMode(mode="calibrated", artifact_path=str(out / "calibration_dev_S1F_v3_PROVISIONAL.json"))
    print(json.dumps(dict(calibration_id=art["calibration_id"], best=best, runtime_state=tm.state, why=tm.why), indent=1))
    # SEEN (not clean) evaluation of the DEV-fixed threshold, reported as such
    rows = []
    for ds, c in DATASETS.items():
        lb = json.loads(c["labels"].read_text())["labels"]
        ob = observations(load_scenes(c["run"]), sp)
        for nm, thr in (("legacy_p90", None), ("fixed_dev_thr", art["thresholds"]["global"])):
            first, ver = run_policy(ob, fixed_thr=thr)
            rows.append(dict(dataset=ds, split=c["split"], mode=nm, thr=thr, **score(first, ver, lb, c["floor"])))
    write(a.out, "calibration_eval", rows)


def cmd_stageloss(a):
    rows = []
    base = PO.load_cfg(str(WS / "config" / "e5_policy.json"))
    sp = PO.StagePolicy("cur", base, None, a1=True, a2=True, a3=False, a4=False)
    for ds, cfg in DATASETS.items():
        labels = json.loads(cfg["labels"].read_text())["labels"]
        sc = load_scenes(cfg["run"])
        obs = observations(sc, sp)
        first, ver = run_policy(obs)
        stops = set()
        sp_path = Path(cfg["run"]) / "stops.jsonl"
        if sp_path.exists():
            for line in open(sp_path):
                try:
                    stops.add(json.loads(line).get("spot"))
                except ValueError:
                    pass
        for key, lab in sorted(labels.items()):
            spot = int(key.split("spot")[1])
            o = [x for x in obs if x["spot"] == spot]
            thr_ok = [x for x in o if x["thr"] is not None and math.isfinite(x["thr"]) and x["sim"] >= x["thr"]]
            rows.append(dict(dataset=ds, split=cfg["split"], spot=spot, label=lab, kind=kind(lab), proposal_present=bool(o), n_obs=len(o),
                             max_sim=max((x["sim"] for x in o), default=None), max_margin=max((x["margin"] for x in o if x["margin"] is not None), default=None),
                             obs_sim_ge_thr=len(thr_ok), obs_L4=sum(1 for x in o if x["L4"]),
                             L4_rules=sorted({w.split(":")[0] for x in o for w in x["L4_rules"]}),
                             verified_rederived=spot in ver, logged_stop=spot in stops,
                             L5="NOT_APPLIED (pre-P2 run: P2 geometry layer not logged)"))
    write(a.out, "stage_loss", rows)


def cmd_reviewkit(a):
    import cv2
    out = Path(a.out) / "review_kit"
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rows = []
    base = PO.load_cfg(str(WS / "config" / "e5_policy.json"))
    sp = PO.StagePolicy("cur", base, None, a1=True, a2=True, a3=False, a4=False)
    for ds, cfg in DATASETS.items():
        labels = json.loads(cfg["labels"].read_text())["labels"]
        idx = list(csv.DictReader(open(cfg["frames"] / "index.csv")))
        st = np.array([float(r["stamp"]) for r in idx])
        obs = observations(load_scenes(cfg["run"]), sp)
        best = {}
        for o in obs:
            if o["spot"] not in best or o["sim"] > best[o["spot"]]["sim"]:
                best[o["spot"]] = o
        for spot, o in sorted(best.items()):
            i = int(np.argmin(np.abs(st - float(o["stamp"])))) if o.get("stamp") is not None else None
            if i is None:
                continue
            img = cv2.imread(str(cfg["frames"] / idx[i]["rgb"]))
            if img is None:
                continue
            b = [int(round(v)) for v in (o["tile"] or o["box"])]
            w, h = b[2] - b[0], b[3] - b[1]
            crop = img[max(0, b[1] - h):b[3] + h, max(0, b[0] - w):b[2] + w]
            cid = f"{ds}_spot{spot}"
            cv2.imwrite(str(out / "crops" / f"{cid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            fr = img.copy()
            cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cv2.imwrite(str(out / "frames" / f"{cid}_source.jpg"), cv2.resize(fr, (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 80])
            rows.append(dict(item_id=cid, dataset=ds, split=cfg["split"], floor=cfg["floor"], spot=spot, frame_index=i,
                             frame_rgb=idx[i]["rgb"], stamp=o["stamp"], tile=b, sim=round(o["sim"], 4),
                             provisional_label=label_of(labels, cfg["floor"], spot) or "", human_label="", physical_target_id="",
                             crop_shows_target="", readable="", human_verified="", reviewer="", notes=""))
    with open(out / "review_items.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # frame sheet for TRUE sign boxes (proposal recall needs targets the pipeline may have missed): 1 frame / 2 s
    fs = []
    for ds, cfg in DATASETS.items():
        idx = list(csv.DictReader(open(cfg["frames"] / "index.csv")))
        st = np.array([float(r["stamp"]) for r in idx])
        for t in np.arange(st[0], st[-1], 2.0):
            i = int(np.argmin(np.abs(st - t)))
            fs.append(dict(dataset=ds, split=cfg["split"], frame_index=i, frame_rgb=str(cfg["frames"] / idx[i]["rgb"]), stamp=float(st[i]),
                           sign_boxes_xyxy="", sign_types="", physical_target_ids="", reviewer="", human_verified=""))
    with open(out / "frame_sheet_sign_boxes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fs[0].keys()))
        w.writeheader()
        w.writerows(fs)
    (out / "README_REVIEW_KO.md").write_text(REVIEW_README)
    print(json.dumps(dict(items=len(rows), frame_sheet=len(fs), out=str(out))))


REVIEW_README = """# 사람 검토 키트 (E5RT2)

목적: 표지판 정확도를 잠정 라벨(Claude 판정) 대신 사람이 확인한 라벨로 평가하기 위함. 이 키트를 채우기 전까지
SIGN_ACCURACY_VERIFIED=NO 이고 calibration 은 활성화되지 않는다(e5rt/calibration.py: human_verified=false 이면 거부).

1. review_items.csv (모델이 본 crop 단위)
   - crops/<item_id>.jpg: 모델이 가장 높은 점수를 준 tile 주변(상하좌우 1배 여유). frames/<item_id>_source.jpg: 원본 프레임 축소+상자.
   - human_label 하나만 고른다: POS_ROOM(호실판) POS_NAME(연구실명/사무실명) NEG_POSTER NEG_GLARE(반사·조명) NEG_SWITCH
     NEG_BOARD(게시판) NEG_DIRECTORY(층 안내) NEG_SHOP(상호명) NEG_DIRECTION(방향 안내) NEG_OTHER AMBIGUOUS UNREADABLE
   - crop_shows_target: 사진에 원하는 표지가 있어도 crop 이 다른 대상을 보면 N 으로 따로 적는다.
   - physical_target_id: 같은 실제 표지판은 같은 id (예: S1F-101). 분할은 이 id 단위로 한다(같은 표지의 여러 프레임을
     DEV/VAL/FINAL 에 섞지 않는다).
   - human_verified: 본인이 직접 확인했으면 Y. 자동으로 Y 로 바꾸지 않는다.
2. frame_sheet_sign_boxes.csv (원본 프레임 단위, 2초 간격)
   - 모델이 놓친 표지판까지 분모에 넣기 위해 실제 표지판 상자(x0,y0,x1,y1; 여러 개는 ;)와 종류, physical_target_id 를 적는다.
3. 분할 규칙: S-1F = DEV(보정용), S-6F = 이미 본 자료(SEEN, 최종 평가 아님). 최종 평가는 새 세션·새 표지판의 group-disjoint 자료.
"""


def write(out, name, rows):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(rows, indent=1, default=str))
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
    print(f"{name}: {len(rows)} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["coldstart", "hardneg", "calibrate", "stageloss", "reviewkit"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--extra-runs", default=None)
    a = ap.parse_args()
    dict(coldstart=cmd_coldstart, hardneg=cmd_hardneg, calibrate=cmd_calibrate, stageloss=cmd_stageloss, reviewkit=cmd_reviewkit)[a.cmd](a)


if __name__ == "__main__":
    main()
