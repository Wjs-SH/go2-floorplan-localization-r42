#!/usr/bin/env python3
"""SEM1 data tools (CPU, stored data only; nothing here creates truth).

  audit     -> SEM1_DATA_AUDIT.csv : run/prompt/model/label provenance, label-run match, validity, denominators.
  labels    -> labels v4 (NEW revision): two axes (visual_type, localization_role) + readability / target_in_model_crop /
               physical_target_id / eligible_view_interval fields. v3 values are only MAPPED as `proposed_*` with
               proposed_by=auto_map_from_v3; human fields stay empty; human_verified is never set here.
  priority  -> human review priority bundle: S1 positives(9)+unreadable(2), S6 positives(17), >=20 hard negatives,
               per spot a chronological clip (first / best / last + up to 3 more), source frame + box, and a JPEG
               CONTEXT PREVIEW that is explicitly NOT the encoder input; plus a list of frames to scan for MISSED signs.
  ui        -> LABEL_UI.html (static, offline): click/keyboard labelling, localStorage autosave, CSV export with history.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAB = Path("/home/unitree/sci_ws/datasets/labels/v3")
EXP = Path("/home/unitree/sci_ws/experiments")
DS = {"S-1F": dict(labels=LAB / "labels_S-1F_v3.json", frames=Path("/home/unitree/sci_ws/datasets/bags/S-1F"), split="DEV"),
      "S-6F": dict(labels=LAB / "labels_S-6F_v3.json", frames=Path("/home/unitree/sci_ws/datasets/bags/S-6F"), split="SEEN_REGRESSION_NOT_CLEAN")}
VISUAL = ["ROOM_PLATE", "ROOM_NAME", "SHOP_SIGN", "DIRECTORY", "DIRECTIONAL_SIGN", "POSTER", "REFLECTION", "SWITCH", "HANDLE", "OTHER", "UNKNOWN"]
ROLE = ["LOCAL_ANCHOR", "DIRECTIONAL_CUE", "NONLOCAL_TEXT", "UNKNOWN"]
V3_MAP = {  # PROPOSAL only (needs human confirmation)
    "POS_ROOM": ("ROOM_PLATE", "LOCAL_ANCHOR", "READABLE_UNVERIFIED"), "POS_NAME": ("ROOM_NAME", "LOCAL_ANCHOR", "READABLE_UNVERIFIED"),
    "POS_ROOM_UNREADABLE": ("ROOM_PLATE", "LOCAL_ANCHOR", "UNREADABLE"), "NEG_BOARD": ("POSTER", "NONLOCAL_TEXT", "UNKNOWN"),
    "NEG_POSTER": ("POSTER", "NONLOCAL_TEXT", "UNKNOWN"), "NEG_DIRECTORY": ("DIRECTORY", "DIRECTIONAL_CUE", "UNKNOWN"),
    "NEG_GLARE": ("REFLECTION", "UNKNOWN", "NOT_TEXT"), "NEG_EXIT": ("DIRECTIONAL_SIGN", "DIRECTIONAL_CUE", "UNKNOWN"),
    "NEG_OTHER": ("OTHER", "UNKNOWN", "UNKNOWN"), "AMBIGUOUS": ("UNKNOWN", "UNKNOWN", "UNKNOWN"),
    "NEG_SHOP": ("SHOP_SIGN", "UNKNOWN", "UNKNOWN")}   # NOT a universal negative: a fixed shop name can be a LOCAL_ANCHOR


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scenes(run):
    out = []
    for l in open(Path(run) / "scenes.jsonl"):
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out


def index(frames):
    rows = list(csv.DictReader(open(frames / "index.csv")))
    return rows, np.array([float(r["stamp"]) for r in rows])


# ------------------------------------------------------------------------------------------------ audit
def cmd_audit(a):
    rows = []
    valid = {"dev_S1F_v3": "VALID (SESSION_CHECKPOINT: v3 runs valid)", "test_S6F_v3": "VALID (SESSION_CHECKPOINT)",
             "test_S6F_v2": "INVALID: wrong fallback prompt 'a small sign mounted on a wall' (SESSION_CHECKPOINT sec.6 item 2)",
             "dev_S1F_v1": "INVALID: wrong fallback prompt", "dev_S1F_v2": "INVALID: wrong fallback prompt", "test_S6F_v1": "INVALID: wrong fallback prompt"}
    for name in ("dev_S1F_v1", "dev_S1F_v2", "dev_S1F_v3", "test_S6F_v1", "test_S6F_v2", "test_S6F_v3"):
        d = EXP / name
        if not d.exists():
            continue
        pj = d / "siglip_prompts.json"
        pr = json.loads(pj.read_text()) if pj.exists() else None
        args = json.loads((d / "args.json").read_text()) if (d / "args.json").exists() else {}
        sc = scenes(d)
        rows.append(dict(item=f"run:{name}", path=str(d), prompts=(pr or {}).get("env") or "MISSING (no siglip_prompts.json)",
                         model=(pr or {}).get("model"), n_scenes=len(sc), dataset=args.get("dataset"), validity=valid.get(name, "UNKNOWN"),
                         scenes_sha256=sha(d / "scenes.jsonl")))
    for fl, cfg in DS.items():
        lab = json.loads(cfg["labels"].read_text())
        L = lab["labels"]
        cnt = {}
        for v in L.values():
            cnt[v] = cnt.get(v, 0) + 1
        pos = sum(cnt.get(k, 0) for k in ("POS_ROOM", "POS_NAME"))
        rows.append(dict(item=f"labels:{fl}", path=str(cfg["labels"]), labels_run=lab.get("run"), labeller=lab.get("labeller"),
                         n_labels=len(L), counts=json.dumps(cnt, sort_keys=True), pos_room_name=pos,
                         pos_incl_unreadable=pos + cnt.get("POS_ROOM_UNREADABLE", 0), labels_sha256=sha(cfg["labels"]),
                         human_verified="NO (Claude provisional)", split=cfg["split"]))
    rows.append(dict(item="analysis:E5RT1+E5RT2 S-6F", validity="VOID: scored labels_S-6F_v3 (spots of test_S6F_v3) against test_S6F_v2 (INVALID prompt, different spot ids)",
                     note="E5RT2 '0/17', 'legacy 2/17', 'fixed thr does not transfer', 'positives max 0.03-0.08' all retracted; recomputed on test_S6F_v3 in SEM1"))
    rows.append(dict(item="analysis:E5RT2 deterministic count", validity="CORRECTED: 108 identical scenes (4x27), not 107; VERIFIED spots = 0 in all 108",
                     note="the 107 came from the superseded first candidate run (26 scenes in 120-150); output-preservation test, NOT positive recognition"))
    rows.append(dict(item="denominators:S-1F", note="POS_ROOM+POS_NAME = 9; + POS_ROOM_UNREADABLE 2 = 11 (older reports); detection vs readability vs OCR denominators must be separated"))
    rows.append(dict(item="review_kit:E5RT2", path="/home/unitree/sci_ws/repairs/e5rt2_20261018_000958/benchmarks/accuracy/review_kit",
                     note="972 crops / 336 frame rows on robot; bundle had 21 previews. crops = JPEG tile+1x context PREVIEW, NOT encoder input; best-score crop only (no chronology)"))
    rows.append(dict(item="preprocessing:provenance", note="model preprocessor_config: resize bicubic(resample=3) 224, mean=std=0.5; pipeline uses cv2 INTER_LINEAR square crop CLIPPED at borders (aspect distortion at image edge), BGR->RGB, mean/std 0.5 -> differs from HF processor resampling; consistent between S1 and S6 v3 runs (same code)"))
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"audit rows {len(rows)} -> {a.out}")


# ------------------------------------------------------------------------------------------------ labels v4
def spot_obs(run):
    obs = {}
    for r in scenes(run):
        for g in r.get("regions") or []:
            if g.get("spot") is None:
                continue
            obs.setdefault(g["spot"], []).append(dict(t=float(r["t"]), stamp=r.get("stamp"), sim=float(g["sim"]), box=g.get("box"),
                                                      tile=g.get("tile") or g.get("box"), xyz=g.get("xyz")))
    return obs


def cmd_labels(a):
    out = dict(revision="v4-proposal", created=time.strftime("%F %T"), schema=dict(visual_type=VISUAL, localization_role=ROLE,
               fields=["visual_type", "localization_role", "text_readability", "target_in_model_crop", "physical_target_id", "source_run",
                       "frame_id", "native_bbox", "polygon", "reviewer", "reviewed_at", "confidence_note", "eligible_view_interval"]),
               note="v3 labels are untouched; this file only PROPOSES a mapping. Human fields are empty. human_verified is never set by tools.",
               items={})
    for fl, cfg in DS.items():
        lab = json.loads(cfg["labels"].read_text())
        run = EXP / Path(lab["run"]).name
        obs = spot_obs(run)
        for key, v3 in lab["labels"].items():
            spot = int(key.split("spot")[1])
            o = obs.get(spot, [])
            vt, role, rd = V3_MAP.get(v3, ("UNKNOWN", "UNKNOWN", "UNKNOWN"))
            out["items"][key] = dict(floor=fl, split=cfg["split"], source_run=lab["run"], spot=spot, v3_label=v3, v3_note=lab.get("notes", {}).get(key),
                                     proposed_visual_type=vt, proposed_localization_role=role, proposed_text_readability=rd,
                                     proposed_by="auto_map_from_v3", visual_type=None, localization_role=None, text_readability=None,
                                     target_in_model_crop=None, physical_target_id=None, native_bbox=None, polygon=None, reviewer=None,
                                     reviewed_at=None, confidence_note=None,
                                     eligible_view_interval=dict(first_t=min((x["t"] for x in o), default=None), last_t=max((x["t"] for x in o), default=None),
                                                                 note="observed-by-pipeline interval, NOT the human eligible interval"),
                                     human_verified=False)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"labels v4 proposal items {len(out['items'])} -> {a.out}")


# ------------------------------------------------------------------------------------------------ priority bundle
def cmd_priority(a):
    import cv2
    out = Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    items = []
    hard_neg = ("NEG_POSTER", "NEG_GLARE", "NEG_BOARD", "NEG_EXIT", "NEG_DIRECTORY", "NEG_OTHER", "AMBIGUOUS")
    for fl, cfg in DS.items():
        lab = json.loads(cfg["labels"].read_text())
        run = EXP / Path(lab["run"]).name
        obs = spot_obs(run)
        rows, st = index(cfg["frames"])
        sel = []
        for key, v in lab["labels"].items():
            if v in ("POS_ROOM", "POS_NAME", "POS_ROOM_UNREADABLE"):
                sel.append((0 if v != "POS_ROOM_UNREADABLE" else 1, key, v))
        negs = sorted(((k, v) for k, v in lab["labels"].items() if v in hard_neg), key=lambda kv: -max((x["sim"] for x in obs.get(int(kv[0].split("spot")[1]), [])), default=0))
        sel += [(2, k, v) for k, v in negs]
        for pri, key, v in sel:
            spot = int(key.split("spot")[1])
            o = sorted(obs.get(spot, []), key=lambda x: x["t"])
            if not o:
                items.append(dict(item=key, priority=pri, v3=v, n_obs=0, note="spot has no observation in its run"))
                continue
            best = max(range(len(o)), key=lambda i: o[i]["sim"])
            pick = sorted({0, best, len(o) - 1, len(o) // 3, (2 * len(o)) // 3})
            clip = []
            for i in pick:
                x = o[i]
                k = int(np.argmin(np.abs(st - float(x["stamp"])))) if x.get("stamp") is not None else None
                if k is None:
                    continue
                img = cv2.imread(str(cfg["frames"] / rows[k]["rgb"]))
                if img is None:
                    continue
                b = [int(round(v)) for v in x["tile"]]
                fr = img.copy()
                cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cv2.putText(fr, f"{key} t={x['t']:.1f} sim={x['sim']:.3f} {'BEST' if i == best else ''}", (6, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                fn = f"{fl}_spot{spot}_{i:03d}"
                cv2.imwrite(str(out / "clips" / f"{fn}_frame.jpg"), cv2.resize(fr, (480, 360)), [cv2.IMWRITE_JPEG_QUALITY, 85])
                w, h = b[2] - b[0], b[3] - b[1]
                prev = img[max(0, b[1] - h):b[3] + h, max(0, b[0] - w):b[2] + w]
                cv2.imwrite(str(out / "clips" / f"{fn}_CONTEXT_PREVIEW_not_encoder_input.jpg"), prev, [cv2.IMWRITE_JPEG_QUALITY, 90])
                clip.append(dict(i=i, t=x["t"], sim=round(x["sim"], 4), frame_index=k, frame_rgb=rows[k]["rgb"], tile=b, frame_img=f"clips/{fn}_frame.jpg",
                                 preview=f"clips/{fn}_CONTEXT_PREVIEW_not_encoder_input.jpg", best=i == best))
            items.append(dict(item=key, floor=fl, split=cfg["split"], priority=pri, v3=v, spot=spot, n_obs=len(o), first_t=o[0]["t"], last_t=o[-1]["t"],
                              best_sim=round(o[best]["sim"], 4), clip=clip))
    # frames to scan for MISSED signs: 1 frame / 2 s where the pipeline had no region >= floor (possible misses)
    miss = []
    for fl, cfg in DS.items():
        lab = json.loads(cfg["labels"].read_text())
        sc = scenes(EXP / Path(lab["run"]).name)
        for r in sc:
            if int(r["t"]) % 2 == 0 and not any((g.get("sim") or 0) >= 0.078 for g in r.get("regions") or []):
                miss.append(dict(floor=fl, t=r["t"], stamp=r.get("stamp"), reason="no region >= sim floor in this scene (scan for missed signs)"))
    n_hn = sum(1 for i in items if i.get("priority") == 2)
    doc = dict(created=time.strftime("%F %T"), counts=dict(positives=sum(1 for i in items if i.get("priority") == 0),
                                                           unreadable=sum(1 for i in items if i.get("priority") == 1), hard_negatives=n_hn,
                                                           missed_scan_frames=len(miss)),
               note="initial diagnostic priority, NOT a statistically sufficient sample. Context previews are JPEG, NOT encoder inputs.",
               items=items, missed_scan=miss)
    (out / "priority_bundle.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(json.dumps(doc["counts"]))


# ------------------------------------------------------------------------------------------------ UI
UI = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SEM1 Label Review</title><style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--acc:#2563eb;--line:#ddd}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#eee;--mut:#9a9a94;--card:#222220;--line:#3a3a37}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif}header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;flex-wrap:wrap;align-items:center}
main{padding:12px 16px}.clip{display:flex;gap:8px;overflow-x:auto}.clip figure{margin:0;min-width:240px}.clip img{width:240px;border-radius:6px}
.clip figcaption{font-size:12px;color:var(--mut)}.row{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}button{border:1px solid var(--line);background:var(--card);color:var(--fg);padding:6px 10px;border-radius:6px;cursor:pointer}
button.on{background:var(--acc);color:#fff}.k{font-size:11px;color:var(--mut)}input{padding:6px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
</style></head><body><header><b>SEM1 표지 라벨 검토</b><span id="pos"></span><button onclick="go(-1)">◀ 이전 (←)</button><button onclick="go(1)">다음 (→) ▶</button>
<button onclick="exp()">CSV 내보내기</button><span class="k">저장: 이 브라우저(localStorage) · human_verified 는 직접 체크</span></header><main id="m"></main>
<script>
const DATA=__DATA__;const VT=__VT__,RO=__RO__,RD=["READABLE","PARTIAL","UNREADABLE","NOT_TEXT"];let i=0;
let S={};try{S=JSON.parse(localStorage.getItem('sem1lab')||'{}')}catch(e){}
function save(){try{localStorage.setItem('sem1lab',JSON.stringify(S))}catch(e){}}
function cur(){const it=DATA[i];S[it.item]=S[it.item]||{hist:[]};return S[it.item]}
function set(k,v){const c=cur();c.hist.push({k,from:c[k]??null,to:v,at:new Date().toISOString()});c[k]=v;save();draw()}
function go(d){i=Math.max(0,Math.min(DATA.length-1,i+d));draw()}
function draw(){const it=DATA[i],c=cur();document.getElementById('pos').textContent=`${i+1}/${DATA.length} ${it.item} (v3 ${it.v3}, pri ${it.priority})`;
let h=`<div class="clip">`+(it.clip||[]).map(x=>`<figure><img src="${x.frame_img}"><figcaption>t=${x.t} sim=${x.sim}${x.best?' BEST':''}</figcaption><img src="${x.preview}"><figcaption>context PREVIEW (encoder 입력 아님)</figcaption></figure>`).join('')+`</div>`;
const grp=(name,arr,keys)=>`<div class="row"><b>${name}</b>`+arr.map((v,j)=>`<button class="${c[name]===v?'on':''}" onclick="set('${name}','${v}')">${v}<span class="k"> ${keys[j]||''}</span></button>`).join('')+`</div>`;
h+=grp('visual_type',VT,['1','2','3','4','5','6','7','8','9','0','u'])+grp('localization_role',RO,['q','w','e','r'])+grp('text_readability',RD,['a','s','d','f'])+grp('target_in_model_crop',['Y','N','UNSURE'],['y','n','?']);
h+=`<div class="row">physical_target_id <input id="pid" value="${c.physical_target_id||''}" onchange="set('physical_target_id',this.value)"> 메모 <input id="note" value="${c.confidence_note||''}" onchange="set('confidence_note',this.value)">
<label><input type="checkbox" ${c.human_verified?'checked':''} onchange="set('human_verified',this.checked)"> 내가 직접 확인함(human_verified)</label> reviewer <input value="${c.reviewer||''}" onchange="set('reviewer',this.value)"></div>`;
document.getElementById('m').innerHTML=h}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const k=e.key;const vi='1234567890u'.indexOf(k);if(vi>=0)return set('visual_type',VT[vi]);
const ri='qwer'.indexOf(k);if(ri>=0)return set('localization_role',RO[ri]);const di='asdf'.indexOf(k);if(di>=0)return set('text_readability',RD[di]);
if(k==='y')return set('target_in_model_crop','Y');if(k==='n')return set('target_in_model_crop','N');if(k==='?')return set('target_in_model_crop','UNSURE');
if(k==='ArrowRight')go(1);if(k==='ArrowLeft')go(-1)});
function exp(){const cols=['item','visual_type','localization_role','text_readability','target_in_model_crop','physical_target_id','confidence_note','reviewer','human_verified','reviewed_at','history'];
const rows=[cols.join(',')].concat(DATA.map(it=>{const c=S[it.item]||{};const last=(c.hist||[]).slice(-1)[0];return [it.item,c.visual_type||'',c.localization_role||'',c.text_readability||'',c.target_in_model_crop||'',c.physical_target_id||'',(c.confidence_note||'').replace(/,/g,' '),c.reviewer||'',c.human_verified?'true':'',last?last.at:'',JSON.stringify(c.hist||[]).replace(/"/g,'""')].map(v=>`"${v}"`).join(',')}));
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\n')],{type:'text/csv'}));a.download='sem1_labels_human.csv';a.click()}
draw();</script></body></html>"""


def cmd_ui(a):
    doc = json.loads(Path(a.bundle).read_text())
    html = UI.replace("__DATA__", json.dumps(doc["items"], ensure_ascii=False)).replace("__VT__", json.dumps(VISUAL)).replace("__RO__", json.dumps(ROLE))
    Path(a.out).write_text(html)
    print(f"ui -> {a.out} ({len(doc['items'])} items)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "labels", "priority", "ui"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle", default=None)
    a = ap.parse_args()
    dict(audit=cmd_audit, labels=cmd_labels, priority=cmd_priority, ui=cmd_ui)[a.cmd](a)


if __name__ == "__main__":
    main()
