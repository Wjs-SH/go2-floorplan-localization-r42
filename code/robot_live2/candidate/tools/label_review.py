#!/usr/bin/env python3
"""Label review package for user verification of the Claude-made spot labels (labels are NOT ground truth until reviewed).

  build : checks label <-> logged spot <-> source frame correspondence, groups near-duplicate spots into physical signs,
          renders one review image per labelled spot (full frame + zoom, best view) and writes a self-contained
          datasets/labels/review/review.html (open in any browser, no server). The page exports reviewed_labels.json/.csv.
  apply : merges a reviewed_labels.json into a NEW versioned label file under datasets/labels/v3/history/ with a change
          log (old -> new, reviewer, time). Existing label files are never modified.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

WS = Path.home() / "sci_ws"
sys.path.insert(0, str(WS / "tools"))
import rescore as RS  # noqa: E402
from metrics_v2 import clusters, spot_xy  # noqa: E402

CLASSES = ["POS_ROOM", "POS_NAME", "POS_FACILITY", "POS_ROOM_UNREADABLE", "NEG_POSTER", "NEG_BOARD", "NEG_DIRECTORY", "NEG_ELEVATOR",
           "NEG_GLARE", "NEG_LIGHT", "NEG_EXIT", "NEG_OTHER", "AMBIGUOUS"]
RUNS = {"S-1F": ("experiments/dev_S1F_v3", "datasets/labels/v3/labels_S-1F_v3.json", "dev"),
        "S-6F": ("experiments/test_S6F_v3", "datasets/labels/v3/labels_S-6F_v3.json", "seen-test")}


def build(a):
    out = WS / "datasets" / "labels" / "review"
    (out / "img").mkdir(parents=True, exist_ok=True)
    items, checks = [], {}
    for ds, (run, lf, split) in RUNS.items():
        sc, st, dsdir = RS.load(WS / run)
        L = json.loads((WS / lf).read_text())
        labels, notes = L["labels"], L.get("notes", {})
        rows = list(csv.DictReader(open(dsdir / "index.csv")))
        stamps = np.array([float(r["stamp"]) for r in rows])
        best = {}
        for s in sc:
            for r in s["regions"]:
                k = f"{ds}|spot{r['spot']}"
                if k in labels and (k not in best or r["sim"] > best[k][1]["sim"]):
                    best[k] = (s, r)
        xy = spot_xy(sc)
        ids = [int(k.split("spot")[1]) for k in labels if int(k.split("spot")[1]) in xy]
        phys = clusters(ids, xy, 1.5)
        missing = [k for k in labels if k not in best]
        frame_ok = 0
        for k, lab in sorted(labels.items(), key=lambda kv: best[kv[0]][0]["t"] if kv[0] in best else 1e9):
            if k not in best:
                continue
            s, r = best[k]
            i = int(np.argmin(np.abs(stamps - s["stamp"])))
            dt = abs(stamps[i] - s["stamp"])
            img = cv2.imread(str(dsdir / rows[i]["rgb"]))
            if img is None:
                continue
            frame_ok += 1
            x0, y0, x1, y1 = (int(round(v)) for v in (r.get("tile") or r["box"]))
            a_ = img.copy()
            cv2.rectangle(a_, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cx, cy, s_ = (x0 + x1) / 2, (y0 + y1) / 2, max(90, 2.4 * max(x1 - x0, y1 - y0))
            X0, Y0, X1, Y1 = int(max(0, cx - s_ / 2)), int(max(0, cy - s_ / 2)), int(min(640, cx + s_ / 2)), int(min(480, cy + s_ / 2))
            z = cv2.resize(img[Y0:Y1, X0:X1], (360, 360), interpolation=cv2.INTER_CUBIC)
            pic = np.hstack([cv2.resize(a_, (480, 360)), z])
            fn = f"img/{ds}_spot{r['spot']}.jpg"
            cv2.imwrite(str(out / fn), pic, [cv2.IMWRITE_JPEG_QUALITY, 85])
            sid = int(k.split("spot")[1])
            group = "ambiguous" if lab == "AMBIGUOUS" else ("clear_positive" if lab.startswith("POS") else "clear_negative")
            items.append(dict(key=k, dataset=ds, split=split, spot=sid, t=s["t"], frame=rows[i]["rgb"], frame_dt_s=round(float(dt), 3),
                              label=lab, note=notes.get(k, ""), group=group, physical_sign=f"{ds}#P{phys[sid]}" if sid in phys else None,
                              image=fn, sim=round(r["sim"], 3), z=None if r["z"] is None else round(r["z"], 2)))
        checks[ds] = dict(labels=len(labels), with_logged_spot=len(best), missing_spot=missing, frames_found=frame_ok,
                          max_frame_dt_s=max([it["frame_dt_s"] for it in items if it["dataset"] == ds] or [0]))
    (out / "review_items.json").write_text(json.dumps(dict(created=time.strftime("%F %T"), checks=checks, items=items), indent=1))
    page = ["<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
            "<title>E5 Label Review</title><style>body{font:14px system-ui;margin:16px;background:#fafafa;color:#111}"
            ".it{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px;margin:10px 0}.it img{max-width:100%}"
            ".hd{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.g_clear_positive{border-left:6px solid #2a9d8f}"
            ".g_clear_negative{border-left:6px solid #e76f51}.g_ambiguous{border-left:6px solid #e9c46a}"
            "button{font-size:15px;margin:4px}select,input{font-size:14px}.bar{position:sticky;top:0;background:#fafafa;padding:6px 0}</style></head><body>",
            "<h2>E5 표지 라벨 검토 (Claude 작성 라벨 — 사용자 검증 전)</h2>",
            "<p>빨간 상자 = SigLIP이 가장 표지답다고 본 조각(정지 시 조준점). 오른쪽은 확대. 상자 안의 대상 기준으로 분류하세요. "
            "같은 '물리 표지' 번호는 같은 실제 표지를 다른 위치에서 본 것입니다.</p>",
            f"<p>대응 검사: {html.escape(json.dumps(checks, ensure_ascii=False))}</p>",
            "<div class='bar'>검토자 <input id='who' value=''> <button onclick='exp(\"json\")'>JSON 내보내기</button><button onclick='exp(\"csv\")'>CSV 내보내기</button> "
            "<label><input type='checkbox' id='onlyamb' onchange='flt()'> 애매한 것만</label></div>"]
    for it in items:
        opts = "".join(f"<option{' selected' if c == it['label'] else ''}>{c}</option>" for c in CLASSES)
        page.append(f"<div class='it g_{it['group']}' data-key='{it['key']}' data-orig='{it['label']}' data-group='{it['group']}'>"
                    f"<div class='hd'><b>{it['key']}</b> <span>{it['split']}</span> <span>t={it['t']:.0f}s</span> <span>SigLIP {it['sim']}</span>"
                    f"<span>거리 {it['z']} m</span><span>물리 표지 {it['physical_sign']}</span><span>{html.escape(it['note'])}</span>"
                    f"<select>{opts}</select><input placeholder='메모' size=30></div><img loading='lazy' src='{it['image']}'></div>")
    page.append("""<script>
function rows(){return [...document.querySelectorAll('.it')].map(d=>({key:d.dataset.key,original:d.dataset.orig,
 reviewed:d.querySelector('select').value,note:d.querySelector('input').value}));}
function exp(kind){const who=document.getElementById('who').value||'unknown';const r=rows();let blob;
 if(kind=='json'){blob=new Blob([JSON.stringify({reviewer:who,time:new Date().toISOString(),items:r},null,1)],{type:'application/json'});}
 else{blob=new Blob(['key,original,reviewed,note\\n'+r.map(x=>[x.key,x.original,x.reviewed,'"'+x.note.replace(/"/g,'""')+'"'].join(',')).join('\\n')],{type:'text/csv'});}
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='reviewed_labels.'+kind;a.click();}
function flt(){const o=document.getElementById('onlyamb').checked;document.querySelectorAll('.it').forEach(d=>{d.style.display=(!o||d.dataset.group=='ambiguous')?'':'none';});}
</script></body></html>""")
    (out / "review.html").write_text("\n".join(page), encoding="utf-8")
    print(json.dumps(checks, indent=1))
    print(f"{len(items)} items -> {out / 'review.html'}")


def apply(a):
    rev = json.loads(Path(a.reviewed).read_text())
    hist = Path(a.hist_dir) if a.hist_dir else WS / "datasets" / "labels" / "v3" / "history"
    hist.mkdir(parents=True, exist_ok=True)
    for ds, (_, lf, _) in RUNS.items():
        cur = json.loads((WS / lf).read_text())["labels"]
        changes = [dict(key=x["key"], old=x["original"], new=x["reviewed"], note=x.get("note", "")) for x in rev["items"]
                   if x["key"].startswith(ds + "|") and x["reviewed"] != cur.get(x["key"])]
        new = dict(cur)
        for c in changes:
            new[c["key"]] = c["new"]
        n = len(list(hist.glob(f"labels_{ds}_rev*.json"))) + 1
        meta = dict(base=lf, reviewer=rev.get("reviewer"), reviewed_at=rev.get("time"), applied_at=time.strftime("%F %T"),
                    n_changes=len(changes), user_verified=True)
        (hist / f"labels_{ds}_rev{n}.json").write_text(json.dumps(dict(meta=meta, labels=new), indent=1, ensure_ascii=False))
        with open(hist / "CHANGELOG.jsonl", "a") as f:
            f.write(json.dumps(dict(meta, dataset=ds, revision=n, changes=changes), ensure_ascii=False) + "\n")
        print(ds, "revision", n, "changes", len(changes))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("build")
    b = sp.add_parser("apply"); b.add_argument("reviewed"); b.add_argument("--hist-dir", default=None)
    a = ap.parse_args()
    {"build": build, "apply": apply}[a.cmd](a)
