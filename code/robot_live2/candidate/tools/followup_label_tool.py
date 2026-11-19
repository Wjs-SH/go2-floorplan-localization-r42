#!/usr/bin/env python3
"""SEM1 follow-up: build the offline label + bbox review tool from REAL robot files (CPU only, no model).

build  : for the 76 SEM1 priority items (26 positive / 2 unreadable / 48 other-provisional) + the 107 "missed sign scan"
         frames, copy the ORIGINAL dataset frames (not previews) into review/frames/, hash them (sha256 read from the
         copied file), record width/height, dataset/run identity (run dir + scenes.jsonl sha256), model / prompts /
         preprocessing code hashes, and attach EXACT encoder inputs only where a SEM1 dump exists for exactly that
         frame (source sha256 equal + tile PNG sha256 verified). Everything else: encoder_input = NOT_AVAILABLE.
         Writes review/LABEL_AND_BOX_UI.html (data embedded) + review/ui_items.json.
Nothing here fills human fields; human_verified is never set by a tool.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "sci_visual_observation"))
from e5rt.data_audit import verify_encoder_input  # noqa: E402

SEM1 = Path("/home/unitree/sci_ws/repairs/e5sem1_20261018_025424")
EXP = Path("/home/unitree/sci_ws/experiments")
DATA = {"S-1F": Path("/home/unitree/sci_ws/datasets/bags/S-1F"), "S-6F": Path("/home/unitree/sci_ws/datasets/bags/S-6F")}
LABELS = {"S-1F": Path("/home/unitree/sci_ws/datasets/labels/v3/labels_S-1F_v3.json"),
          "S-6F": Path("/home/unitree/sci_ws/datasets/labels/v3/labels_S-6F_v3.json")}
MODEL = Path("/home/unitree/sci_ws/models/siglip-base-patch16-224/model.safetensors")
LABEL_REVISION = "human-r1"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def provenance(cand):
    pre = [cand / "src/sci_visual_observation/siglip_worker.py", cand / "src/sci_legacy_e5/region_siglip_rerank.py"]
    runs = {}
    for fl, lp in LABELS.items():
        lab = json.loads(lp.read_text())
        run = EXP / Path(lab["run"]).name
        pj = run / "siglip_prompts.json"
        runs[fl] = dict(run_id=run.name, run_path=str(run), scenes_sha256=sha(run / "scenes.jsonl"),
                        prompts=json.loads(pj.read_text()) if pj.exists() else None,
                        prompts_sha256=sha(pj) if pj.exists() else None, labels_file=str(lp), labels_sha256=sha(lp),
                        labels_run=lab.get("run"), labeller=lab.get("labeller"))
    return dict(model=str(MODEL), model_sha256=sha(MODEL), preprocess_code_sha256={p.name: sha(p) for p in pre}, runs=runs,
                label_revision=LABEL_REVISION, note="all hashes read from the actual files on the robot at build time")


def copy_frame(out, fl, rgb_rel):
    src = DATA[fl] / rgb_rel
    dst = out / "frames" / fl / Path(rgb_rel).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copyfile(src, dst)                    # a real copy (never a hardlink to the dataset)
    import cv2
    im = cv2.imread(str(dst))
    return dict(frame_path=str(dst.relative_to(out)), frame_abs=str(dst), source_dataset_path=str(src), frame_sha256=sha(dst),
                source_sha256=sha(src), width=int(im.shape[1]), height=int(im.shape[0]))


def encoder_inputs(out, item, frame_sha):
    d = SEM1 / "analysis" / "roi" / "encoder_input" / item.replace("|", "_")
    if not (d / "meta.json").exists():
        return dict(status="NOT_AVAILABLE", why="no SEM1 dump for this item (regenerate with sem1_crop_lab --dump under supervisor)")
    meta = json.loads((d / "meta.json").read_text())
    if meta.get("source_sha256") != frame_sha:
        return dict(status="NOT_AVAILABLE", why="SEM1 dump was made on a different frame of this item")
    tiles = []
    for t in meta["tiles"]:
        png = d / f"tile{t['i']:02d}_224_rgb_as_png.png"
        ok, why = verify_encoder_input(png, t)
        if not ok:
            return dict(status="NOT_AVAILABLE", why=f"dump verification failed: {why}")
        dst = out / "encoder_input" / item.replace("|", "_") / png.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(png, dst)
        tiles.append(dict(png=str(dst.relative_to(out)), box_src=t["box_src"], rgb_uint8_sha256=t["rgb_uint8_sha256"]))
    return dict(status="EXACT_ENCODER_INPUT_VERIFIED", tiles=tiles, region_box=meta["region_box"], prompts=meta.get("prompts"))


def build(a):
    out = Path(a.out)
    cand = Path(a.candidate)
    bundle = json.loads((SEM1 / "review" / "priority_bundle.json").read_text())
    items = []
    grp = {0: "POSITIVE_PROVISIONAL", 1: "UNREADABLE_PROVISIONAL", 2: "OTHER_PROVISIONAL_NEG_OR_AMBIGUOUS"}
    for it in bundle["items"]:
        fl = it.get("floor") or it["item"].split("|")[0]
        frames = []
        for c in it.get("clip", []):
            fr = copy_frame(out, fl, c["frame_rgb"])
            role = "BEST" if c.get("best") else ("FIRST" if c is it["clip"][0] else ("LAST" if c is it["clip"][-1] else "MIDDLE"))
            fr.update(frame_id=f"{fl}:{c['frame_index']}", frame_index=c["frame_index"], t_scene=c["t"], clip_role=role,
                      pipeline_tile=c["tile"], model_sim=c["sim"], preview=str(Path("previews") / Path(c["preview"]).name))
            pv = SEM1 / "review" / c["preview"]
            pdst = out / fr["preview"]
            pdst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pv, pdst)
            fr["encoder_input"] = encoder_inputs(out, it["item"], fr["frame_sha256"])
            frames.append(fr)
        items.append(dict(item=it["item"], group=grp[it["priority"]], dataset_id=fl, split="DEV" if fl == "S-1F" else "SEEN_REGRESSION",
                          run_id=json.loads(LABELS[fl].read_text())["run"].split("/")[-1], spot=it.get("spot"),
                          hint=dict(v3=it.get("v3"), best_sim=it.get("best_sim")), frames=frames))
    # missed-sign scan: original frames where the pipeline had no region >= floor
    import numpy as np
    for m in bundle["missed_scan"]:
        fl = m["floor"]
        rows = list(csv.DictReader(open(DATA[fl] / "index.csv")))
        st = np.array([float(r["stamp"]) for r in rows])
        k = int(np.argmin(np.abs(st - float(m["stamp"]))))
        fr = copy_frame(out, fl, rows[k]["rgb"])
        fr.update(frame_id=f"{fl}:{k}", frame_index=k, t_scene=m["t"], clip_role="SCAN", pipeline_tile=None, model_sim=None,
                  preview=None, encoder_input=dict(status="NOT_AVAILABLE", why="no candidate on this frame"))
        items.append(dict(item=f"MISSED_SCAN|{fl}|t{m['t']:.1f}", group="MISSED_SIGN_SCAN", dataset_id=fl,
                          split="DEV" if fl == "S-1F" else "SEEN_REGRESSION", run_id=json.loads(LABELS[fl].read_text())["run"].split("/")[-1],
                          spot=None, hint=dict(v3=None, best_sim=None), frames=[fr]))
    prov = provenance(cand)
    doc = dict(created=time.strftime("%F %T %Z"), label_revision=LABEL_REVISION, provenance=prov, items=items,
               counts={g: sum(1 for i in items if i["group"] == g) for g in sorted({i["group"] for i in items})})
    (out / "ui_items.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    tpl = (HERE / "followup_label_ui_template.html").read_text()
    html = tpl.replace("__DATA__", json.dumps(doc, ensure_ascii=False))
    (out / "LABEL_AND_BOX_UI.html").write_text(html)
    (out / "human_exports").mkdir(exist_ok=True)
    n_exact = sum(1 for i in items for f in i["frames"] if f["encoder_input"]["status"].startswith("EXACT"))
    print(json.dumps(dict(items=len(items), counts=doc["counts"], frames=sum(len(i["frames"]) for i in items), exact_encoder_frames=n_exact,
                          ui=str(out / "LABEL_AND_BOX_UI.html"))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidate", required=True)
    a = ap.parse_args()
    build(a)


if __name__ == "__main__":
    main()
