"""R34: SEMANTIC_ANCHORED end to end on the REAL loc2 store: REAL E6E7Link._anchor_start/_anchor_job -> semantic_anchor.json -> REAL E3 Tracker.intake_anchored -> real keyframes -> E8. No stub of the channel. OFFLINE_COUNTERFACTUAL. Run: test_anchor_wiring_real.py OUT"""
import json
import shutil
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
STG = HERE.parent / "candidate/live2"
sys.path.insert(0, str(STG))
import e3_tracker as E3  # noqa: E402
import e6_semantic as ES  # noqa: E402
import e6e7_link as L  # noqa: E402
import e8_pair as EP  # noqa: E402

OPR = Path.home() / "sci_ws/repairs/livefield2_20261018_151606"
X = OPR / "experiments/X20261101_193139"


def main(out):
    out = Path(out); shutil.rmtree(out, ignore_errors=True); (out / "e2").mkdir(parents=True)
    res = dict(kind="OFFLINE_COUNTERFACTUAL", checks=[], ok=True)

    def ck(n, ok, **k):
        res["checks"].append(dict(name=n, ok=bool(ok), **k)); res["ok"] &= bool(ok); print(("PASS " if ok else "FAIL ") + n, str(k)[:220], flush=True)
    t0 = json.loads((X / "experiment.json").read_text())["created_wall"]
    O3 = next(json.loads(l) for l in open(X / "state/e8_observations.jsonl") if json.loads(l).get("event") == "STORED" and json.loads(l)["observation_id"].endswith("O003"))
    fd = [json.loads(l) for l in open(X / "state/e2/e2_passes.jsonl") if '"floor_done"' in l][:9]
    sess = json.loads((X / "archive/live2_store/session.json").read_text())["odom_session_id"]
    pf = {str(p["floor"]): dict(status="DONE", H=1825, map_version=p["candidates"][0]["map_version"], light=str(OPR / f"runtime/e2_cache/light/kci_{p['floor']}F_{p['candidates'][0]['map_version']}")) for p in fd}
    (out / "e2/e2_state.json").write_text(json.dumps(dict(status="IDLE", pass_id=1, odom_session=sess, per_floor=pf, floors_done=[p["floor"] for p in fd], floors_total=9, candidates=[dict(c, H=1825, odom_session=sess) for p in fd for c in p["candidates"]],
                                                          candidates_pass=[], candidates_coarse=[])))
    tr = E3.Tracker(str(X / "archive/live2_store"), out / "e2", out / "e3.jsonl")
    tr.check_session()
    tr.store.refresh()
    sf = ES.SemanticFloors()
    lk = L.E6E7Link.__new__(L.E6E7Link)
    lk.state, lk.tr, lk.sf, lk.last = out, tr, sf, None
    r = dict(observation_id=O3["observation_id"], usable=True, decisive=True, floor_likelihood={"4": 0.98, "5": 0.002}, labels=[["408", 0.68]])
    pm = types.SimpleNamespace(group=[dict(O3, capture_wall=t0 + 521.0, image_sha256="sha")], gid="G", sf=sf, single_obs_region=lambda r_, rows=None: EP.PairManager.single_obs_region(pm, r_, rows))
    lk._anchor_start(pm, r)
    t = time.time()
    while lk._anchor_thread.is_alive() and time.time() - t < 200:
        time.sleep(0.2)
    doc = json.loads((out / "e2/semantic_anchor.json").read_text())
    ck("anchored fit ran through the link", doc.get("status") == "ANCHORED_PROVISIONAL" and doc["candidates"], status=doc.get("status"), runs=[{k: v for k, v in x.items() if k in ("status", "why", "sec", "n_frames", "room")} for x in doc["runs"]])
    c0 = doc["candidates"][0]
    ck("tagged with its evidence and PROVISIONAL", c0["provenance"] == "SEMANTIC_ANCHORED" and c0["provisional"] and c0["model_status"] == "UNVERIFIED_ANCHOR_MODEL" and c0["evidence"]["observation_id"] == O3["observation_id"] and c0["evidence"]["map_version"],
       evidence=c0["evidence"])
    n0 = len(tr.H)
    added = tr.intake_anchored()
    hs = [h for h in tr.H if h.rec.get("provenance") == "SEMANTIC_ANCHORED"]
    ck("REAL E3 takes it as a hypothesis next to E2's", added >= 1 and hs and len(tr.H) >= n0, added=added, ids=[h.id for h in hs])
    s = hs[0].summary()
    ck("the summary carries the provenance", s["provenance"] == "SEMANTIC_ANCHORED" and s["anchor_evidence"]["observation_id"] == O3["observation_id"])
    ck("intake is idempotent", tr.intake_anchored() == 0)
    tr2 = E3.Tracker(str(X / "archive/live2_store"), out / "e2", out / "e3b.jsonl"); tr2.check_session(); tr2.session = "OTHER"
    ck("another odometry session takes nothing", tr2.intake_anchored() == 0 and not tr2.H)
    hy = [dict(id=h.id, floor=h.floor, rec=h.rec) for h in hs]
    o1 = dict(O3, ocr_lines=[["408", 0.997]], vlm=None, read=True, image_quality=dict(ok=True, flags=[]))
    ev = EP.evaluate_group([o1], hy, sf, None, None)
    ck("one reading alone: E8 says a second plate is needed (no fix path)", ev["verdict"] == "MORE_OBSERVATION_REQUIRED" and ev["n_valid_independent"] == 1)
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    print("ALL_OK" if res["ok"] else "SOME_FAILED", flush=True)
    return res["ok"]


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
