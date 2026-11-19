#!/usr/bin/env python3
"""e5_runner_logged.py — OFF-BOARD (GPU computer) wrapper around the canonical event runner.

Runs ~/kci_event_handoff_20261009/code/live_event_runner.py UNCHANGED (imported, not edited) and additionally logs
every SigLIP request/response so the E5 packet contains the full score evidence (live mode of the canonical runner
does not write per-region scores; --save-regions exists only in replay mode).

Copy this file next to the bundle's code/ folder on the GPU computer and run from inside code/:

  E5_SIGLIP_PY=/path/to/python-with-siglip  E5_SIGLIP_LOG=<out>/siglip_log.jsonl \
  python3 ../e5_runner_logged.py live --go2 http://<robot>:8088 --aim --no-facility \
      --fac-map-sem ../maps/4F/4F_semantic_map.json --fac-map-base ../maps/4F/base_map.png --out <out>

Env:
  E5_SIGLIP_PY   python that has torch+transformers+the SigLIP model (replaces the hard-coded VLM_PY).
  E5_SIGLIP_LOG  jsonl path for the score log (default <cwd>/siglip_log.jsonl).
  E5_SAVE_FRAMES 1 -> also save every scored frame as JPEG next to the log (frames/<n>.jpg).
The SigLIP model path itself is still set inside region_siglip_rerank.py (MODEL); on the GPU computer it must point
to the existing local siglip-base-patch16-224 copy (field guide §5). No model is downloaded by this wrapper.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
import live_event_runner as LER  # noqa: E402  (canonical, unmodified)
import cv2  # noqa: E402

LOG = Path(os.environ.get("E5_SIGLIP_LOG", "siglip_log.jsonl"))
SAVE = os.environ.get("E5_SAVE_FRAMES", "1") == "1"
LOG.parent.mkdir(parents=True, exist_ok=True)
if SAVE:
    (LOG.parent / "frames").mkdir(exist_ok=True)

_py = os.environ.get("E5_SIGLIP_PY")
if _py:
    LER.VLM_PY = _py
_orig_init = LER.SiglipClient.__init__
_orig_score = LER.SiglipClient.score
_n = {"i": 0}


def _init(self, py=None):
    _orig_init(self, py or LER.VLM_PY)


def _score(self, bgr, regs):
    t_wall = time.time()
    sim, sec = _orig_score(self, bgr, regs)
    i = _n["i"]
    _n["i"] += 1
    frame = None
    if SAVE:
        frame = f"frames/{i:06d}.jpg"
        cv2.imwrite(str(LOG.parent / frame), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    rec = dict(i=i, t_wall=t_wall, sec=float(sec), frame=frame, frame_shape=list(bgr.shape),
               prompts=os.environ.get("KCI_SIGN_PROMPTS_JSON") or os.environ.get("KCI_SIGN_PROMPTS") or "roomstd(default)",
               regions=[[float(v) for v in r] for r in regs], sim=[float(v) for v in sim],
               adv=[float(v) for v in getattr(self, "last_adv", [])], board=[float(v) for v in getattr(self, "last_board", [])],
               ceil=[float(v) for v in getattr(self, "last_ceil", [])],
               tiles=[None if t is None else [float(v) for v in t] for t in getattr(self, "last_tiles", [])])
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return sim, sec


LER.SiglipClient.__init__ = _init
LER.SiglipClient.score = _score   # VlmClient defines its own __init__/ask and is not affected

if __name__ == "__main__":
    LER.main()
