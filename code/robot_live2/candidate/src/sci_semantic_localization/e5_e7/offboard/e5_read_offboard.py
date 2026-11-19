#!/usr/bin/env python3
"""e5_read_offboard.py — OFF-BOARD (GPU computer) semantic reading of the pan/tilt capture (E5-C).

The event bundle has NO OCR (audit + code read). This reader uses the SAME local Qwen3-VL model and the same
environment the bundle already uses (live_vlm_worker.py: KCI_VLM_MODEL, local_files_only=True) with a dedicated
room-number question. No model is downloaded. Raw model text and token log-probabilities are saved verbatim.

  KCI_VLM_MODEL=/local/Qwen3-VL-2B-Instruct  python3 e5_read_offboard.py --robot http://<robot>:8088 --out <dir>
  (or --image <file.jpg> for an already-copied capture)

Output <out>/e5_reading.json:
  {engine, model, prompt, image, image_sha256, raw_answer, token_logprob_sum, seq_prob, label_alternatives:[{label,p}],
   t_request, t_answer, sec}
"""
import argparse
import hashlib
import json
import math
import os
import re
import time
import urllib.request
from pathlib import Path

PROMPT = ("This photo shows a room sign on a building wall. Read the room number printed on the sign. "
          "Answer with only the room number (for example 401 or B102). If no room number is readable, answer none.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=None, help="aim console base URL; fetches /full.jpg")
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_req = time.time()
    if a.image:
        img_bytes = Path(a.image).read_bytes()
        src = a.image
    else:
        with urllib.request.urlopen(a.robot.rstrip("/") + "/full.jpg", timeout=10) as r:
            img_bytes = r.read()
        src = a.robot.rstrip("/") + "/full.jpg"
        (out / "e5_capture_full.jpg").write_bytes(img_bytes)
    import io

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_dir = os.environ.get("KCI_VLM_MODEL", "/data3/kci_models/Qwen3-VL-2B-Instruct")
    proc = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16,
                                                        device_map="cuda").eval()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = proc.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": img},
                                                               {"type": "text", "text": PROMPT}]}],
                                 tokenize=True, add_generation_prompt=True, return_dict=True,
                                 return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        o = model.generate(**x, max_new_tokens=12, do_sample=False, output_scores=True, return_dict_in_generate=True)
    sec = time.time() - t0
    gen = o.sequences[:, x["input_ids"].shape[1]:]
    ans = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
    lp = 0.0
    for step, sc in enumerate(o.scores):
        tok = int(gen[0, step])
        lp += float(torch.log_softmax(sc[0].float(), -1)[tok])
    p = math.exp(lp)
    m = re.findall(r"[A-Za-z]?\d{2,4}", ans)
    alts = [dict(label=m[0].upper(), p=p), dict(label="__OTHER__", p=1 - p)] if m else [dict(label="__NONE__", p=1.0)]
    rec = dict(engine="Qwen3-VL (bundle live_vlm_worker model), greedy", ocr_engine="NONE_INSTALLED",
               model=model_dir, prompt=PROMPT, image=src, image_sha256=hashlib.sha256(img_bytes).hexdigest(),
               raw_answer=ans, token_logprob_sum=lp, seq_prob=p, label_alternatives=alts,
               t_request=t_req, t_answer=time.time(), sec=sec)
    (out / "e5_reading.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
