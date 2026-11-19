#!/usr/bin/env python3
"""Sign-vs-not discrimination study on logged SigLIP tile embeddings (fp16, 768-d) — development split only.

  1) zero-shot: AUC of (max positive prompt - max negative prompt) for candidate prompt sets
  2) linear probe: L2 logistic regression on embeddings, leave-one-group-out CV where a group = physical sign cluster
     (spots within 1.5 m), observation-level and spot-level (max over observations) AUC
Writes models/e5_probe/probe_<name>.npz (weights + bias + training manifest) when --save is given.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

WS = Path.home() / "sci_ws"
sys.path.insert(0, str(WS / "src" / "sci_legacy_e5"))


def load_obs(run, labels, ds_name, min_sim=-1.0):
    X, y, g, sid, sims, ts = [], [], [], [], [], []
    for l in open(Path(run) / "scenes.jsonl"):
        s = json.loads(l)
        for r in s["regions"]:
            k = f"{ds_name}|spot{r['spot']}"
            if r.get("emb") is None or k not in labels or labels[k] == "AMBIGUOUS" or r["sim"] < min_sim:
                continue
            X.append(np.frombuffer(base64.b64decode(r["emb"]), np.float16).astype(np.float32))
            y.append(1 if labels[k].startswith("POS") else 0)
            sid.append(k)
            g.append(r["xyz"][:2] if r["xyz"] else [1e6, 1e6])
            sims.append(r["sim"])
            ts.append(s["t"])
    return np.array(X), np.array(y), np.array(g), np.array(sid), np.array(sims), np.array(ts)


def clusters(sid, g, radius=1.5):
    """Group spots into physical-sign clusters by their mean xy (avoid leakage between near-duplicate spots)."""
    keys = sorted(set(sid))
    cen = {k: g[sid == k].mean(0) for k in keys}
    lab, c = {}, 0
    for k in keys:
        if k in lab:
            continue
        stack = [k]
        lab[k] = c
        while stack:
            a = stack.pop()
            for b in keys:
                if b not in lab and np.linalg.norm(cen[a] - cen[b]) < radius:
                    lab[b] = c
                    stack.append(b)
        c += 1
    return np.array([lab[k] for k in sid])


def auc(y, s):
    """Mann-Whitney AUC (ties count 1/2)."""
    y, s = np.asarray(y), np.asarray(s, float)
    if len(set(y.tolist())) < 2:
        return None
    pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


class LogReg:
    """L2-regularised, class-balanced logistic regression (LBFGS, torch, CPU). C as in sklearn (penalty 1/(2C)|w|^2)."""

    def __init__(self, C=0.1):
        self.C = C

    def fit(self, X, y):
        X, y = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
        w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        npos, nneg = float(y.sum()), float(len(y) - y.sum())
        sw = torch.where(y > 0, len(y) / (2 * npos), len(y) / (2 * nneg))
        opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = X @ w + b
            loss = (sw * torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")).sum() + 0.5 / self.C * (w @ w)
            loss.backward()
            return loss
        opt.step(closure)
        self.coef_, self.intercept_ = w.detach().numpy()[None, :], b.detach().numpy()
        return self

    def predict_proba(self, X):
        z = X @ self.coef_[0] + self.intercept_[0]
        p = 1 / (1 + np.exp(-z))
        return np.stack([1 - p, p], 1)


def spot_auc(y, s, sid):
    keys = sorted(set(sid))
    yy = [int(y[sid == k][0]) for k in keys]
    ss = [float(s[sid == k].max()) for k in keys]
    return auc(np.array(yy), np.array(ss)), len(keys)


def text_emb(prompts):
    from transformers import AutoModel, AutoProcessor
    P = WS / "models" / "siglip-base-patch16-224"
    proc = AutoProcessor.from_pretrained(str(P), local_files_only=True)
    m = AutoModel.from_pretrained(str(P), local_files_only=True).eval()
    with torch.no_grad():
        t = proc(text=prompts, return_tensors="pt", padding="max_length")
        e = m.get_text_features(**t)
    return torch.nn.functional.normalize(e.float(), dim=-1).numpy()


PROMPTS = {
    "legacy_roomstd": (["a small room name plate", "a small room name plate near a door", "a hallway sign perpendicular to the wall"], None),
    "plate_vs_neg": (["a small room name plate", "a small room name plate near a door", "a hallway sign perpendicular to the wall"],
                     ["a poster on a wall", "a notice board with papers on a wall", "a light switch on a wall", "a door",
                      "a glass door", "a fire alarm button", "a window", "a bright reflection of light"]),
    "text_sign": (["a sign with text", "a room number sign", "a blue name plate with white text", "a nameplate with a room number"],
                  ["a poster on a wall", "a notice board with papers on a wall", "a light switch on a wall", "a door",
                   "a glass door", "a fire alarm button", "a window", "a bright reflection of light"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(WS / "experiments" / "dev_S1F_v2"))
    ap.add_argument("--labels", default=str(WS / "datasets" / "labels" / "labels_flat.json"))
    ap.add_argument("--ds", default="S-1F")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    labels = json.loads(Path(a.labels).read_text())
    X, y, g, sid, sims, ts = load_obs(a.run, labels, a.ds)
    grp = clusters(sid, g)
    out = dict(n_obs=int(len(y)), n_pos_obs=int(y.sum()), n_spots=len(set(sid)), n_groups=int(len(set(grp))))
    out["siglip_raw"] = dict(obs_auc=auc(y, sims), spot_auc=spot_auc(y, sims, sid)[0])
    for name, (pos, neg) in PROMPTS.items():
        tp = text_emb(pos)
        sp = (X @ tp.T).max(1)
        sc = sp - (X @ text_emb(neg).T).max(1) if neg else sp
        out[f"zs_{name}"] = dict(obs_auc=auc(y, sc), spot_auc=spot_auc(y, sc, sid)[0])
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    pred = np.zeros(len(y))
    for gi in sorted(set(grp)):
        tr, te = grp != gi, grp == gi
        if len(set(y[tr])) < 2:
            pred[te] = 0.5
            continue
        clf = LogReg(C=a.C).fit(Xn[tr], y[tr])
        pred[te] = clf.predict_proba(Xn[te])[:, 1]
    out["probe_logo_cv"] = dict(C=a.C, obs_auc=auc(y, pred), spot_auc=spot_auc(y, pred, sid)[0])
    print(json.dumps(out, indent=1))
    if a.save:
        clf = LogReg(C=a.C).fit(Xn, y)
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.save, w=clf.coef_[0].astype(np.float32), b=np.float32(clf.intercept_[0]),
                 manifest=json.dumps(dict(trained_on=a.run, labels=a.labels, ds=a.ds, C=a.C, **{k: v for k, v in out.items() if k.startswith("n_")})))
        print("saved", a.save)


if __name__ == "__main__":
    main()
