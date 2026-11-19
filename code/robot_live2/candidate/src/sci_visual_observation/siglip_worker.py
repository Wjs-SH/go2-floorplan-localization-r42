#!/usr/bin/env python3
"""SCI SigLIP worker (GPU, persistent process, stdio pickle protocol identical to legacy live_siglip_worker.py).

Legacy fields (sim/adv/board/ceil/tiles) are produced by the UNCHANGED legacy RegionSiglip code so the A0 baseline is
bit-compatible. SCI adds hard-negative evidence used only by the SCI Observation Policy:
  neg[c][i]  : best-tile embedding of region i vs hard-negative class c (max over phrases), cosine
  ctx[c][i]  : 2.5x wide context crop of region i vs class c (poster/board/directory/elevator need context)
  ctx_plate  : wide crop vs "small plate next to a door" phrases (reference for ctx margins)
Costs: neg = one matmul on embeddings already computed; ctx = one extra forward per region (same as legacy board pass,
which it replaces when the legacy board pass is also requested).

Request : {"bgr": ndarray, "regions": [[x0,y0,x1,y1],...], "board": bool, "sci": bool}
Response: legacy keys + {"neg": {c: [...]}, "ctx": {c: [...]}, "ctx_plate": [...], "t_infer": float}
"""
import json, os, pickle, struct, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sci_legacy_e5"))
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402
from region_siglip_rerank import RegionSiglip  # noqa: E402

# Hard-negative vocabulary (SCI). Phrases are object names, no layout assumptions.
NEG = {
    "poster": ["a poster on a wall", "an advertisement poster", "a printed event announcement"],
    "board": ["a notice board with papers on a wall", "a bulletin board with many papers"],
    "glare": ["a bright reflection of light on a glossy surface", "sunlight glare on a wall", "a reflection on a glass window"],
    "light": ["a ceiling light", "a bright lamp on the ceiling"],
    "elevator": ["an elevator floor directory sign", "elevator call buttons", "an elevator door"],
    "directory": ["a building floor guide map", "a large directory board listing many rooms"],
    "other": ["a door", "a window", "a fire extinguisher cabinet", "an emergency exit sign", "a light switch on a wall"],
}
CTX_CLASSES = ("poster", "board", "directory", "elevator", "glare")
# E5-RT 2026-10-17: default 0.078 (= legacy sim_floor): a region below the floor can never enter legacy 'like' nor P2 VERIFIED,
#   so its context value is never used for a decision. Onboard replay (benchmark/siglip/run4 ctx_lazy, 30 field scenes):
#   context forward p50 117 -> 56 ms, crops 234 -> 99, board drift <= 1.8e-4, 0 board-gate flips. SCI_CTX_MIN=-1e9 restores all.
#   (older note: fp16 batch effects flipped 1/6 A0 stops on S-1F with this option -> keep it measurable via the env.)
CTX_MIN = float(os.environ.get("SCI_CTX_MIN", "0.078"))
VISION_ONLY = os.environ.get("SCI_E5RT_VISION_ONLY", "1") == "1"   # drop text tower after caching (GPU reserved 914 -> ~300 MB)
GPU_PREP = os.environ.get("SCI_E5RT_GPU_PREP", "1") == "1"         # uint8 crops, normalise on GPU (inputs bit-identical, 14 -> 5 ms)
PLATE = ["a door with a small room number sign", "a small sign next to a door", "a small plate on a wall"]
# SEM1: optional EXACT encoder-input dump (off by default). Saves, per request and up to a sample cap, the uint8 RGB
#   crops exactly as fed to normalisation (224x224, lossless PNG), the crop boxes in source-frame pixels, the source frame
#   sha256, and the sha256 of the actual fp16 normalised tensor. A JPEG review preview is NOT this.
DUMP_DIR = os.environ.get("SCI_SEM1_ENCODER_DUMP_DIR") or None
DUMP_MAX = int(os.environ.get("SCI_SEM1_ENCODER_DUMP_MAX", "200"))


class SciSiglip(RegionSiglip):
    def __init__(self):
        super().__init__()
        self.proc_text = None
        from transformers import AutoProcessor
        import region_siglip_rerank as R
        proc = AutoProcessor.from_pretrained(R.MODEL, local_files_only=True)
        self.neg_txt = {}
        with torch.no_grad():
            for c, ps in list(NEG.items()) + [("_plate", PLATE)]:
                t = proc(text=ps, return_tensors="pt", padding="max_length").to("cuda")
                e = self.model.get_text_features(**t)
                self.neg_txt[c] = torch.nn.functional.normalize(getattr(e, "pooler_output", e).float(), dim=-1)

    def _prep_u8(self, rgb, b):
        """Identical crop/resize to RegionSiglip._prep but returns uint8 (normalisation happens on the GPU)."""
        x0, y0, x1, y1 = (int(round(v)) for v in b)
        s = max(x1 - x0, y1 - y0, 32)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        height, width = rgb.shape[:2]
        c = rgb[int(max(0, cy - s / 2)):int(min(height, cy + s / 2)), int(max(0, cx - s / 2)):int(min(width, cx + s / 2))]
        if c.size == 0:
            c = np.zeros((8, 8, 3), np.uint8)
        return cv2.resize(c, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

    def _to_gpu(self, crops_u8=None, crops_f32=None):
        if crops_u8 is not None:
            if not hasattr(self, "_mean_t"):
                self._mean_t = torch.tensor(self.mean, device="cuda").view(1, 3, 1, 1)
                self._std_t = torch.tensor(self.std, device="cuda").view(1, 3, 1, 1)
            return torch.from_numpy(np.stack(crops_u8)).to("cuda").permute(0, 3, 1, 2).float().div_(255.0).sub_(self._mean_t).div_(self._std_t).half()
        return torch.from_numpy(np.stack(crops_f32)).permute(0, 3, 1, 2).to("cuda").half()

    def _events(self):
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def score_sci(self, bgr, regions, tile=100):
        """Same tiling/selection as RegionSiglip.score (legacy semantics) but keeps the best-tile embedding."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crops, owner, subs_all = [], [], []
        for k, b in enumerate(regions):
            nx = max(1, int(np.ceil((b[2] - b[0]) / tile))); ny = max(1, int(np.ceil((b[3] - b[1]) / tile)))
            xs, ys = np.linspace(b[0], b[2], nx + 1), np.linspace(b[1], b[3], ny + 1)
            for i in range(nx):
                for j in range(ny):
                    sb = [xs[i], ys[j], xs[i + 1], ys[j + 1]]
                    crops.append(self._prep_u8(rgb, sb) if GPU_PREP else self._prep(rgb, sb)); owner.append(k); subs_all.append(sb)
        x = self._to_gpu(crops_u8=crops) if GPU_PREP else self._to_gpu(crops_f32=crops)
        self.n_tiles = len(crops)
        self._last_tile_input = (crops, subs_all, owner, x)
        self.ev_tiles = self._events()
        self.ev_tiles[0].record()
        with torch.no_grad():
            e = self.model.get_image_features(pixel_values=x)
            en = torch.nn.functional.normalize(getattr(e, "pooler_output", e).float(), dim=-1)
            sim = (en @ self.txt.T).max(1).values.cpu().numpy()
            self._last_tile_sim = sim                                       # SEM1: per-tile raw cosine (dump only)
            conf = (en @ self.conf_txt.T).max(1).values.cpu().numpy()
            negs = {c: (en @ self.neg_txt[c].T).max(1).values.cpu().numpy() for c in NEG}
        out = np.full(len(regions), -1.0); best = [None] * len(regions); adv = np.zeros(len(regions)); bi = [0] * len(regions)
        for n, (o, s, sb, c) in enumerate(zip(owner, sim, subs_all, conf)):
            if s > out[o]:
                out[o], best[o], adv[o], bi[o] = s, [float(v) for v in sb], s - c, n
        self.best_tiles, self.best_adv = best, adv
        self.last_emb = [en[bi[i]].half().cpu().numpy().tobytes() for i in range(len(regions))]   # fp16 768-d, for offline analysis
        self.last_neg = {c: [float(negs[c][bi[i]]) for i in range(len(regions))] for c in NEG}
        self.ev_tiles[1].record()
        return out

    def context_scores(self, bgr, regions, k=2.5, smin=120):
        """Legacy board_scores crop geometry; returns legacy board/ceil arrays plus SCI ctx classes."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); crops = []
        for b in regions:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            s = max(max(b[2] - b[0], b[3] - b[1]) * k, smin)
            x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
            x1, y1 = int(min(rgb.shape[1], cx + s / 2)), int(min(rgb.shape[0], cy + s / 2))
            c = cv2.resize(rgb[y0:y1, x0:x1], (self.size, self.size), interpolation=cv2.INTER_AREA)
            crops.append(c if GPU_PREP else (c.astype(np.float32) / 255.0 - self.mean) / self.std)
        x = self._to_gpu(crops_u8=crops) if GPU_PREP else self._to_gpu(crops_f32=crops)
        self.ev_ctx = self._events()
        self.ev_ctx[0].record()
        with torch.no_grad():
            e = self.model.get_image_features(pixel_values=x)
            e = torch.nn.functional.normalize(getattr(e, "pooler_output", e).float(), dim=-1)
            plate = (e @ self.plate_txt.T).max(1).values
            board = ((e @ self.board_txt.T).max(1).values - plate).cpu().numpy()
            ceil = ((e @ self.ceil_txt.T).max(1).values - plate).cpu().numpy()
            ctx = {c: (e @ self.neg_txt[c].T).max(1).values.cpu().numpy().tolist() for c in CTX_CLASSES}
            pl = (e @ self.neg_txt["_plate"].T).max(1).values.cpu().numpy().tolist()
        self.ev_ctx[1].record()
        return board, ceil, ctx, pl


def read_msg(f):
    head = f.read(4)
    if len(head) < 4:
        return None
    return pickle.loads(f.read(struct.unpack("<I", head)[0]))


def write_msg(f, obj):
    b = pickle.dumps(obj, protocol=4); f.write(struct.pack("<I", len(b)) + b); f.flush()


def _dump(m, req, rep, state):
    """Write the exact encoder input of this request (tiles) into DUMP_DIR (bounded)."""
    import hashlib
    if not DUMP_DIR or state["n"] >= DUMP_MAX or not hasattr(m, "_last_tile_input"):
        return
    crops, boxes, owner, x = m._last_tile_input
    rid = req.get("dump_id") or f"req{state['req']:06d}"
    d = Path(DUMP_DIR) / rid
    d.mkdir(parents=True, exist_ok=True)
    src_sha = hashlib.sha256(np.ascontiguousarray(req["bgr"]).tobytes()).hexdigest()
    items = []
    for i, (c, b, o) in enumerate(zip(crops, boxes, owner)):
        if state["n"] >= DUMP_MAX:
            break
        arr = c if c.dtype == np.uint8 else None
        f = None
        if arr is not None:
            f = d / f"tile{i:03d}_r{o}.png"
            cv2.imwrite(str(f), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))       # lossless; stored RGB->BGR for PNG
        t_sha = hashlib.sha256(x[i].detach().cpu().numpy().tobytes()).hexdigest()
        items.append(dict(i=i, region=int(o), box_src=[float(v) for v in b], png=None if f is None else f.name,
                          rgb_uint8_sha256=None if arr is None else hashlib.sha256(arr.tobytes()).hexdigest(),
                          tensor_fp16_sha256=t_sha, tile_sim=float(m._last_tile_sim[i]) if hasattr(m, "_last_tile_sim") else None))
        state["n"] += 1
    meta = dict(request=rid, source_bgr_sha256=src_sha, source_shape=list(req["bgr"].shape), regions=[list(r) for r in req["regions"]],
                preprocess=dict(color="BGR->RGB (cv2.cvtColor)", crop="square s=max(w,h,32) around box centre, CLIPPED to frame (not padded)",
                                resize="cv2.INTER_LINEAR to 224x224", normalise="(x/255-mean)/std on GPU (fp16)" if GPU_PREP else "(x/255-mean)/std CPU",
                                mean=[float(v) for v in m.mean], std=[float(v) for v in m.std], size=int(m.size)),
                model=os.environ.get("SCI_SIGLIP_MODEL", "default ~/sci_ws/models/siglip-base-patch16-224"),
                prompts=state.get("prompts"), score="raw cosine, max over prompts; region sim = max over its tiles", tiles=items)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))


def main():
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr
    m = SciSiglip()
    m.score_sci(np.zeros((480, 640, 3), np.uint8), [[0, 0, 200, 200]])
    m.context_scores(np.zeros((480, 640, 3), np.uint8), [[0, 0, 200, 200]])
    if VISION_ONLY:                                  # text features are cached on the GPU; the tower is never used again
        import gc
        m.model.text_model = None
        gc.collect(); torch.cuda.empty_cache()
    import region_siglip_rerank as R
    pos = json.loads(os.environ["KCI_SIGN_PROMPTS_JSON"]) if os.environ.get("KCI_SIGN_PROMPTS_JSON") \
        else R.PROMPT_SETS[os.environ.get("KCI_SIGN_PROMPTS", "wall")]
    dump_state = dict(n=0, req=0, prompts=pos)
    write_msg(out, {"ready": True, "pid": os.getpid(), "prompts": pos, "neg": NEG, "model": R.MODEL,
                    "e5rt": dict(ctx_min=CTX_MIN, vision_only=VISION_ONLY, gpu_prep=GPU_PREP)})
    while True:
        req = read_msg(inp)
        if req is None:
            break
        if req.get("ping"):
            write_msg(out, {"pong": True}); continue
        t0 = time.time(); regs = req["regions"]
        rep = {"sim": [], "adv": [], "board": [], "ceil": [], "tiles": [], "neg": {}, "ctx": {}, "ctx_plate": []}
        if regs:
            sim = m.score_sci(req["bgr"], regs)
            rep.update(sim=[float(s) for s in sim], adv=[float(a) for a in m.best_adv], tiles=[list(b) for b in m.best_tiles], neg=m.last_neg,
                       emb=m.last_emb)
            if req.get("board") or req.get("sci"):
                # Lazy context pass (exact for legacy decisions): a region whose current SigLIP score is below the legacy
                # sim_floor can never enter legacy `like`, so its board value is never read; it gets the legacy default -1.
                ctx_min = float(req.get("ctx_min", CTX_MIN))
                sel = [i for i, s in enumerate(sim) if s >= ctx_min]
                board, ceil = [-1.0] * len(regs), [-1.0] * len(regs)
                ctx = {c: [None] * len(regs) for c in CTX_CLASSES}
                pl = [None] * len(regs)
                if sel:
                    b_, c_, x_, p_ = m.context_scores(req["bgr"], [regs[i] for i in sel])
                    for k, i in enumerate(sel):
                        board[i], ceil[i], pl[i] = float(b_[k]), float(c_[k]), p_[k]
                        for c in CTX_CLASSES:
                            ctx[c][i] = x_[c][k]
                if req.get("board"):
                    rep.update(board=board, ceil=ceil)
                rep.update(ctx=ctx, ctx_plate=pl, n_ctx=len(sel))
            torch.cuda.synchronize()
            rep["gpu_ms"] = dict(tiles=m.ev_tiles[0].elapsed_time(m.ev_tiles[1]),
                                 ctx=(m.ev_ctx[0].elapsed_time(m.ev_ctx[1]) if rep.get("n_ctx") else 0.0))
            rep["n_tiles"] = m.n_tiles
        rep["sec"] = time.time() - t0
        rep["t_infer"] = rep["sec"]
        if DUMP_DIR and regs:
            dump_state["req"] += 1
            try:
                _dump(m, req, rep, dump_state)
            except Exception as e:                       # a diagnostic dump never breaks scoring
                sys.stderr.write(f"encoder dump failed: {e!r}\n")
        write_msg(out, rep)


if __name__ == "__main__":
    main()
