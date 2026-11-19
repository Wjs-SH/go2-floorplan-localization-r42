#!/usr/bin/env python3
"""Download official model files into ~/sci_ws/models and write a provenance JSON (repo, revision, sha256)."""
import hashlib, json, sys, time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path.home() / "sci_ws" / "models"
JOBS = {
    "siglip": ("google/siglip-base-patch16-224", "siglip-base-patch16-224",
               ["config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json",
                "spiece.model", "tokenizer.json", "tokenizer_config.json", "README.md"]),
    "qwen_gguf": ("Qwen/Qwen3-VL-2B-Instruct-GGUF", "Qwen3-VL-2B-Instruct-GGUF",
                  ["Qwen3VL-2B-Instruct-Q8_0.gguf", "mmproj-Qwen3VL-2B-Instruct-F16.gguf", "README.md"]),
    "qwen_hf": ("Qwen/Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Instruct", None),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def run(key):
    repo, sub, files = JOBS[key]
    api = HfApi()
    info = api.model_info(repo, files_metadata=True)
    rev = info.sha
    meta = {s.rfilename: s for s in info.siblings}
    if files is None:
        files = [f for f in meta if not f.startswith(".")]
    out = ROOT / sub
    out.mkdir(parents=True, exist_ok=True)
    rec = {"repo": repo, "revision": rev, "url": f"https://huggingface.co/{repo}", "downloaded": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": {}}
    for f in files:
        t0 = time.time()
        p = hf_hub_download(repo, f, revision=rev, local_dir=str(out))
        d = sha(p)
        lfs = getattr(meta.get(f), "lfs", None)
        ref = lfs.sha256 if lfs is not None else None
        ok = (ref is None) or (ref == d)
        rec["files"][f] = {"sha256": d, "hub_lfs_sha256": ref, "match": ok, "bytes": Path(p).stat().st_size, "sec": round(time.time() - t0, 1)}
        print(key, f, rec["files"][f], flush=True)
        if not ok:
            Path(p).rename(str(p) + ".HASH_MISMATCH")
    (out / "PROVENANCE.json").write_text(json.dumps(rec, indent=1))
    print("DONE", key, all(v["match"] for v in rec["files"].values()), flush=True)


if __name__ == "__main__":
    for k in sys.argv[1:]:
        run(k)
