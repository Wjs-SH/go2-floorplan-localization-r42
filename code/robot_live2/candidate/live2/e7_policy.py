"""e7_policy.py — which E7 paths may produce a FIX in THIS e34 process (2026-10-30, staging). File-only, read once at e34 start.

2026-10-30 21:xx: three modes. mode=OFF (env LIVE2_STRUCTURE_FIX=0: the path does not run), mode=SHADOW (DEFAULT: the
structure path runs and records what it WOULD decide, it never writes a FIX), mode=ENABLED (FIX allowed) — ENABLED only when:
  * <root>/config/e7_fix_policy.json exists with "structure_fix": "ENABLED" and a non-empty "verification_report"
  * its "code_sha256" equals the sha256 of the E7 decision code actually loaded (e6e7_link.py + e7_rematch.py + this file):
    any later code change disables the path again until it is re-verified
  * env LIVE2_STRUCTURE_FIX is not "0" (the env can only switch the path OFF; "1" never enables it)
There is no default-on: a missing / unreadable / mismatching file, a reboot or a console restart all leave FIX output off
(SHADOW). `enabled` below always means "may output a FIX" (== mode ENABLED).
SEMANTIC (PATH B: E6 reading -> E8 pair -> E7) is unchanged and reported here only so the tablet shows both scopes.
GT is never read."""
import hashlib
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                   # <root>/candidate/live2/e7_policy.py -> <root>
POLICY_FILE = ROOT / "config" / "e7_fix_policy.json"
CODE_FILES = ("e6e7_link.py", "e7_rematch.py", "e7_policy.py", "e7_retro.py")


def code_sha256(files=CODE_FILES, here=HERE):
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode() + b"\0" + (here / f).read_bytes() + b"\0")
    return h.hexdigest()


def structure_policy(policy_file=None, env=None):
    """-> dict(enabled, why, code_sha256, policy_file, env). Never raises: any doubt -> enabled False."""
    pf = Path(policy_file) if policy_file else POLICY_FILE
    env = os.environ if env is None else env
    out = dict(enabled=False, mode="SHADOW", policy_file=str(pf), env=env.get("LIVE2_STRUCTURE_FIX"), code_sha256=None,
               verification_report=None)
    try:
        out["code_sha256"] = code_sha256()
    except OSError as ex:
        return dict(out, mode="OFF", why=f"E7 code unreadable ({type(ex).__name__})")
    if env.get("LIVE2_STRUCTURE_FIX") == "0":
        return dict(out, mode="OFF", why="LIVE2_STRUCTURE_FIX=0 (forced off, not even shadow)")
    try:
        cfg = json.loads(pf.read_text())
    except FileNotFoundError:
        return dict(out, why="no verified enable file (default SHADOW: verify and record, never FIX)")
    except (OSError, ValueError) as ex:
        return dict(out, why=f"enable file unreadable ({type(ex).__name__}) -> OFF")
    if not isinstance(cfg, dict) or cfg.get("structure_fix") != "ENABLED":
        return dict(out, why=f"enable file says {cfg.get('structure_fix') if isinstance(cfg, dict) else '?'} -> OFF")
    if not cfg.get("verification_report"):
        return dict(out, why="enable file has no verification_report -> OFF")
    out["verification_report"] = cfg.get("verification_report")
    if cfg.get("code_sha256") != out["code_sha256"]:
        return dict(out, why="E7 code changed since the enable file was verified -> OFF")
    return dict(out, enabled=True, mode="ENABLED", why=f"verified enable file ({cfg.get('verification_report')})")


SEMANTIC_POLICY_FILE = ROOT / "config" / "e7_semantic_policy.json"


def semantic_policy(policy_file=None, env=None):
    """R37: semantic 2-room FIX. Default SHADOW; ENABLED only by a file {"semantic_fix": "ENABLED", "verification_report": ...}; env LIVE2_SEMANTIC_FIX=0 always switches it off. Read at each decision."""
    pf = Path(policy_file) if policy_file else SEMANTIC_POLICY_FILE
    env = os.environ if env is None else env
    if env.get("LIVE2_SEMANTIC_FIX") == "0":
        return dict(mode="SHADOW", why="LIVE2_SEMANTIC_FIX=0")
    try:
        cfg = json.loads(pf.read_text())
    except (OSError, ValueError):
        return dict(mode="SHADOW", why="no enable file")
    if isinstance(cfg, dict) and cfg.get("semantic_fix") == "ENABLED" and cfg.get("verification_report"):
        return dict(mode="ENABLED", why=f"enable file ({cfg.get('verification_report')})")
    return dict(mode="SHADOW", why="enable file not valid")


def long_structure_policy(policy_file=None, env=None):
    """R38: whole-path structure FIX (final_verify). Separate from the old 30 s window path, which stays SHADOW. Same enable file, key structure_long_fix; env LIVE2_STRUCTURE_LONG_FIX=0 switches it off."""
    pf = Path(policy_file) if policy_file else SEMANTIC_POLICY_FILE
    env = os.environ if env is None else env
    if env.get("LIVE2_STRUCTURE_LONG_FIX") == "0":
        return dict(mode="SHADOW", why="LIVE2_STRUCTURE_LONG_FIX=0")
    try:
        cfg = json.loads(pf.read_text())
    except (OSError, ValueError):
        return dict(mode="SHADOW", why="no enable file")
    if isinstance(cfg, dict) and cfg.get("structure_long_fix") == "ENABLED" and cfg.get("verification_report"):
        return dict(mode="ENABLED", why=f"enable file ({cfg.get('verification_report')})")
    return dict(mode="SHADOW", why="enable file does not enable it")


def write_report(state_dir, structure, pid=None):
    """<state>/e7_policy.json: what THIS e34 process applies (pid + start time, so the tablet can tell a stale file)."""
    rec = dict(kind="LIVE2_E7_FIX_POLICY", pid=pid or os.getpid(), started_wall=time.time(),
               structure=dict(enabled=bool(structure.get("enabled")), mode=structure.get("mode") or ("ENABLED" if structure.get("enabled") else "OFF"),
                              why=structure.get("why"), env=structure.get("env"),
                              code_sha256=structure.get("code_sha256"), verification_report=structure.get("verification_report")),
               semantic=dict(enabled=True, scope="E6 decisive reading consistent + E8 pair (2 independent plates) + E7 geometry unique "
                                                  "vs E3 + E2 rivals + FAST-LIO >= 8 Hz; E6 hardware needs approval B"))
    state_dir = Path(state_dir)
    tmp = state_dir / f".e7_policy.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(rec, default=str))
    os.replace(tmp, state_dir / "e7_policy.json")
    return rec
