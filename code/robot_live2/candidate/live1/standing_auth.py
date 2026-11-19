"""standing_auth.py — STANDING_RESEARCH_AUTHORIZATION (development standing permission), stdlib only.

What it replaces: the per-boot, 60-minute approval A record that had to be re-issued through Claude Code for every
session. What it does NOT replace: per-boot runtime checks (PIDs, ROS graph, odometry, sensors, resources) — those are
evaluated fresh by ctl gates at every [실험 시작]; nothing from a previous boot is reused.

Files (runtime/ctl/auth/):
  standing.key            32-byte HMAC key, 0600, created by `ctl standing-grant` (CLI on the robot only)
  standing_research.json  the grant: modules, per-module limits, device, user text + sha256, created, grant_id, hmac
  standing_state.json     ACTIVE / SUSPENDED / REVOKED for that grant_id (+ who / when), hmac
The web can only SUSPEND, RESUME (a suspended, not revoked grant) and REVOKE. It can never create a grant, widen the module
list or raise a limit: those exist only in standing_research.json, whose HMAC is checked on every read; a hand-edited or
web-written grant without the key-made HMAC is INVALID. A revoked grant stays revoked until a new CLI grant.

Scope is fixed here and checked again on read: read-only sensor processing + localization research modules. Never
e2_gpu, frame_probe, pan-tilt, IMX477, focus, HOME, serial, /cmd_vel, posture, physical stop, ROPUS service changes.
"""
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

STANDING_MODULES = ("fastlio", "lio2", "hub", "e2", "e34", "observe", "rec", "web")
NEVER = ("e2_gpu", "frame_probe", "pantilt", "imx477", "focus", "home", "serial", "cmd_vel", "posture", "stop_robot", "ropus")
DEV_CAP_S = 4 * 3600          # development upper bound per experiment until long-run stability is measured
DEFAULT_EXP_S = 2 * 3600      # default per-experiment maximum (operator-adjustable in the web, 10 min .. DEV_CAP_S)
MIN_EXP_S = 600


def _canon(d):
    return json.dumps({k: d[k] for k in sorted(d) if k != "hmac"}, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


class Standing:
    def __init__(self, auth_dir):
        self.dir = Path(auth_dir)
        self.key_p = self.dir / "standing.key"
        self.grant_p = self.dir / "standing_research.json"
        self.state_p = self.dir / "standing_state.json"
        self.settings_p = self.dir / "standing_run_settings.json"

    # ---------------- key / signing
    def _key(self, create=False):
        if not self.key_p.exists():
            if not create:
                return None
            self.dir.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.key_p), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(secrets.token_bytes(32))
        return self.key_p.read_bytes()

    def _sign(self, d, key):
        return hmac.new(key, _canon(d), hashlib.sha256).hexdigest()

    def _write(self, path, d, key):
        d = dict(d, hmac=self._sign(d, key))
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1))
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        return d

    def _read(self, path, key):
        try:
            d = json.loads(path.read_text())
        except (OSError, ValueError):
            return None, "MISSING"
        if not key or not hmac.compare_digest(str(d.get("hmac", "")), self._sign(d, key)):
            return d, "HMAC_INVALID"
        return d, None

    # ---------------- CLI only
    def grant(self, user_text, device_name, device_stable_id, by):
        """Create a new grant (replaces any previous one, new grant_id). Called only by `ctl standing-grant` on the robot."""
        if not user_text or not user_text.strip():
            raise ValueError("user text required")
        key = self._key(create=True)
        limits = {m: {"max_run_s": DEV_CAP_S} for m in STANDING_MODULES}
        limits["lio2"]["accum_cloud_registered"] = True
        limits["e2"]["device"] = "cpu"
        limits["observe"].update(mode="SHADOW", start_temp_c=72, start_mem_mb=4300, abort_temp_c=80, abort_mem_mb=2500)
        g = dict(kind="STANDING_RESEARCH_AUTHORIZATION", grant_id=secrets.token_hex(8), created=time.strftime("%F %T %Z"),
                 created_epoch=time.time(), modules=list(STANDING_MODULES), excluded=list(NEVER), limits=limits,
                 device=dict(name=device_name, stable_id=device_stable_id), user_text=user_text,
                 user_text_sha256=hashlib.sha256(user_text.encode()).hexdigest(), recorded_by=by,
                 boot_bound=False, expires=None)
        g = self._write(self.grant_p, g, key)
        self._write(self.state_p, dict(grant_id=g["grant_id"], state="ACTIVE", since=time.time(), by=by, why="granted"), key)
        return g

    # ---------------- web (downgrade / restore only)
    def set_state(self, state, by, why=""):
        if state not in ("ACTIVE", "SUSPENDED", "REVOKED"):
            raise ValueError(state)
        key = self._key()
        g, err = self._read(self.grant_p, key)
        if err:
            raise PermissionError(f"NO_VALID_GRANT {err}")
        cur, serr = self._read(self.state_p, key)
        cur_state = None if serr or (cur or {}).get("grant_id") != g["grant_id"] else cur.get("state")
        if cur_state == "REVOKED":
            raise PermissionError("REVOKED (a new grant needs `ctl standing-grant` on the robot)")
        if state == "ACTIVE" and cur_state != "SUSPENDED":
            raise PermissionError(f"RESUME_ONLY_FROM_SUSPENDED (state {cur_state})")
        return self._write(self.state_p, dict(grant_id=g["grant_id"], state=state, since=time.time(), by=by, why=why), key)

    def exp_max_s(self):
        d = {}
        try:
            d = json.loads(self.settings_p.read_text())
        except (OSError, ValueError):
            pass
        v = d.get("experiment_max_s", DEFAULT_EXP_S)
        try:
            return int(min(DEV_CAP_S, max(MIN_EXP_S, float(v))))
        except (TypeError, ValueError):
            return DEFAULT_EXP_S

    def set_exp_max_s(self, seconds, by):
        s = int(seconds)
        if not MIN_EXP_S <= s <= DEV_CAP_S:
            raise ValueError(f"experiment max must be {MIN_EXP_S}..{DEV_CAP_S} s")
        tmp = self.settings_p.with_name(f".{self.settings_p.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(dict(experiment_max_s=s, set=time.strftime("%F %T %Z"), by=by)))
        os.replace(tmp, self.settings_p)
        return s

    # ---------------- every check
    def status(self, module=None):
        """dict(valid, why, state, modules, limits, grant_id, device, file). No boot / expiry test: the grant is standing."""
        key = self._key()
        g, err = self._read(self.grant_p, key)
        out = dict(kind="STANDING", valid=False, why=None, state=None, file=str(self.grant_p), boot_bound=False, expires_epoch=None)
        if err:
            out["why"] = "STANDING_" + err
            return out
        mods = [m for m in g.get("modules", []) if m in STANDING_MODULES]          # scope re-checked on read
        st, serr = self._read(self.state_p, key)
        state = "SUSPENDED" if serr else (st.get("state") if st.get("grant_id") == g.get("grant_id") else "SUSPENDED")
        exp_s = self.exp_max_s()
        limits = {m: dict(v, max_run_s=min(exp_s, int(v.get("max_run_s", DEV_CAP_S)))) for m, v in (g.get("limits") or {}).items()
                  if m in mods}
        out.update(state=state, modules=mods, limits=limits, grant_id=g.get("grant_id"), device=g.get("device"),
                   created=g.get("created"), experiment_max_s=exp_s, excluded=list(NEVER))
        if state != "ACTIVE":
            out["why"] = f"STANDING_{state}"
        elif module is not None and module not in mods:
            out["why"] = f"MODULE_NOT_IN_STANDING_SCOPE:{module}"
        else:
            out["valid"] = True
        return out
