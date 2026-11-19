#!/usr/bin/env python3
"""SCI E5 Observation Policy + event state machine.

SigLIP is NOT the stop trigger. Each candidate region (world-anchored legacy wall spot) passes through gates; only a
candidate that passes every enabled gate becomes STOP_REQUEST. Gates are grouped into ablation stages:

  A0  legacy decide() output (baseline, produced by the front end, not here)
  A1  + temporal persistence  : the same world-anchored wall spot must have been seen in >= min_scenes scenes and be
                                sign-like (SigLIP >= sim_floor) in >= min_pass of its last `window` observations
  A2  + hard negatives        : poster/board/directory/elevator/glare/light evidence (SigLIP negatives + context crop),
                                saturation/specular ratio, horizontal-surface rejection
  A3  + observability         : distance window, obliquity, cut-off at image border, pan/tilt reachability at the
                                predicted stop position, estimated text height in the IMX477 capture, robot speed
  A4  + E4 ambiguity gate     : suppress when E4 says the pose is CONFIDENT; E4_UNAVAILABLE -> independent E5 mode

Every region gets a decision {state, display, reasons[]}; all are logged. Thresholds live in config/e5_policy.json.
"""
from __future__ import annotations

import copy
import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

DEFAULT_CFG = {
    "legacy": {"sim_floor": 0.078, "sign_max_width_m": 0.85, "floor_band_m": [0.6, 2.8], "board_max": 0.02},
    "A1": {"window": 3, "min_pass": 2, "min_scenes": 2},
    "A2": {"neg_margin": 0.0, "ctx_margin": 0.02, "glare_sat_max": 0.05, "glare_spec_max": 0.10,
           "horizontal_max_cos": 0.7, "reject_elevator": True, "min_depth_valid": 0.3,
           "neg_classes": ["poster", "board", "directory", "glare", "light", "elevator"],
           "ctx_classes": ["poster", "board", "directory", "glare", "elevator"]},
    "A3": {"d_min_m": 1.0, "d_max_m": 5.0, "obliquity_max_deg": 60.0, "edge_px": 6, "min_text_px": 18,
           "stop_reaction_s": 1.0, "max_speed_mps": 1.2},
    "A4": {"enabled": True},
    "dedup": {"same_m": 2.0, "same_sec": 60.0},
    "event": {"stable_lin_mps": 0.05, "stable_yaw_dps": 3.0, "stable_hold_s": 0.7, "wait_stable_timeout_s": 20.0,
              "aim_timeout_s": 25.0, "capture_retries": 2, "min_capture_sharp": 40.0, "max_capture_sat": 0.08},
}

DISPLAY = {  # reason prefix -> UI display state
    "board": "REJECTED_POSTER", "poster": "REJECTED_POSTER", "directory": "REJECTED_POSTER", "width": "REJECTED_POSTER",
    "elevator": "REJECTED_ELEVATOR", "glare": "REJECTED_GLARE", "light": "REJECTED_GLARE", "saturation": "REJECTED_GLARE",
    "specular": "REJECTED_GLARE", "no_depth": "REJECTED_GLARE", "horizontal": "REJECTED_GLARE",
    "too_far": "REJECTED_DISTANCE", "too_close": "REJECTED_DISTANCE", "obliquity": "REJECTED_ANGLE",
    "cut_off": "REJECTED_ANGLE", "pantilt": "REJECTED_PANTILT", "text_small": "REJECTED_DISTANCE",
    "speed": "VERIFYING", "persist": "VERIFYING", "legacy": "CANDIDATE", "height": "CANDIDATE",
    "already": "ALREADY_OBSERVED", "e4": "E4_SUPPRESSED", "busy": "CANDIDATE",
}


def load_cfg(path=None):
    cfg = copy.deepcopy(DEFAULT_CFG)
    if path and Path(path).exists():
        user = json.loads(Path(path).read_text())
        for k, v in user.items():
            if isinstance(v, dict):
                cfg.setdefault(k, {}).update(v)
            else:
                cfg[k] = v
    return cfg


class StagePolicy:
    """One ablation stage with its own memories (persistence history, fired spots)."""

    def __init__(self, name, cfg, geom, a1=True, a2=True, a3=True, a4=True):
        self.name, self.cfg, self.geom = name, cfg, geom
        self.a1, self.a2, self.a3, self.a4 = a1, a2, a3, a4
        self.hist = defaultdict(lambda: deque(maxlen=max(1, cfg["A1"]["window"])))   # spot -> recent legacy_pass flags
        self.fired_spots, self.fired = set(), []                                   # fired = [(t, xyz)]
        self.events = []

    # ------------------------------------------------------------------ gates
    def legacy_gate(self, r, thr):
        c, why = self.cfg["legacy"], []
        if r["msim"] is None or r["msim"] < thr:
            why.append("legacy:below_relative_threshold")
        if r["sim"] < c["sim_floor"]:
            why.append("legacy:below_sim_floor")
        if r["z"] is not None:
            wm, _ = self.geom.size_m(r["box"], r["z"])
            if wm > c["sign_max_width_m"]:
                why.append("width:physical_width_%.2fm" % wm)
        if r.get("hgt") is not None and not (c["floor_band_m"][0] <= r["hgt"] <= c["floor_band_m"][1]):
            why.append("height:%.2fm_outside_band" % r["hgt"])
        if r["board"] > c["board_max"]:
            why.append("board:context_board_margin_%.3f" % r["board"])
        return why

    def hardneg_gate(self, r, f):
        c, why = self.cfg["A2"], []
        neg, ctx, cp = r.get("neg") or {}, r.get("ctx") or {}, r.get("ctx_plate")
        s = r["sim"]
        for cls in c["neg_classes"]:
            if cls == "elevator" and not c["reject_elevator"]:
                continue
            v = neg.get(cls)
            if v is not None and v - s > c["neg_margin"]:
                why.append(f"{cls}:tile_neg_{v - s:+.3f}")
        if cp is not None:
            for cls in c["ctx_classes"]:
                if cls == "elevator" and not c["reject_elevator"]:
                    continue
                v = ctx.get(cls)
                if v is not None and v - cp > c["ctx_margin"]:
                    why.append(f"{cls}:context_{v - cp:+.3f}")
        if f["glare"]["sat"] > c["glare_sat_max"]:
            why.append("saturation:%.3f" % f["glare"]["sat"])
        if f["glare"]["spec"] > c["glare_spec_max"]:
            why.append("specular:%.3f" % f["glare"]["spec"])
        pl = f["plane"]
        if pl.get("valid_in", 1.0) < c["min_depth_valid"]:
            why.append("no_depth:valid_%.2f" % pl.get("valid_in", 0.0))
        if pl.get("ok") and f.get("up") is not None and abs(float(np.dot(pl["normal"], f["up"]))) > c["horizontal_max_cos"]:
            why.append("horizontal:surface_normal_vertical")
        return why

    def observability_gate(self, r, f):
        c, why = self.cfg["A3"], []
        z = f.get("z_tile") or r["z"]
        if z is None:
            return ["too_far:no_depth_range"]
        v = f.get("speed")
        z_pred = z if v is None else z - v * c["stop_reaction_s"] * max(0.0, math.cos(f["bearing"]))
        f["z_pred"] = z_pred
        if z > c["d_max_m"]:
            why.append("too_far:%.1fm" % z)
        if z_pred < c["d_min_m"]:
            why.append("too_close:pred_%.1fm" % z_pred)
        ob = f.get("obliquity_pred")
        if ob is None and f["plane"].get("ok"):
            ob = f["plane"]["obliquity_deg"]
        if ob is not None and ob > c["obliquity_max_deg"]:
            why.append("obliquity:pred_%.0fdeg" % ob)
        if f["edge"] < c["edge_px"]:
            why.append("cut_off:edge_%.0fpx" % f["edge"])
        pt = f["pantilt"]
        if not pt["ok"]:
            why.append("pantilt:%s" % pt["why"])
        if f["text_px"] is not None and f["text_px"] < c["min_text_px"]:
            why.append("text_small:%.0fpx" % f["text_px"])
        if v is not None and v > c["max_speed_mps"]:
            why.append("speed:%.2fmps" % v)
        return why

    def persist_gate(self, r):
        c = self.cfg["A1"]
        h = self.hist.get(r["spot"]) if r["spot"] is not None else None
        n_pass = sum(h) if h else 0
        if r["spot"] is None:
            return ["persist:no_world_spot"]
        if r["nscene"] < c["min_scenes"] or n_pass < c["min_pass"]:
            return [f"persist:{n_pass}/{len(h) if h else 0}_scenes_{r['nscene']}"]
        return []

    def already(self, r, t):
        d = self.cfg["dedup"]
        if r["spot"] is not None and r["spot"] in self.fired_spots:
            return ["already:same_wall_spot"]
        if r["xyz"] is not None:
            for tf, p in self.fired:
                if t - tf <= d["same_sec"] and math.hypot(p[0] - r["xyz"][0], p[1] - r["xyz"][1]) <= d["same_m"]:
                    return ["already:within_%.1fm_%.0fs" % (d["same_m"], d["same_sec"])]
        return []

    # ------------------------------------------------------------------ scene
    def step(self, t, regions, feats, thr, e4, busy=False):
        """Returns (decisions per region, stop dict or None)."""
        out, cands = [], []
        for r, f in zip(regions, feats):
            ss = r.get("score_status") or "FRESH"
            if ss != "FRESH":                       # E5RT2: no fresh SigLIP score = no evidence either way (typed)
                out.append(dict(state="DEFER", reasons=[ss], display="CANDIDATE"))
                continue
            why_leg = self.legacy_gate(r, thr)
            if r["spot"] is not None:               # per-spot evidence: SigLIP sign-likeness this scene
                self.hist[r["spot"]].append(r["sim"] >= self.cfg["legacy"]["sim_floor"])
            why = list(why_leg)
            hard = self.hardneg_gate(r, f) if self.a2 else []
            obs = self.observability_gate(r, f) if self.a3 else []
            per = self.persist_gate(r) if self.a1 else []
            dup = self.already(r, t)
            # order of precedence for the displayed state: rejection evidence > deferral > verification
            reasons = why + hard + obs + per + dup
            if hard or [w for w in why if not w.startswith("legacy")]:
                state = "REJECT"
            elif why:
                state = "TRACK" if r["sim"] >= self.cfg["legacy"]["sim_floor"] else "SEARCH"
            elif obs:
                state = "DEFER"
            elif per:
                state = "VERIFY"
            elif dup:
                state = "REJECT"
            else:
                state = "PASS"
            out.append(dict(state=state, reasons=reasons, display=self.display(state, reasons)))
            if state == "PASS":
                cands.append(len(out) - 1)
        stop = None
        if cands:
            if self.a4 and e4 is not None and e4.get("available") and e4.get("level") == "CONFIDENT":
                for j in cands:
                    out[j].update(state="DEFER", display="E4_SUPPRESSED", reasons=out[j]["reasons"] + ["e4:pose_confident"])
            elif busy:
                for j in cands:
                    out[j].update(state="DEFER", display="CANDIDATE", reasons=out[j]["reasons"] + ["busy:event_in_progress"])
            else:
                j = max(cands, key=lambda k: (regions[k]["msim"] or -1, regions[k]["sim"]))
                r = regions[j]
                out[j].update(state="STOP_REQUEST", display="STOP_REQUEST")
                if r["spot"] is not None:
                    self.fired_spots.add(r["spot"])
                if r["xyz"] is not None:
                    self.fired.append((t, r["xyz"]))
                stop = dict(t=t, stage=self.name, index=j, box=r["box"], z=r["z"], xyz=r["xyz"], spot=r["spot"],
                            sim=r["sim"], msim=r["msim"], nobs=r["nobs"], nscene=r["nscene"],
                            features={k: feats[j][k] for k in ("pantilt", "text_px", "edge", "speed", "z_pred", "z_tile", "obliquity_pred", "aim_px") if k in feats[j]},
                            tile=r.get("tile"),
                            plane={k: v for k, v in feats[j]["plane"].items() if k != "normal"},
                            glare=feats[j]["glare"], e4=e4)
                for k in cands:
                    if k != j:
                        out[k].update(state="DEFER", display="CANDIDATE", reasons=out[k]["reasons"] + ["busy:other_candidate_selected"])
                self.events.append(stop)
        return out, stop

    @staticmethod
    def display(state, reasons):
        if state == "PASS":
            return "VERIFYING"
        if state == "SEARCH":
            return "SEARCHING"
        for w in reasons:
            k = w.split(":")[0]
            if k == "legacy":
                continue
            return DISPLAY.get(k, "CANDIDATE")
        return "CANDIDATE"


def make_stages(cfg, geom):
    return {"A1": StagePolicy("A1", cfg, geom, a1=True, a2=False, a3=False, a4=False),
            "A2": StagePolicy("A2", cfg, geom, a1=True, a2=True, a3=False, a4=False),
            "A3": StagePolicy("A3", cfg, geom, a1=True, a2=True, a3=True, a4=False),
            "A4": StagePolicy("A4", cfg, geom, a1=True, a2=True, a3=True, a4=True)}


# ======================================================================= event state machine
EVENT_STATES = ["STOP_REQUEST", "WAIT_STABLE", "STOP_CONFIRMED", "AIM", "CAPTURE", "CAPTURED", "RELEASE",
                "READ_ASYNC", "LOCALIZE_ASYNC", "FIX_CONFIRMED", "FAILED", "CLOSED"]


class EventFSM:
    """Synchronous part of one observation event: STOP_REQUEST -> WAIT_STABLE -> AIM -> CAPTURE -> RELEASE.
    Reading/localisation continue asynchronously after RELEASE (tracked by the reader via event id).
    Every transition is logged with time and reason. No robot motion command is ever sent from here."""

    def __init__(self, ev_id, stop, cfg, clock=time.time):
        self.id, self.stop, self.cfg, self.clock = ev_id, stop, cfg["event"], clock
        self.state, self.t_enter = "STOP_REQUEST", clock()
        self.log = [dict(state="STOP_REQUEST", t=self.t_enter, reason="policy_pass", t_scene=stop["t"])]
        self.stable_since, self.retries, self.result = None, 0, {}

    def to(self, state, reason, **kw):
        now = self.clock()
        self.log[-1]["t_exit"] = now
        self.log.append(dict(state=state, t=now, reason=reason, **kw))
        self.state, self.t_enter = state, now

    @property
    def active(self):
        return self.state in ("STOP_REQUEST", "WAIT_STABLE", "STOP_CONFIRMED", "AIM", "CAPTURE")

    def on_odom(self, lin, yaw_rate):
        """Called every cycle with the current robot speed (None = odometry unavailable)."""
        c, now = self.cfg, self.clock()
        if self.state == "STOP_REQUEST":
            self.to("WAIT_STABLE", "waiting_for_operator_stop")
        if self.state != "WAIT_STABLE":
            return
        if lin is None:
            if now - self.t_enter > c["wait_stable_timeout_s"]:
                self.to("FAILED", "odometry_unavailable_cannot_confirm_stop")
            return
        if lin <= c["stable_lin_mps"] and yaw_rate <= c["stable_yaw_dps"]:
            self.stable_since = self.stable_since if self.stable_since is not None else now
            if now - self.stable_since >= c["stable_hold_s"]:
                self.to("STOP_CONFIRMED", f"odom_stable_{c['stable_hold_s']}s", lin=lin, yaw_rate=yaw_rate)
                return
        else:
            self.stable_since = None
        # R1: the timeout is checked on EVERY branch (legacy checked it only while 'moving': 10-15 ev5 confirmed at 20.38 s)
        if now - self.t_enter > c["wait_stable_timeout_s"]:
            self.to("FAILED", "stop_not_confirmed_timeout")
