"""E5RT2: threshold mode for the semantic confirmation (L3) - legacy relative p90 vs a DEV-fixed calibration artifact.

legacy_p90 (default, comparison mode): thr = running 90th percentile of every sim seen in this run (inf for the first 50
  values). Kept bit-identical so its cold-start delay can be MEASURED, not hidden.
calibrated: thr = a FIXED value from a calibration artifact fitted on the DEV split only. It is ACTIVE only if the
  artifact is complete, its labels are human-verified and its id matches the requested one. Otherwise the mode is
  UNCALIBRATED: semantic results are TENTATIVE (displayed, logged) and no STOP request is issued from them.
  An artifact built from provisional (Claude) labels is never activatable. No threshold is ever invented at runtime,
  and the posters/lights met during a run never move the final STOP threshold in calibrated mode.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

LEGACY, CALIBRATED = "legacy_p90", "calibrated"
ACTIVE, UNCALIBRATED = "CALIBRATION_ACTIVE", "UNCALIBRATED"
REQUIRED = ("calibration_id", "method", "split", "thresholds", "data", "human_verified", "score", "created")


def artifact_id(d):
    keep = {k: d.get(k) for k in ("method", "split", "thresholds", "data", "score")}
    return "cal-" + hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:16]


class ThresholdMode:
    def __init__(self, mode=None, artifact_path=None, expect_id=None):
        self.mode = (mode or os.environ.get("SCI_E5RT2_THRESHOLD_MODE", LEGACY)).strip() or LEGACY
        self.path = artifact_path or os.environ.get("SCI_E5RT2_CALIBRATION") or None
        self.expect_id = expect_id or os.environ.get("SCI_E5RT2_CALIBRATION_ID") or None
        self.artifact, self.state, self.why = None, None, []
        if self.mode == LEGACY:
            self.state = LEGACY
            return
        if self.mode != CALIBRATED:
            raise ValueError(f"threshold mode {self.mode!r}")
        self.state, self.why = UNCALIBRATED, self._load()
        if not self.why:
            self.state = ACTIVE

    def _load(self):
        why = []
        if not self.path:
            return ["NO_CALIBRATION_ARTIFACT"]
        try:
            d = json.loads(Path(self.path).read_text())
        except (OSError, ValueError) as e:
            return [f"ARTIFACT_UNREADABLE:{type(e).__name__}"]
        miss = [k for k in REQUIRED if k not in d]
        if miss:
            why.append(f"ARTIFACT_INCOMPLETE:{miss}")
        if d.get("human_verified") is not True:
            why.append("LABELS_NOT_HUMAN_VERIFIED")
        if d.get("split") != "DEV":
            why.append(f"NOT_FITTED_ON_DEV:{d.get('split')}")
        if not miss and artifact_id(d) != d.get("calibration_id"):
            why.append("ARTIFACT_ID_MISMATCH(content changed)")
        if self.expect_id and d.get("calibration_id") != self.expect_id:
            why.append("NOT_THE_REQUESTED_CALIBRATION")
        thr = (d.get("thresholds") or {}).get("global")
        if not isinstance(thr, (int, float)) or not math.isfinite(thr):
            why.append("NO_FINITE_THRESHOLD")
        self.artifact = d
        return why

    def apply(self, res):
        """Owner side, before the policies: returns dict(thr_used, legacy_thr, state, stop_allowed, why). In calibrated +
        ACTIVE state the scene's 'like' flags are recomputed with the fixed threshold."""
        legacy = res.get("thr")
        if self.state == LEGACY:
            return dict(thr_used=legacy, legacy_thr=legacy, state=LEGACY, stop_allowed=True, why=[])
        if self.state != ACTIVE:
            return dict(thr_used=legacy, legacy_thr=legacy, state=UNCALIBRATED, stop_allowed=False,
                        why=list(self.why) + ["TENTATIVE_ONLY:no STOP from uncalibrated semantics"])
        thr = float(self.artifact["thresholds"]["global"])
        for r in res.get("regions") or []:
            m = r.get("msim")
            r["like_legacy"] = r.get("like")
            r["like"] = bool(m is not None and m >= thr and (r.get("score_status") or "FRESH") == "FRESH")
        res["thr_legacy"], res["thr"] = legacy, thr
        return dict(thr_used=thr, legacy_thr=legacy, state=ACTIVE, stop_allowed=True, why=[],
                    calibration_id=self.artifact["calibration_id"])

    def describe(self):
        return dict(mode=self.mode, state=self.state, why=list(self.why), path=self.path,
                    calibration_id=(self.artifact or {}).get("calibration_id"))
