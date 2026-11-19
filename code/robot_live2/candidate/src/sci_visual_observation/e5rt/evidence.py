"""Temporal confirmation policies for one world-anchored track (pure Python, no I/O).

Score definitions used here are the ones the field code actually logs (verified in baseline code):
  sim   = max over 100 px tiles of max over the 3 'roomstd' prompts of RAW COSINE (SigLIP fp16 image feature vs cached
          text feature). Not a probability; never passed through a sigmoid here.
  thr   = legacy relative threshold: running 90th percentile of every region sim seen so far (inf during the first 50
          values = warm-up). The 'about 0.104' in the student report is this value.
  conf  = strongest tile-level hard-negative cosine of the SAME tile/feature (poster/board/directory/light by config).
  floor = legacy sim_floor 0.078.

Policies (all receive the same chronological observations):
  Mean3Legacy        msim = mean of the last 3 sims (cached/duplicate observations included, exactly like legacy) >= thr,
                     sim >= floor, persistence (>= 2 scenes, >= 1 pass in last 3)  -> reproduces P2 layers L2+L3.
  MaxOnly            max sim so far >= thr (COMPARISON ONLY - known to accept one-frame glare spikes).
  EmaHysteresis      time-aware EMA of (sim - thr), enter >= +h_in, leave < -h_out, >= 2 distinct observations.
  QualityEvidence    bounded sequential evidence (heuristic, NOT a calibrated posterior):
                        d_t = sim_t - thr_t ; m_t = sim_t - conf_t
                        strong negative (m_t < -m_neg)         -> inc = -c            (weight 1, quality does not excuse it)
                        d_t >= 0                               -> inc = w_t * min(d_t, c)
                        d_t <  0                               -> inc = w_t * max(d_t, -c)   (blurred shortfall down-weighted)
                        E_t = exp(-dt/tau) * E_prev + inc ; without a strong negative E_t is floored at -c/2
                        (shortfall = not yet visible, never a rejection by itself)
                     only FRESH observations (new frame, not a cache hit, appearance changed) update E / n_distinct.
                     TENTATIVE: strong single fresh observation (d_t >= d_strong, m_t >= 0) or E >= e_tent.
                     CONFIRMED: E >= e_conf AND >= min_distinct distinct POSITIVE obs AND last fresh obs not strongly negative.
                     leave CONFIRMED when E < e_leave; REJECTED when E <= -e_rej; EXPIRED when no fresh obs for ttl_s.
States: UNKNOWN, TENTATIVE, CONFIRMED, REJECTED, EXPIRED (+ reason codes from contracts).
"""
from __future__ import annotations

import math
from collections import deque

from . import contracts as C

UNKNOWN, TENTATIVE, CONFIRMED, REJECTED, EXPIRED = "UNKNOWN", "TENTATIVE", "CONFIRMED", "REJECTED", "EXPIRED"


def _fresh_key(o):
    return (o.get("frame_id"), o.get("appearance"))


class _Base:
    name = "base"

    def __init__(self, floor=0.078):
        self.floor = floor
        self.state, self.reason = UNKNOWN, C.TEMPORAL_PENDING
        self.n_obs, self.n_distinct, self.seen = 0, 0, set()
        self.t_first = self.t_state = self.t_confirm = None
        self.last_fresh_t = None

    def _distinct(self, o):
        """True if o is new independent evidence (not a cache hit, not the same frame/appearance)."""
        if o.get("cached"):
            return False
        k = _fresh_key(o)
        if k in self.seen:
            return False
        self.seen.add(k)
        return True

    def _set(self, st, reason, t):
        if st != self.state:
            self.state, self.t_state = st, t
            if st == CONFIRMED and self.t_confirm is None:
                self.t_confirm = t
        self.reason = reason

    def snapshot(self):
        return dict(policy=self.name, state=self.state, reason=self.reason, n_obs=self.n_obs, n_distinct=self.n_distinct,
                    t_first=self.t_first, t_confirm=self.t_confirm)


class Mean3Legacy(_Base):
    name = "mean3_legacy"

    def __init__(self, floor=0.078, window=3, min_scenes=2, min_pass=1):
        super().__init__(floor)
        self.sims, self.passes, self.scenes = [], deque(maxlen=window), set()
        self.window, self.min_scenes, self.min_pass = window, min_scenes, min_pass

    def update(self, o):
        t = o["t"]
        self.t_first = t if self.t_first is None else self.t_first
        self.n_obs += 1
        if self._distinct(o):
            self.n_distinct += 1
        self.sims.append(o["sim"])                       # legacy: every observation, cached or not
        self.passes.append(o["sim"] >= self.floor)
        self.scenes.add(o.get("scene_t", t))
        thr = o.get("thr")
        msim = sum(self.sims[-self.window:]) / len(self.sims[-self.window:])
        if thr is None or not math.isfinite(thr):
            self._set(UNKNOWN, C.TEMPORAL_PENDING + ":threshold_warmup", t)
        elif msim < thr or o["sim"] < self.floor:
            self._set(UNKNOWN, C.SEMANTIC_MARGIN_LOW, t)
        elif len(self.scenes) < self.min_scenes or sum(self.passes) < self.min_pass:
            self._set(TENTATIVE, C.TEMPORAL_PENDING, t)
        else:
            self._set(CONFIRMED, "msim_ge_thr", t)
        return self.state


class MaxOnly(_Base):
    name = "max_only"

    def __init__(self, floor=0.078):
        super().__init__(floor)
        self.best = -1.0

    def update(self, o):
        t = o["t"]
        self.t_first = t if self.t_first is None else self.t_first
        self.n_obs += 1
        if self._distinct(o):
            self.n_distinct += 1
        self.best = max(self.best, o["sim"])
        thr = o.get("thr")
        if thr is None or not math.isfinite(thr):
            self._set(UNKNOWN, C.TEMPORAL_PENDING + ":threshold_warmup", t)
        elif self.best >= thr and self.best >= self.floor:
            self._set(CONFIRMED, "max_ge_thr", t)
        else:
            self._set(UNKNOWN, C.SEMANTIC_MARGIN_LOW, t)
        return self.state


class EmaHysteresis(_Base):
    name = "ema_hysteresis"

    def __init__(self, floor=0.078, tau_s=2.0, h_in=0.0, h_out=0.006, min_distinct=2):
        super().__init__(floor)
        self.tau, self.h_in, self.h_out, self.min_distinct = tau_s, h_in, h_out, min_distinct
        self.ema, self.t_last = None, None

    def update(self, o):
        t = o["t"]
        self.t_first = t if self.t_first is None else self.t_first
        self.n_obs += 1
        thr = o.get("thr")
        if not self._distinct(o):
            return self.state
        self.n_distinct += 1
        if thr is None or not math.isfinite(thr):
            self._set(UNKNOWN, C.TEMPORAL_PENDING + ":threshold_warmup", t)
            return self.state
        d = o["sim"] - thr
        if self.ema is None:
            self.ema = d
        else:
            a = math.exp(-max(0.0, t - self.t_last) / self.tau)
            self.ema = a * self.ema + (1 - a) * d
        self.t_last = t
        if self.state == CONFIRMED:
            if self.ema < -self.h_out:
                self._set(UNKNOWN, C.SEMANTIC_MARGIN_LOW + ":hysteresis_leave", t)
        elif self.ema >= self.h_in and self.n_distinct >= self.min_distinct and o["sim"] >= self.floor:
            self._set(CONFIRMED, "ema_ge_thr", t)
        elif self.ema >= self.h_in:
            self._set(TENTATIVE, C.TEMPORAL_PENDING, t)
        else:
            self._set(UNKNOWN, C.SEMANTIC_MARGIN_LOW, t)
        return self.state


class QualityEvidence(_Base):
    name = "quality_evidence"

    def __init__(self, floor=0.078, tau_s=4.0, c=0.02, m_neg=0.02, d_strong=0.012, e_tent=0.004, e_conf=0.008,
                 e_leave=0.0, e_rej=0.03, min_distinct=2, ttl_s=8.0, w_min=0.2):
        super().__init__(floor)
        self.p = dict(tau_s=tau_s, c=c, m_neg=m_neg, d_strong=d_strong, e_tent=e_tent, e_conf=e_conf, e_leave=e_leave,
                      e_rej=e_rej, min_distinct=min_distinct, ttl_s=ttl_s, w_min=w_min)
        self.E, self.t_last, self.last_strong_neg = 0.0, None, False
        self.n_pos = 0                                 # distinct fresh observations with sim >= thr
        self.trace = []

    def tick(self, t):
        """Time passes without fresh evidence (e.g. track out of view): expiry only, no evidence change."""
        if self.last_fresh_t is not None and t - self.last_fresh_t > self.p["ttl_s"] and self.state not in (REJECTED, EXPIRED):
            self._set(EXPIRED, C.STALE_RESULT, t)
        return self.state

    def update(self, o):
        p, t = self.p, o["t"]
        self.t_first = t if self.t_first is None else self.t_first
        self.n_obs += 1
        thr = o.get("thr")
        if not self._distinct(o):                      # cache hit / same frame: shown, never counted
            return self.tick(t)
        if thr is None or not math.isfinite(thr):
            self.n_distinct += 1
            self.last_fresh_t = t
            self._set(UNKNOWN, C.TEMPORAL_PENDING + ":threshold_warmup", t)
            return self.state
        self.n_distinct += 1
        self.last_fresh_t = t
        sim = o["sim"]
        d = sim - thr
        conf = o.get("conf")
        m = None if conf is None else sim - conf
        w = max(p["w_min"], min(1.0, float(o.get("quality", 1.0))))
        strong_neg = m is not None and m < -p["m_neg"]
        if d >= 0 and not strong_neg:
            self.n_pos += 1
        if strong_neg:
            inc = -p["c"]
        elif d >= 0:
            inc = w * min(d, p["c"])
        else:
            inc = w * max(d, -p["c"])
        if self.t_last is not None:
            self.E *= math.exp(-max(0.0, t - self.t_last) / p["tau_s"])
        self.E += inc
        if not strong_neg:                             # a low score (far / small / blurred) is "not yet seen", not proof
            self.E = max(self.E, -0.5 * p["c"])        # of a non-sign: bounded shortfall memory; REJECTED needs confusers
        self.t_last, self.last_strong_neg = t, strong_neg
        self.trace.append(dict(t=t, d=d, m=m, w=w, inc=inc, E=self.E))
        if self.E <= -p["e_rej"]:
            self._set(REJECTED, C.SEMANTIC_MARGIN_LOW + ":evidence_negative", t)
        elif self.state == CONFIRMED and self.E >= p["e_leave"] and not strong_neg:
            self._set(CONFIRMED, "evidence_holds", t)
        elif (self.E >= p["e_conf"] and self.n_pos >= p["min_distinct"] and not strong_neg and sim >= self.floor):
            self._set(CONFIRMED, "evidence_ge_conf", t)
        elif (d >= p["d_strong"] and (m is None or m >= 0)) or self.E >= p["e_tent"]:
            self._set(TENTATIVE, C.TEMPORAL_PENDING, t)
        else:
            self._set(UNKNOWN, C.SEMANTIC_MARGIN_LOW if d < 0 else C.TEMPORAL_PENDING, t)
        return self.state

    def snapshot(self):
        s = super().snapshot()
        s.update(E=self.E, params=dict(self.p))
        return s


POLICIES = {"mean3_legacy": Mean3Legacy, "max_only": MaxOnly, "ema_hysteresis": EmaHysteresis,
            "quality_evidence": QualityEvidence}


def roi_quality(sharp, mean_gray, sat_frac, min_side_px, sharp_ref=60.0, min_side_ref=24.0):
    """Quality weight in [0,1] for ONE observation of a ROI, from metrics computed on the SAME ROI/resolution.
    sharp = Laplacian variance of the ROI at full resolution (grey), mean_gray 0..255, sat_frac = fraction >= 245,
    min_side_px = shorter side of the ROI in full-res pixels. The reference values are TUNABLE_DEV_VALUEs."""
    def clip01(x):
        return max(0.0, min(1.0, x))
    q_sharp = clip01(sharp / sharp_ref) if sharp is not None else 1.0
    q_exp = clip01(1.0 - abs((mean_gray if mean_gray is not None else 128.0) - 128.0) / 128.0 * 0.8)
    q_sat = clip01(1.0 - 4.0 * (sat_frac or 0.0))
    q_size = clip01((min_side_px or min_side_ref) / min_side_ref)
    return round(q_sharp * q_exp * q_sat * q_size, 4)
