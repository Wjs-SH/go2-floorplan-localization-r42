"""SEM1: private counterfactual WOULD_STOP / WOULD_DEFER evaluator (shadow only, never an actuator path).

The real FSM may be busy because actuation/requests are disabled in shadow; the counterfactual question is "what would the
policy have proposed if requests were allowed?". It reads the P2 timing-layer summary of every tick (computed
regardless of busy) and keeps its OWN memory (dedup), separate from P2's fired/observed memory, so it never marks a
sign as observed and never changes P2 state. Output = private JSONL + private status dict; nothing is sent anywhere.
  WOULD_STOP   : a VERIFIED track that is readable now (no 'why') and not deferred, first time per track within dedup_s
  WOULD_DEFER  : candidate deferred (geometry unavailable / no feasible window / E4 suppression), with the reason
  A WOULD_STOP is not an observation and not a STOP request.
"""
from __future__ import annotations

import json
import time

DEFER_STATES = ("DEFER_GEOMETRY_UNAVAILABLE", "E4_SUPPRESSED")


class WouldStopEvaluator:
    def __init__(self, path=None, dedup_s=30.0, clock=time.monotonic):
        self.path, self.dedup_s, self.clock = path, dedup_s, clock
        self.mem = {}                 # spot -> last WOULD_STOP mono (counterfactual memory only)
        self.defer_last = {}
        self.counts = dict(ticks=0, would_stop=0, would_defer=0)
        self.last = []

    def _write(self, rec):
        self.last = (self.last + [rec])[-20:]
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    def on_tick(self, t_rel, summary, context=None):
        self.counts["ticks"] += 1
        now = self.clock()
        out = []
        for s in summary or []:
            k, st, why = s.get("spot"), s.get("state"), s.get("why") or []
            if st in DEFER_STATES or (st in ("VERIFIED", "WAIT_READABLE") and "NO_FEASIBLE_STOP_WINDOW" in why):
                if now - self.defer_last.get((k, st), -1e9) >= 2.0:          # rate-limited log, not a count inflation
                    self.defer_last[(k, st)] = now
                    self.counts["would_defer"] += 1
                    rec = dict(kind="WOULD_DEFER", spot=k, state=st, why=why, t_rel=t_rel, mono=now, context=context)
                    self._write(rec)
                    out.append(rec)
            elif st == "VERIFIED" and not why:
                if now - self.mem.get(k, -1e9) >= self.dedup_s:
                    self.mem[k] = now
                    self.counts["would_stop"] += 1
                    rec = dict(kind="WOULD_STOP", spot=k, pred=s.get("pred"), now=s.get("now"), box_frame=s.get("box_frame"),
                               t_rel=t_rel, mono=now, context=context,
                               note="counterfactual only: not an observation, not a STOP request, not sent anywhere")
                    self._write(rec)
                    out.append(rec)
        return out

    def status(self):
        return dict(counts=dict(self.counts), last=self.last[-5:], note="PRIVATE counterfactual (shadow); actuator = Null")
