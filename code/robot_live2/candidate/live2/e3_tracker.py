"""e3_tracker.py — LIVE2 E3: incremental update of the REAL E2 hypotheses (no full-floor fit per frame).

Input: the student LiveStore written by lio_monitor2 (odom_link.csv rows in the odom_link convention, samples/<sid> proxy
structure points, far_points.npz), e2_state.json (E2 candidates as they are produced), the E2 light maps (npy, mmap).

Per hypothesis h (one E2 candidate = floor + map pose of the E2 window anchor + phi, ppm, beta, mirror):
  propagate   pose_s = e2_core.pose_in_map(h, x_s, y_s, yaw_s)  — the student pose_tools.map_pose_px transform, valid only
              inside the SAME odom session (session change -> every hypothesis is invalidated, never re-used)
  re-evaluate on a KEYFRAME s (moved >= KF_MOVE_M or turned >= KF_TURN_DEG since the previous keyframe, s after h.fit_t):
              the student merit terms on that one frame: wall cost (mean of the lowest 70 % of point->wall distances,
              capped 5 m, metric_fit_v2 field F), T2 (> 0.6 m from any wall), T3 (> 0.5 m outside the building footprint),
              X (through-wall path length between the previous and this keyframe), traversable pose, + twoway (student
              floor score). frame_merit = cost + 1.5 T2 + 3.0 T3 + 1.0 min(X, 0.3) + 0.5 [pose not traversable]
  merge       a re-search candidate on the same floor whose pose (propagated to its fit time) is within MERGE_M and
              MERGE_DEG of an alive hypothesis from an EARLIER pass joins that hypothesis (one ledger, no double count)
  score       S = (W_E2 * e2_merit + sum of the last N_KF frame_merits) / (W_E2 + n)   (lower = better, metres; not a
              probability). Evidence is keyed by (hypothesis, sample_id): a sample is counted once, samples inside the E2
              window (t <= fit_t) are never re-counted, a merged duplicate keeps one ledger.
Lifecycle (no single weak observation removes a hypothesis):
  ACTIVE -> SUSPENDED   only after >= SUSP_N consecutive keyframes with S - best > SUSP_GAP (and >= 3 keyframes of
                        evidence), or >= 2 hard violations (X > 0.3 m student infeasible) in the last 5 keyframes
  SUSPENDED -> ACTIVE   S - best < REACT_GAP on REACT_N consecutive keyframes (recovery path; suspended ones keep being
                        propagated and scored, round-robin within the per-keyframe budget)
  -> DROPPED            only to respect the memory cap (worst suspended first) or on odom session change; logged, shown.
Local tracking: a fixed E2 transform drifts over a long path (1 % scale / 1 deg = 1-2 m after 100 m; seen on the S6
replay, run 2). After a keyframe has been SCORED with the current transform, the hypothesis is re-aligned on its last
TRK_N keyframes by a bounded grid search (+-0.2 m, +-1 deg, +-1 % scale, student trimmed wall cost) and re-anchored at
that keyframe. A hypothesis can only creep, never jump to another place; the next keyframe again scores the prediction.
Stationary robot: no keyframe -> no new evidence, hypotheses and scores are kept as they are.
"""
import csv
import json
import math
import os
import time
import zlib
from collections import deque
from pathlib import Path

import numpy as np

KF_MOVE_M, KF_TURN_DEG = 0.5, 15.0
W_E2, N_KF = 3.0, 20
SUSP_GAP, SUSP_N, REACT_GAP, REACT_N = 0.25, 3, 0.12, 2
X_HARD, CAP_ALL, BUDGET_PER_KF, CATCHUP_PER_TICK = 0.3, 120, 60, 8
MERGE_M, MERGE_DEG = 1.0, 10.0          # = student metric_fit_v2 DUP_M / DUP_DEG; across passes only
W_FREE, W_OUT, W_CROSS, W_TRAV = 1.5, 3.0, 1.0, 0.5
FAR_M, OUT_M = 0.6, 0.5
BAD_RECENT = 1.0                        # legacy absolute rule (kept for reference; no longer raises the request, see below)
# Geometry consistency (2026-10-19 diagnosis on real 4F data, tools/e3_term_diag.py): with the FAST-LIO registered-cloud proxy
# input ~40 % of points are > 0.6 m from any plan wall (clutter), so EVERY hypothesis has frame merit ~1 m, the correct one
# included (best-recent p50 0.77, p90 1.09) while the crowd median is 2.18 -> separation (median - best) p10 0.87 m.
# Inconsistent = the best no longer separates from the crowd, or nothing fits at all.
SEP_MIN_M, SEP_N, ABS_FAIL_M, CROWD_MIN = 0.3, 3, 2.0, 5
GEO_MIN_KF = 8          # 2026-10-20 field: with 3 keyframes per fresh coarse hypothesis the test fired falsely at 48-50 s
N_EVID = 3                              # keyframes before a hypothesis may outrank evidence-backed ones (PENDING_EVIDENCE)
BACKLOG_KF, BACKLOG_ROOM = 10, 10       # R35: with this many pending keyframes, suspended hypotheses get at most this many slots per keyframe (deferred, no penalty)
EVAL_PER_KF = 40                        # measured ~13-16 ms per hypothesis-keyframe on the Orin NX -> ~0.55 s per keyframe
TICK_BUDGET_S = 0.9
SEM_DECISIVE_P = 0.6                    # OCR alone >= 0.85 conf gives p 0.68 (student fuse); a 0.5/0.5 conflict is not decisive
SEM_ORDER = {"SEMANTIC_CONSISTENT": 0, "TEXT_FLOOR_SUPPORT": 0.5, None: 1, "NO_LABEL": 1, "PLACE_MISMATCH": 2, "FLOOR_MISMATCH": 3}
# 2026-11-18 B7: TEXT_FLOOR_SUPPORT = a plate's words (no room number) point to this hypothesis' floor (e6e7_link, map_text). Support only: set on hypotheses
# without a room-number verdict, never lowers anyone below 'no reading', and the next room-number reading replaces it (the number has priority).
# 2026-11-18 B4: hypotheses a decisive plate reading found consistent (suspended ones included) are evaluated and tracked first on every keyframe.
SEM_FIRST = os.environ.get("LIVE2_E3_SEM_FIRST", "1") == "1"                     # catch-up keyframes per tick stop after this much time
# local tracking (per hypothesis, after the keyframe was scored with the PREVIOUS transform = honest prediction):
TRK_N, TRK_XY_M, TRK_DEG, TRK_SCALE = 3, (-0.2, -0.1, 0.0, 0.1, 0.2), (-1.0, -0.5, 0.0, 0.5, 1.0), (-0.01, 0.0, 0.01)
TRK_SCALE_CUM = 0.05      # R42: cumulative tracking scale bound around the E2 fit (+-5 %, the rel_unc of semantic_scale_v1.json); +-1 % per step stays


def _A(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [-s, -c]])


def pose_in_map(h, x, y, yaw):
    phi = math.radians(h['phi'])
    p = np.asarray(h['pos'], float) + h['ppm'] * (_A(phi) @ (np.array([x, y], float) - np.asarray(h['anchor'], float)))
    hd = phi + yaw + h['beta']
    return float(p[0]), float(p[1]), float((hd + math.pi) % (2 * math.pi) - math.pi)


def ang_deg(a, b):
    return abs((math.degrees(a - b) + 180.0) % 360.0 - 180.0)


class LightMap:
    def __init__(self, d):
        d = Path(d)
        self.meta = json.loads((d / 'meta.json').read_text())
        self.H, self.W = self.meta['H'], self.meta['W']
        self.F = np.load(d / 'F_q4.npy', mmap_mode='r')
        self.out = np.load(d / 'outd_q4.npy', mmap_mode='r')
        self.wall_tr = np.load(d / 'wall_tr.npy', mmap_mode='r')
        self.ok = np.load(d / 'ok.npy', mmap_mode='r')
        self.wall_real = np.load(d / 'wall_real.npy', mmap_mode='r')

    def sample(self, arr, px, big):
        x = np.rint(px[:, 0]).astype(int); y = np.rint(px[:, 1]).astype(int)
        inb = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.H)
        v = np.full(len(px), big, float)
        v[inb] = arr[y[inb], x[inb]] / 4.0
        return v

    def crossing_m(self, track_px, ppm):
        """metric_fit_v2._crossing_m on the through-wall mask (map outside counts as crossing)."""
        if len(track_px) < 2:
            return 0.0
        seg = np.diff(track_px, axis=0)
        n = np.maximum(np.ceil(np.hypot(seg[:, 0], seg[:, 1])).astype(int), 1)
        pts = np.vstack([track_px[i] + seg[i] * (np.arange(n[i])[:, None] / n[i]) for i in range(len(seg))] + [track_px[-1:]])
        x, y = np.rint(pts[:, 0]).astype(int), np.rint(pts[:, 1]).astype(int)
        inb = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.H)
        bad = ~inb
        bad[inb] = self.wall_tr[y[inb], x[inb]]
        return float(bad.sum() / ppm)

    def traversable(self, px, py):
        x, y = int(round(px)), int(round(py))
        return bool(0 <= x < self.W and 0 <= y < self.H and self.ok[y, x])


class Store:
    """Read-only view of the student LiveStore written by lio_monitor2."""

    def __init__(self, root):
        self.root = Path(root)
        self.rows, self._mtime = [], None

    def session(self):
        try:
            return json.loads((self.root / 'session.json').read_text())
        except (OSError, ValueError):
            return None

    def refresh(self):
        p = self.root / 'odom_link.csv'
        try:
            mt = p.stat().st_mtime
        except OSError:
            self.rows = []
            return False
        if mt == self._mtime:
            return False
        self._mtime = mt
        try:
            self.rows = [(r['sample_id'], float(r['t_sec']), float(r['fastlio_x']), float(r['fastlio_y']), float(r['fastlio_yaw_rad']))
                         for r in csv.DictReader(open(p))]
        except (OSError, ValueError, KeyError):
            return False
        return True

    def points(self, sid):
        try:
            with np.load(self.root / 'samples' / sid / 'data_short.npz') as d:
                a = d['a1_3_1_all_candidate_xy'].astype(float); role = d['a1_3_1_all_candidate_role']
            p = a[role != 4]
        except (OSError, KeyError, ValueError):
            return None
        try:
            with np.load(self.root / 'far_points.npz') as f:
                if sid in f.files and len(f[sid]):
                    p = np.vstack([p, f[sid].astype(float)])
        except (OSError, ValueError, KeyError):
            pass
        return p


FLOOR_GAP_MIN = 0.03        # 2026-11-13 esy v2: accumulated floor order is used only when its 1st floor leads the 2nd by this much (offline 83 points: 75 -> 86 % right)
# 2026-11-14 esy v2b: the accumulated floor order overrides the leader ONLY when this file exists (field 11-14: coarse rounds take ~50 s on the robot, 2 rounds by 2 min, the override delayed the right floor on 2F/1F by ~48 s) -> default OFF, kci_floor_acc is still recorded by E2
KCI_RANK_ON = (Path.home() / "kci_loadtest_20261113" / "e2_rank.on").exists()
# 2026-11-14 esy v2c: 구조만으로 하는 '확정' 판정(기록만, 순위는 안 바꾼다). 살아 있는 후보(ACTIVE + 근거 프레임 N_EVID 개 이상)가 모두 같은 층이고
# 1위 후보에서 KCI_CONF_R_M 안에 모여 있는 상태로 KCI_CONF_M 만큼 걸으면 그 층을 확정으로 적는다(e3_state 의 kci_confirm). 현장 16건을 이 파일로 재생(11-14):
# 10 m 맞음 8 · 틀림 1 · 없음 7 / 12 m 7·1·8 / 15 m 6·0·10 / 20 m 5·0·11. 틀린 층에 모여 있던 가장 긴 구간이 13.9 m 여서 15 m 로 둔다(여유 1.1 m, 같은 16건으로 고른 값 — 새 기록으로 다시 볼 것).
KCI_CONF_M, KCI_CONF_R_M = 15.0, 3.0
WIN_PATH_M, WIN_EVERY = 30.0, 2   # 2026-11-13 esy v2: physical model — the hypothesis' CURRENT transform must not put the last 30 m of path through a wall to its other side


def wall_pass_any(wall, track_px, ppm, ratio=3.0, extra_m=2.0, max_runs=60):
    """True when the path enters a wall and leaves it on the other side: the cells just before and just after a wall run are not connected
    through free space inside a window around them (same meaning as loc_worker.wall_passes, 10-13; fast version, misses a few, no false alarms
    in 226 offline cases)."""
    from scipy import ndimage
    H, W = wall.shape
    t = np.asarray(track_px, float)
    if len(t) < 2:
        return False
    seg = np.diff(t, axis=0)
    n = np.maximum(np.ceil(np.hypot(seg[:, 0], seg[:, 1])).astype(int), 1)
    P = np.vstack([t[i] + seg[i] * (np.arange(n[i])[:, None] / n[i]) for i in range(len(seg))] + [t[-1:]])
    P = np.rint(P).astype(int)
    P = P[(P[:, 0] >= 0) & (P[:, 0] < W) & (P[:, 1] >= 0) & (P[:, 1] < H)]
    if len(P) < 3:
        return False
    inw = np.asarray(wall[P[:, 1], P[:, 0]], bool)
    if not inw.any():
        return False
    d = np.diff(np.r_[0, inw.astype(np.int8), 0])
    k = 0
    for i, j in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if i == 0 or j >= len(P):
            continue
        k += 1
        if k > max_runs:
            break
        (ax, ay), (bx, by) = P[i - 1], P[j]
        eu = float(np.hypot(ax - bx, ay - by))
        lim = int(np.ceil(max(ratio * eu, eu + extra_m * ppm)))
        x0, y0 = max(0, min(ax, bx) - lim), max(0, min(ay, by) - lim)
        x1, y1 = min(W, max(ax, bx) + lim + 1), min(H, max(ay, by) + lim + 1)
        lab, _ = ndimage.label(~np.asarray(wall[y0:y1, x0:x1], bool), structure=np.ones((3, 3)))
        if lab[ay - y0, ax - x0] != lab[by - y0, bx - x0]:
            return True
    return False


class Hypothesis:
    _n = 0
    FLOOR_RANK = None                       # 2026-11-13 esy: set by Tracker.intake from e2_state kci_floor_acc (needs >= 2 rounds on all 9 floors)

    def __init__(self, rec, light_dir, session):
        Hypothesis._n += 1
        self.id = f"H{Hypothesis._n:03d}_F{rec['floor']}"
        self.rec = dict(rec)
        self.floor = int(rec['floor'])
        self.light_dir = light_dir
        self.session = session
        self.status = 'ACTIVE'
        self.status_why = 'E2_CANDIDATE'
        self.e2_merit, self.e2_twoway = float(rec['merit']), rec.get('twoway')
        self.fit_t = float(rec['fit_t'])
        self.passes = [rec.get('pass_id')]
        self.stage = rec.get('stage') or 'exact'          # 'coarse' until an exact E2 candidate of the same place merges in
        self.pass_keys = [(rec.get('pass_id'), self.stage)]
        self.ev = deque(maxlen=N_KF)
        self.used = set()
        self.n_kf = 0
        self.viol = deque(maxlen=5)
        self.worse_n = self.better_n = 0
        self.last_kf_t, self.last_kf_px = self.fit_t, None
        self.pose = None
        self.pose_t = self.pose_wall = None      # odom time / wall time of the last propagation (odometry only)
        self.last_kf_wall = None                 # wall time of the last geometry (LiDAR keyframe) evaluation
        self.created_wall = time.time()
        self.history = deque(maxlen=200)
        self.obs = deque(maxlen=TRK_N)          # (x, y, yaw, pts_subsample) of the last keyframes, for tracking
        self.corr = dict(n=0, xy_m=0.0, deg=0.0, scale=0.0, saturated=0)
        self.sem_verdict, self.sem = None, []

    def score(self):
        s = [e['frame_merit'] for e in self.ev]
        return (W_E2 * self.e2_merit + sum(s)) / (W_E2 + len(s))

    def evidence(self):
        return 'EVIDENCE' if self.n_kf >= N_EVID else 'PENDING_EVIDENCE'

    def rank_key(self):
        """Evidence-backed hypotheses first (by S), then pending ones (by S): a fresh E2 candidate whose S is still only
        its (optimistic, window-fitted) E2 merit cannot push out a hypothesis verified on LiDAR keyframes.
        A decisive E6 semantic verdict (e6_semantic: room label read with p >= SEM_DECISIVE_P) orders within each
        evidence group: consistent < no reading < place mismatch < floor mismatch. Reversible (latest reading wins)."""
        k = (1 if getattr(self, 'win_pass', False) else 0, 0 if self.n_kf >= N_EVID else 1, SEM_ORDER.get(self.sem_verdict, 1), self.score())
        fr = Hypothesis.FLOOR_RANK               # 2026-11-13 esy: floor order from E2's accumulated coarse scores first (None = original order)
        return ((fr.index(self.floor) if self.floor in fr else len(fr)),) + k if fr else k

    def recent(self, n=5):
        s = [e['frame_merit'] for e in list(self.ev)[-n:]]
        return None if not s else float(np.mean(s))

    def summary(self, best=None):
        s = self.score()
        tw = [e['twoway'] for e in self.ev if e['twoway'] is not None]
        return dict(id=self.id, floor=self.floor, status=self.status, why=self.status_why, score=round(s, 4), evidence=self.evidence(),
                    gap_to_best=None if best is None else round(s - best, 4), e2_merit=round(self.e2_merit, 4),
                    e2_twoway=self.e2_twoway, n_kf=self.n_kf, recent=None if self.recent() is None else round(self.recent(), 4),
                    twoway_mean=None if not tw else round(float(np.mean(tw)), 4), violations=int(sum(self.viol)),
                    pose_px=None if self.pose is None else [round(self.pose[0], 1), round(self.pose[1], 1)],
                    pose_m=None if self.pose is None else [round(self.pose[0] / self.rec['ppm'], 2),
                                                           round((self.rec['H'] - self.pose[1]) / self.rec['ppm'], 2)],
                    heading_deg=None if self.pose is None else round(math.degrees(self.pose[2]), 1),
                    ppm=self.rec['ppm'], phi=self.rec['phi'], beta_deg=self.rec.get('beta_deg'), fit_t=self.fit_t,
                    passes=self.passes, map_version=self.rec.get('map_version'), odom_session=self.session,
                    last_kf_t=self.last_kf_t, provenance=self.rec.get('provenance') or ('E2_REGIONAL' if self.rec.get('regional_request') else 'E2'),
                    anchor_evidence=self.rec.get('evidence'), **self.contract(s, best), tracking={k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.corr.items()},
                    verified=False, kind='UNVERIFIED_HYPOTHESIS')


    def contract(self, s, best):
        """P2 record contract. global_pose is ALWAYS odometry-propagated from the hypothesis transform; it counts as
        geometry-checked only when it is the pose of the newest LiDAR keyframe that was scored. uncertainty is a spread of
        the student costs (metres), not a covariance and not a probability."""
        fm = [e['frame_merit'] for e in self.ev]
        fresh = self.pose_t is not None and self.n_kf > 0 and abs(self.pose_t - self.last_kf_t) < 1e-6
        return dict(hypothesis_id=self.id, floor_id=self.floor, geometry_score=round(s, 4),
                    global_pose=None if self.pose is None else dict(
                        floor=self.floor, px=round(self.pose[0], 1), py=round(self.pose[1], 1),
                        heading_deg=round(math.degrees(self.pose[2]), 1), odom_t=self.pose_t,
                        source='GEOMETRY_CHECKED_KEYFRAME' if fresh else 'ODOM_PROPAGATED'),
                    geometry=dict(n_keyframes=self.n_kf, last_keyframe_t=self.last_kf_t if self.n_kf else None,
                                  last_keyframe_wall=self.last_kf_wall, e2_only=self.n_kf == 0,
                                  odom_s_since_keyframe=None if self.pose_t is None or not self.n_kf else round(self.pose_t - self.last_kf_t, 2)),
                    uncertainty=dict(kind='COST_SPREAD_NOT_COVARIANCE',
                                     frame_merit_std_m=None if len(fm) < 2 else round(float(np.std(fm)), 4),
                                     gap_to_best_m=None if best is None else round(s - best, 4),
                                     tracking_shift_m=round(self.corr['xy_m'], 3)),
                    e2_stage=self.stage, semantic=dict(verdict=self.sem_verdict, last=self.sem[-1] if self.sem else None, n=len(self.sem),
                                                       semantic_score=(self.sem[-1] or {}).get("semantic_score") if self.sem else None,
                                                       observed_room_id=(self.sem[-1] or {}).get("observed_room_id") if self.sem else None,
                                                       map_room_id=(self.sem[-1] or {}).get("map_room_id") if self.sem else None,
                                                       update_timestamp=(self.sem[-1] or {}).get("wall") if self.sem else None,
                                                       supporting_evidence=[x.get("observation_id") for x in self.sem if x.get("decisive")]),
                    supporting_observations=dict(e2_passes=[p for p in self.passes if p is not None], keyframes=self.n_kf,
                                                 samples=len(self.used)),
                    last_update_timestamp=self.pose_wall)


class Tracker:
    def __init__(self, store_root, e2_dir, log_path=None, clock=time.time):
        self.store = Store(store_root)
        self.e2_dir = Path(e2_dir)
        self.clock = clock
        self.H = []
        self.dropped = deque(maxlen=200)
        self.lights = {}
        self.session = None
        self.kf = []                     # (sid, t, x, y, yaw) keyframes of the current session
        self.seen_pass = set()
        self.pass_floor_seen = set()
        self.log_path = Path(log_path) if log_path else None
        self.timing = dict(last_kf_ms=None, kf_ms=deque(maxlen=200), last_tick_ms=None, evals=0)
        self.events = deque(maxlen=50)
        self.last_eval_wall = None
        self.rr = 0
        self.request = None
        self.leader, self.leader_streak = None, 0
        self.low_sep, self.deferred, self.geometry = 0, 0, None

    # ---------- helpers ----------
    def light(self, d):
        if d not in self.lights:
            self.lights[d] = LightMap(d)
        return self.lights[d]

    def ev(self, kind, **k):
        rec = dict(wall=self.clock(), kind=kind, **k)
        self.events.append(rec)
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')

    def alive(self):
        return [h for h in self.H if h.status in ('ACTIVE', 'SUSPENDED')]

    # ---------- session & keyframes ----------
    def check_session(self):
        s = self.store.session()
        sid = None if s is None else s.get('odom_session_id')
        if sid != self.session:
            if self.session is not None:
                for h in self.alive():
                    h.status, h.status_why = 'DROPPED', 'ODOM_SESSION_CHANGED'
                    self.dropped.append(h.summary())
                self.ev('session_changed', old=self.session, new=sid, invalidated=len(self.H))
                self.H = []
                self.request = dict(why='ODOM_SESSION_CHANGED', wall=self.clock())
            self.session = sid
            self.kf = []
            self.seen_pass, self.pass_floor_seen = set(), set()
        return sid

    def update_keyframes(self):
        self.store.refresh()
        last = self.kf[-1] if self.kf else None
        for r in self.store.rows:
            if last is not None and r[1] <= last[1]:
                continue
            if last is None or math.hypot(r[2] - last[2], r[3] - last[3]) >= KF_MOVE_M or ang_deg(r[4], last[4]) >= KF_TURN_DEG:
                self.kf.append(r); last = r
        if len(self.kf) > 3000:
            self.kf = self.kf[-3000:]

    def rows_between(self, t0, t1):
        return [r for r in self.store.rows if t0 <= r[1] <= t1]

    # ---------- E2 intake ----------
    def region_ack(self, rid, hyp_id, how):
        """R34: tell the E2 coordinator its regional candidates reached E3 (delivery status of the request ledger)."""
        ack = getattr(self, '_region_ack', None)
        if ack is None:
            ack = self._region_ack = {}
        a = ack.setdefault(rid, dict(delivered_wall=time.time(), added=0, merged=0, hypotheses=[]))
        a[how] += 1
        a['hypotheses'].append(hyp_id)
        try:
            p = Path(self.e2_dir) / 'e2_region_ack.json'
            tmp = p.with_name(f'.{p.name}.{os.getpid()}.tmp')
            tmp.write_text(json.dumps(ack, default=str))
            os.replace(tmp, p)
        except OSError as ex:
            self.ev('region_ack_failed', error=str(ex))

    def intake_anchored(self):
        """R34: SEMANTIC_ANCHORED candidates (semantic_anchor.py) enter as provisional hypotheses of this odometry session. Added to the E2 hypotheses, never instead of them."""
        try:
            doc = json.loads((self.e2_dir / 'semantic_anchor.json').read_text())
        except (OSError, ValueError):
            return 0
        added = 0
        for rec in doc.get('candidates') or []:
            key = (rec.get('pass_id'), 'anchored', rec['floor'], round(rec['px'], 1), round(rec['py'], 1), round(rec['phi_deg'], 2))
            if key in self.pass_floor_seen or rec.get('odom_session') != self.session or not rec.get('light_dir'):
                continue
            self.pass_floor_seen.add(key)
            h_new = Hypothesis(rec, rec['light_dir'], self.session)
            dup = self.find_dup(h_new)
            if dup is not None:
                dup.rec.setdefault('also_anchored_by', []).append(rec.get('evidence'))
                dup.pass_keys.append((rec.get('pass_id'), 'anchored'))
                self.ev('merged', into=dup.id, pass_id=rec.get('pass_id'), stage='anchored', floor=rec['floor'])
                Hypothesis._n -= 1
                continue
            h_new.sem_verdict = "SEMANTIC_CONSISTENT"       # R36: built from that reading; the next decisive reading can replace it
            self.H.append(h_new); added += 1
            self.ev('intake_anchored', id=h_new.id, floor=h_new.floor, evidence=rec.get('evidence'))
        if added:
            self.cap()
        return added

    def intake(self, e2):
        """E2 candidates -> hypotheses. COARSE candidates (approximate all-floor stage, stage='coarse') enter too, so every
        floor gets LiDAR keyframe evidence early; the exact candidate of the same place later merges into it (one ledger)."""
        if not e2 or e2.get('odom_session') != self.session or self.session is None:
            Hypothesis.FLOOR_RANK = None
            return 0
        acc = e2.get('kci_floor_acc') or {}
        nn = acc.get('n') or {}
        mean = acc.get('mean') or {}
        rk = list(acc.get('rank') or [])
        gap = (float(mean.get(str(rk[1]), 0)) - float(mean.get(str(rk[0]), 0))) if len(rk) >= 2 else 0.0
        Hypothesis.FLOOR_RANK = rk if (KCI_RANK_ON and rk and acc.get('session') == self.session and len(nn) >= 9 and min(nn.values()) >= 2 and gap >= FLOOR_GAP_MIN) else None
        added = 0
        pf_exact = e2.get('per_floor') or {}
        pf_coarse = (e2.get('coarse') or {}).get('per_floor') or {}
        for rec in (e2.get('candidates_coarse') or []) + (e2.get('candidates_refresh') or []) + (e2.get('candidates_pass') or []):
            stage = rec.get('stage') or 'exact'
            key = (rec.get('pass_id'), stage, rec['floor'], round(rec['px'], 1), round(rec['py'], 1), round(rec['phi_deg'], 2))
            if key in self.pass_floor_seen:
                continue
            pf = (pf_exact if stage == 'exact' else pf_coarse).get(str(rec['floor']), {})
            light = pf.get('light')
            if not light:
                continue
            self.pass_floor_seen.add(key)
            rec = dict(rec, H=pf.get('H'), stage=stage)
            h_new = Hypothesis(rec, light, self.session)
            rid = rec.get('regional_request')
            dup = self.find_dup(h_new)
            if dup is not None:
                if stage == 'exact' or dup.stage == 'coarse':      # an exact E2 score replaces a coarse one, never the reverse
                    dup.e2_merit, dup.e2_twoway = h_new.e2_merit, h_new.e2_twoway
                if stage == 'exact':
                    dup.stage = 'exact'
                dup.passes.append(rec.get('pass_id')); dup.pass_keys.append((rec.get('pass_id'), stage))
                self.ev('merged', into=dup.id, pass_id=rec.get('pass_id'), stage=stage, floor=rec['floor'])
                Hypothesis._n -= 1
                if rid:
                    self.region_ack(rid, dup.id, 'merged')
                continue
            self.H.append(h_new); added += 1
            if rid:
                self.region_ack(rid, h_new.id, 'added')
        if added:
            self.ev('intake', added=added, total=len(self.H), pass_id=e2.get('pass_id'))
            self.cap()
        return added

    def find_dup(self, h_new):
        """Same floor, propagated pose within MERGE_M and heading within MERGE_DEG at the new candidate's fit time."""
        r = self.nearest_row(h_new.fit_t)
        if r is None:
            return None
        pn = pose_in_map(h_new.rec, r[2], r[3], r[4])
        for h in self.alive():
            if h.floor != h_new.floor or (h_new.rec.get('pass_id'), h_new.stage) in h.pass_keys:   # E2 de-duplicates within a stage
                continue
            po = pose_in_map(h.rec, r[2], r[3], r[4])
            if math.hypot(pn[0] - po[0], pn[1] - po[1]) / h.rec['ppm'] < MERGE_M and ang_deg(pn[2], po[2]) < MERGE_DEG:
                return h
        return None

    def nearest_row(self, t):
        if not self.store.rows:
            return None
        return min(self.store.rows, key=lambda r: abs(r[1] - t))

    def cap(self):
        al = self.alive()
        if len(al) <= CAP_ALL:
            return
        sus = sorted([h for h in al if h.status == 'SUSPENDED'], key=lambda h: -h.score())
        act = sorted([h for h in al if h.status == 'ACTIVE'], key=lambda h: -h.score())
        for h in (sus + act)[:len(al) - CAP_ALL]:
            h.status, h.status_why = 'DROPPED', 'MEMORY_CAP'
            self.dropped.append(h.summary())
            self.ev('dropped', id=h.id, why='MEMORY_CAP', score=h.score())

    # ---------- evaluation ----------
    def eval_one(self, h, kf, prev_kf_t, pts):
        sid, t, x, y, yaw = kf
        L = self.light(h.light_dir)
        ppm, mirror = h.rec['ppm'], h.rec['mirror']
        px, py, hd = pose_in_map(h.rec, x, y, yaw)
        p = pts
        if len(p) > 400:
            p = p[np.random.RandomState(zlib.crc32(sid.encode()) % 97).choice(len(p), 400, replace=False)]
        mp = np.array([px, py]) + ppm * ((p * [1, mirror]) @ _A(hd).T)
        big = float(max(L.H, L.W))
        err = np.minimum(L.sample(L.F, mp, big) / ppm, 5.0)
        cost = float(np.mean(np.sort(err)[:max(1, int(.7 * len(err)))]))
        T2 = float(np.mean(err > FAR_M))
        T3 = float(np.mean(L.sample(L.out, mp, big) / ppm > OUT_M))
        seg = [pose_in_map(h.rec, r[2], r[3], r[4])[:2] for r in self.rows_between(prev_kf_t, t)] + [(px, py)]
        X = L.crossing_m(np.asarray(seg, float), ppm)
        trav = L.traversable(px, py)
        tw = None
        try:
            import twoway as TW
            d_obs = TW.sector_obs(np.asarray(pts, float) * [1, mirror], hd)
            tw = float(TW.twoway_score(L.wall_real, px, py, ppm, d_obs)[0])
        except Exception:
            tw = None
        fm = cost + W_FREE * T2 + W_OUT * T3 + W_CROSS * min(X, X_HARD) + W_TRAV * (0.0 if trav else 1.0)
        h.obs.append((x, y, yaw, p))
        hard = X > X_HARD
        h.win_n = getattr(h, 'win_n', 0) + 1
        if h.win_n % WIN_EVERY == 1 or not hasattr(h, 'win_pass'):
            try:
                rows = self.store.rows
                i1 = len(rows) - 1
                while i1 > 0 and rows[i1][1] > t:
                    i1 -= 1
                i0, dist = i1, 0.0
                while i0 > 0 and dist < WIN_PATH_M:
                    dist += math.hypot(rows[i0][2] - rows[i0 - 1][2], rows[i0][3] - rows[i0 - 1][3]); i0 -= 1
                path = [pose_in_map(h.rec, r[2], r[3], r[4])[:2] for r in rows[i0:i1 + 1]]
                h.win_pass = bool(len(path) > 1 and wall_pass_any(L.wall_tr, path, ppm))
            except Exception:
                h.win_pass = False
        if getattr(h, 'win_pass', False):        # 2026-11-13 esy v2: physical model — counts as a hard violation (2 of the last 5 -> SUSPENDED)
            fm += W_CROSS * (X_HARD - min(X, X_HARD))
            hard = True
        return dict(sid=sid, t=t, cost=cost, T2=T2, T3=T3, X=X, trav=trav, twoway=tw, frame_merit=fm, inl10=float(np.mean(err < .1)),
                    pose=(px, py, hd), hard=hard, win_pass=bool(getattr(h, 'win_pass', False)))

    def track(self, h):
        """Bounded re-alignment of h on its last TRK_N keyframes; re-anchors h.rec at the newest one."""
        if len(h.obs) < 2:
            return
        L = self.light(h.light_dir)
        rec = h.rec
        x0, y0, yaw0, _ = h.obs[-1]
        px0, py0, _ = pose_in_map(rec, x0, y0, yaw0)
        mirror, beta = rec['mirror'], rec['beta']
        big = float(max(L.H, L.W))
        obs = [(x, y, yaw, p[np.linspace(0, len(p) - 1, min(150, len(p))).astype(int)] * [1, mirror]) for (x, y, yaw, p) in h.obs]

        def cost(combos):
            K = len(combos)
            tot = np.zeros(K)
            for (x, y, yaw, q) in obs:
                d_od = np.array([x - x0, y - y0])
                phi = np.deg2rad(rec['phi']) + combos[:, 2]
                ppm = rec['ppm'] * combos[:, 3]
                c, s_ = np.cos(phi), np.sin(phi)
                off = np.stack([c * d_od[0] - s_ * d_od[1], -s_ * d_od[0] - c * d_od[1]], 1) * ppm[:, None]   # A(phi) d_od
                hd = phi + yaw + beta
                ch, sh = np.cos(hd), np.sin(hd)
                mx = (q[:, 0][None] * ch[:, None] - q[:, 1][None] * sh[:, None]) * ppm[:, None]
                my = (-q[:, 0][None] * sh[:, None] - q[:, 1][None] * ch[:, None]) * ppm[:, None]
                X = px0 + combos[:, 0:1] * rec['ppm'] + off[:, 0:1] + mx
                Y = py0 - combos[:, 1:2] * rec['ppm'] + off[:, 1:2] + my
                e = np.minimum(L.sample(L.F, np.c_[X.ravel(), Y.ravel()], big).reshape(K, -1) / ppm[:, None], 5.0)
                tot += np.sort(e, 1)[:, :max(1, int(.7 * e.shape[1]))].mean(1)
            return tot
        dxy = np.array(TRK_XY_M)
        # coarse-to-fine: translation first (25), then rotation x scale at the best translation (15) -> 40 evaluations
        c1 = np.array([(a, b, 0.0, 1.0) for a in dxy for b in dxy])
        k1 = cost(c1)
        a_, b_ = c1[int(np.argmin(k1))][:2]
        c2 = np.array([(a_, b_, t, 1.0 + s_) for t in np.deg2rad(TRK_DEG) for s_ in TRK_SCALE])
        combos = np.vstack([c1, c2])
        best_cost = np.r_[k1, cost(c2)]
        p0 = h.__dict__.setdefault('ppm0', float(rec['ppm']))     # R42: the scale E2 fitted; tracking may creep, not wander (X20261103_003729 H026_F4 reached 54.22 px/m, outside E2's whole range; 4F leader 33.09)
        best_cost = np.where(np.abs(np.log(rec['ppm'] * combos[:, 3] / p0)) <= math.log(1.0 + TRK_SCALE_CUM) + 1e-9, best_cost, np.inf)
        i = int(np.argmin(best_cost))
        ident = int(np.where((combos[:, :3] == 0).all(1) & (combos[:, 3] == 1.0))[0][0])
        a, b, t, s_ = combos[i]
        if self.log_path:          # 2026-10-21: EXP1 H032 drifted 0.91 -> 6.0 m with 55/85 saturated corrections; the direction
            try:                   # could not be analysed afterwards -> one line per tracking decision (separate file, no UI list)
                with open(self.log_path.with_name("e3_track.jsonl"), "a") as f:
                    f.write(json.dumps(dict(wall=round(self.clock(), 3), id=h.id, floor=h.floor, n_obs=len(h.obs), a_m=float(a), b_m=float(b),
                                            dphi_deg=round(math.degrees(t), 2), scale=float(s_), heading_rad=round(float(yaw0 + np.deg2rad(rec['phi']) + beta), 4),
                                            cost_best=round(float(best_cost[i]), 4), cost_identity=round(float(best_cost[ident]), 4),
                                            applied=bool(best_cost[i] < best_cost[ident] - 1e-9))) + "\n")
            except OSError:
                pass
        if not np.isfinite(best_cost[i]) or best_cost[i] >= best_cost[ident] - 1e-9:           # identity is (one of) the best: nothing to change
            return
        h.rec = dict(rec, pos=[px0 + a * rec['ppm'], py0 - b * rec['ppm']], anchor=[x0, y0], phi=rec['phi'] + math.degrees(t),
                     ppm=rec['ppm'] * s_)
        h.corr['n'] += 1; h.corr['xy_m'] += math.hypot(a, b); h.corr['deg'] += abs(math.degrees(t)); h.corr['scale'] += abs(s_ - 1)
        if abs(a) == max(TRK_XY_M) or abs(b) == max(TRK_XY_M) or abs(math.degrees(t)) == max(TRK_DEG):
            h.corr['saturated'] += 1

    def tick(self):
        """One E3 step: session check, E2 intake, then evaluate up to CATCHUP_PER_TICK new keyframes."""
        t0 = time.perf_counter()
        self.check_session()
        self.update_keyframes()
        e2 = None
        try:
            e2 = json.loads((self.e2_dir / 'e2_state.json').read_text())
        except (OSError, ValueError):
            pass
        self.intake(e2)
        self.intake_anchored()
        n_done = 0
        n_done = self.work_keyframes()
        # current pose of every alive hypothesis (propagation only, no evidence)
        if self.store.rows:
            r = self.store.rows[-1]
            for h in self.alive():
                if r[1] >= h.fit_t - 1e-6:
                    h.pose = pose_in_map(h.rec, r[2], r[3], r[4])
                    h.pose_t, h.pose_wall = r[1], self.clock()
        try:
            self.kci_confirm_update()
        except Exception as ex:                       # 판정이 넘어져도 추적은 그대로 돈다. 쌓인 값은 지우지 않는다
            if isinstance(getattr(self, 'kci_confirm', None), dict):
                self.kci_confirm['error'] = repr(ex)[:200]
            else:
                self.kci_confirm = dict(state='ERROR', error=repr(ex)[:200])
        self.timing['last_tick_ms'] = round(1000 * (time.perf_counter() - t0), 1)
        return n_done

    def work_keyframes(self):
        """R35: order of work unchanged (oldest keyframe first). 11-01 loc1: each new floor adds hypotheses fitted on an old window, and they replayed every keyframe since that window. The score only uses the last N_KF
        keyframes (h.ev), so older ones are recomputed for nothing: a hypothesis behind by more than N_KF starts at the N_KF newest. Same as a deferred keyframe: no evidence, no penalty."""
        kfs = self.kf
        if kfs:
            floor_t = kfs[-(N_KF + 1)][1] if len(kfs) > N_KF else None
            if floor_t is not None:
                for h in self.alive():
                    if h.last_kf_t < floor_t:
                        h.catchup_skipped_s = round(getattr(h, 'catchup_skipped_s', 0.0) + floor_t - h.last_kf_t, 1)
                        h.last_kf_t = floor_t
        pend = self.pending()
        self.backlog_n = len(pend)
        n = 0
        t_b = time.perf_counter()
        for kf in pend[:CATCHUP_PER_TICK]:
            self.eval_kf(kf)
            n += 1
            if time.perf_counter() - t_b > TICK_BUDGET_S:
                break
        return n

    def pending(self):
        """Keyframes some alive hypothesis has not evaluated yet (per hypothesis: after its own last keyframe)."""
        al = self.alive()
        if not al:
            return []
        t_min = min(h.last_kf_t for h in al)
        return [k for k in self.kf if k[1] > t_min and any(k[1] > h.last_kf_t and k[0] not in h.used for h in al)]

    def eval_kf(self, kf):
        t0 = time.perf_counter()
        sid, t = kf[0], kf[1]
        al = [h for h in self.alive() if t > h.last_kf_t and sid not in h.used]
        pts = self.store.points(sid)
        if pts is None or len(pts) < 20:
            for h in al:
                h.used.add(sid); h.last_kf_t = t
            self.ev('kf_skipped', sid=sid, why='NO_POINTS')
            return
        # budget: pending (need evidence) + best actives first, the rest round-robin; nobody is removed, only deferred
        semf = [h for h in al if SEM_FIRST and h.sem_verdict == "SEMANTIC_CONSISTENT"][:EVAL_PER_KF]     # B4: plate-consistent first (any status)
        pend = [h for h in al if h.status == 'ACTIVE' and h.n_kf < N_EVID and h not in semf]
        act = sorted([h for h in al if h.status == 'ACTIVE' and h.n_kf >= N_EVID and h not in semf], key=lambda h: h.score())
        sus = [h for h in al if h.status == 'SUSPENDED' and h not in semf]
        first = (semf + pend + act)[:EVAL_PER_KF]
        rest = [h for h in pend + act if h not in first] + sus
        room = max(0, EVAL_PER_KF - len(first))
        if getattr(self, 'backlog_n', 0) > BACKLOG_KF:
            room = min(room, BACKLOG_ROOM)
        if rest and room:
            self.rr = (self.rr + 1) % max(1, len(rest))
            rest = (rest[self.rr:] + rest[:self.rr])[:room]
        else:
            rest = []
        self.deferred = len(al) - len(first) - len(rest)
        n_eval = len(first) + len(rest)
        for h in first + rest:
            e = self.eval_one(h, kf, h.last_kf_t, pts)
            h.used.add(sid)
            h.ev.append(e); h.n_kf += 1; h.viol.append(1 if e['hard'] else 0)
            h.last_kf_t, h.last_kf_wall = t, self.clock()
            h.history.append((round(t, 2), round(h.score(), 4), h.status))
            self.timing['evals'] += 1
            if h.status == 'ACTIVE' or h.n_kf % 4 == 0 or h in semf:      # suspended hypotheses: every 4th evaluation (CPU bound); B4: plate-consistent ones every time
                self.track(h)
        for h in al:
            if h not in first and h not in rest:
                h.last_kf_t = max(h.last_kf_t, t)          # deferred: this keyframe is skipped for h (no evidence, no penalty)
        self.lifecycle()
        act = [h for h in self.H if h.status == 'ACTIVE']
        lead = min(act, key=lambda h: h.rank_key()).id if act else None
        self.leader_streak = self.leader_streak + 1 if lead == self.leader and lead is not None else 0
        self.leader = lead
        ms = 1000 * (time.perf_counter() - t0)
        self.timing['last_kf_ms'] = round(ms, 1); self.timing['kf_ms'].append(ms)
        self.last_eval_wall = self.clock()
        self.last_kf = dict(sid=sid, t=t, n_eval=n_eval, deferred=self.deferred, ms=round(ms, 1))

    def kci_confirm_update(self):
        """2026-11-14 esy v2c: '후보가 한 층·한 자리로 모인 채 얼마나 걸었나' 로 확정을 적는다. 기록만 한다(순위·상태는 건드리지 않는다)."""
        C = getattr(self, 'kci_confirm', None)
        if not isinstance(C, dict) or C.get('session') != self.session or 'path_m' not in C:
            C = self.kci_confirm = dict(session=self.session, state='NONE', path_m=0.0, gather_floor=None, gather_from_m=0.0, streak_m=0.0,
                                        confirmed=None, first=None, rule=dict(streak_m=KCI_CONF_M, radius_m=KCI_CONF_R_M))
            self._kci_xy = None
        if not self.store.rows:
            return
        r = self.store.rows[-1]
        if self._kci_xy is not None:
            C['path_m'] += math.hypot(r[2] - self._kci_xy[0], r[3] - self._kci_xy[1])
        self._kci_xy = (r[2], r[3])
        act = [h for h in sorted(self.alive(), key=lambda h: h.rank_key()) if h.status == 'ACTIVE' and h.n_kf >= N_EVID and h.pose is not None]
        f, good = None, False
        if act:
            f = int(act[0].floor)
            x0, y0 = act[0].pose[0] / act[0].rec['ppm'], act[0].pose[1] / act[0].rec['ppm']
            good = all(int(h.floor) == f and math.hypot(h.pose[0] / h.rec['ppm'] - x0, h.pose[1] / h.rec['ppm'] - y0) <= KCI_CONF_R_M for h in act)
        if not good:
            C['gather_floor'] = None
        elif C['gather_floor'] != f:
            C['gather_floor'], C['gather_from_m'] = f, C['path_m']
        C['streak_m'] = round(C['path_m'] - C['gather_from_m'], 2) if C['gather_floor'] is not None else 0.0
        C['n_alive_evidence'] = len(act)
        if C['gather_floor'] is not None and C['streak_m'] >= KCI_CONF_M and C['confirmed'] != f:
            C['confirmed'] = f
            ev = dict(floor=f, odom_t=r[1], path_m=round(C['path_m'], 1), hypothesis=act[0].id)
            C['last'] = dict(ev, wall=self.clock())
            if C['first'] is None:
                C['first'] = dict(C['last'])
            C['state'] = 'CONFIRMED'
            self.ev('kci_confirm', **ev)               # ev() 가 wall 을 스스로 붙인다(같은 이름을 넘기면 오류)
        C['state'] = 'CONFIRMED' if C['confirmed'] is not None else ('GATHERING' if C['gather_floor'] is not None else 'NONE')

    def leader_lag(self):
        """Newest keyframe minus the leader's last evaluated keyframe, in odometry seconds (R35: what E4 actually depends on)."""
        if not self.kf:
            return None
        h = next((x for x in self.H if x.id == self.leader), None)
        return None if h is None else round(max(0.0, self.kf[-1][1] - h.last_kf_t), 1)

    def lifecycle(self):
        act = [h for h in self.H if h.status == 'ACTIVE']
        if not act:
            return
        ref = [h for h in act if h.n_kf >= N_EVID] or act          # reference = evidence-backed best (not a fresh E2 merit)
        best = min(h.score() for h in ref)
        for h in self.alive():
            gap = h.score() - best
            if h.status == 'ACTIVE':
                h.worse_n = h.worse_n + 1 if (h.n_kf >= 3 and gap > SUSP_GAP) else 0
                if h.worse_n >= SUSP_N or sum(h.viol) >= 2:
                    h.status = 'SUSPENDED'
                    h.status_why = f'GAP>{SUSP_GAP} x{h.worse_n}' if h.worse_n >= SUSP_N else 'HARD_VIOLATIONS>=2/5'
                    h.better_n = 0
                    self.ev('suspended', id=h.id, why=h.status_why, score=round(h.score(), 4), best=round(best, 4))
            else:
                h.better_n = h.better_n + 1 if (gap < REACT_GAP and sum(h.viol) < 2) else 0
                if h.better_n >= REACT_N:
                    h.status, h.status_why, h.worse_n = 'ACTIVE', 'RECOVERED', 0
                    self.ev('reactivated', id=h.id, score=round(h.score(), 4), best=round(best, 4))
        self.geometry_check()

    def apply_semantic(self, result, observation_id, source):
        """E6 semantic result (e6_semantic.evaluate) -> per-hypothesis verdict. Only a decisive reading changes ranking;
        nothing is deleted. source (AUTO_E5 / OPERATOR_ASSISTED) is kept on every record."""
        alts = result.get("alternatives") or []
        decisive = bool(alts) and max(p for _, p in alts) >= SEM_DECISIVE_P
        n = 0
        for h in self.alive():
            v = (result.get("per_hypothesis") or {}).get(h.id)
            if not v:
                continue
            h.sem.append(dict(v, observation_id=observation_id, source=source, decisive=decisive, wall=self.clock()))
            if decisive:
                h.sem_verdict = v["verdict"]
                n += 1
        self.ev("semantic", observation_id=observation_id, source=source, decisive=decisive, n_hyp=n,
                labels=[a[0] for a in alts], floor_likelihood=result.get("floor_likelihood"))
        return n

    def geometry_check(self):
        """GEOMETRY_INCONSISTENT request (re-search) when the best recent misfit stops separating from the crowd for SEP_N
        keyframes, or nothing fits (best recent > ABS_FAIL_M). Not raised on the absolute clutter level alone."""
        rec = [h.recent() for h in self.alive() if h.n_kf >= GEO_MIN_KF and h.recent() is not None]
        g = dict(n=len(rec), best_recent=None, crowd_median_recent=None, separation=None, low_sep_streak=self.low_sep)
        if rec:
            best, med = min(rec), float(np.median(rec))
            g.update(best_recent=round(best, 3), crowd_median_recent=round(med, 3), separation=round(med - best, 3))
            low = len(rec) >= CROWD_MIN and med - best < SEP_MIN_M
            self.low_sep = self.low_sep + 1 if low else 0
            g['low_sep_streak'] = self.low_sep
            why = None
            if best > ABS_FAIL_M:
                why = f'NOTHING_FITS best recent {best:.2f} > {ABS_FAIL_M} m'
            elif self.low_sep >= SEP_N:
                why = f'NO_SEPARATION median-best {med - best:.2f} < {SEP_MIN_M} m for {self.low_sep} keyframes'
            if why:
                self.request = dict(why='GEOMETRY_INCONSISTENT', detail=why, wall=self.clock(), best_recent=round(best, 3),
                                    separation=round(med - best, 3))
        self.geometry = g

    def snapshot(self):
        al = self.alive()
        act = [h for h in al if h.status == 'ACTIVE']
        ref = [h for h in act if h.n_kf >= N_EVID] or act
        best = min((h.score() for h in ref), default=None)
        km = list(self.timing['kf_ms'])
        return dict(session=self.session, leader=self.leader, leader_streak_kf=self.leader_streak, n_alive=len(al), n_active=len(act), n_suspended=len(al) - len(act),
                    n_dropped=len(self.dropped), hypotheses=[dict(h.summary(best), rank=i + 1) for i, h in enumerate(sorted(al, key=lambda h: h.rank_key()))],
                    dropped=list(self.dropped)[-20:], n_keyframes=len(self.kf), kf_pending=len(self.pending()),
                    last_kf=getattr(self, 'last_kf', None), last_eval_wall=self.last_eval_wall, leader_kf_lag_s=self.leader_lag(),
                    timing=dict(last_tick_ms=self.timing['last_tick_ms'], last_kf_ms=self.timing['last_kf_ms'], evals=self.timing['evals'],
                                kf_ms_p50=None if not km else round(float(np.median(km)), 1),
                                kf_ms_p95=None if not km else round(float(np.percentile(km, 95)), 1)),
                    events=list(self.events)[-12:], request=self.request, geometry=getattr(self, 'geometry', None),
                    deferred_last_kf=getattr(self, 'deferred', 0),
                    kci_confirm=getattr(self, 'kci_confirm', None),
                    params=dict(KF_MOVE_M=KF_MOVE_M, KF_TURN_DEG=KF_TURN_DEG, W_E2=W_E2, N_KF=N_KF, SUSP_GAP=SUSP_GAP, SUSP_N=SUSP_N,
                                REACT_GAP=REACT_GAP, REACT_N=REACT_N, MERGE_M=MERGE_M, CAP_ALL=CAP_ALL, BUDGET_PER_KF=BUDGET_PER_KF))
