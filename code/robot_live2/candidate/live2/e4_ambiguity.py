"""e4_ambiguity.py — LIVE2 E4: ambiguity state from the REAL E2/E3 hypotheses. No synthetic candidate, no probability.

States (first matching rule wins):
  RECOVERY_REQUIRED   odom session changed under existing hypotheses, odometry / feed input stale, every E2 floor failed,
                      or none left after a full search
  GEOMETRY_INCONSISTENT  E3 request: the best hypothesis no longer separates from the crowd, or nothing fits (relative
                      test; the absolute clutter level of the proxy input alone is not a failure)
  INITIALIZING        no odom session yet, store shorter than the E2 window, or no floor result yet
  SEARCH_INCOMPLETE   the first building pass has not covered every floor (dominance would be premature)
  AMBIGUOUS           a DISTINCT competitor (other floor, > SEP_M apart, or heading > SEP_DEG) within GAP_AMB of the best,
                      or >= 2 floors within GAP_AMB
  CONFIDENT_GEOMETRY  gap >= GAP_CONF, the same leader for >= LEADER_KF_MIN keyframes, best has >= KF_MIN keyframes and >= PATH_MIN_M of path since its E2 window,
                      recent wall cost <= COST_OK, no hard violation in its recent keyframes, inputs fresh.
                      Geometry only: E6/E7 are not connected, so this is never shown as a final pose.
  DOMINANT_UNVERIFIED otherwise (a clear leader without enough independent evidence, or without a semantic check)
Scores are E3 S values (metres of wall misfit + student penalties, lower = better). Thresholds are first engineering
values (not tuned on ground truth) and are written into every output so a reader can judge them.
"""
import json
import math
from pathlib import Path

import numpy as np

GAP_AMB, GAP_CONF, SEP_M, SEP_DEG = 0.05, 0.15, 2.0, 30.0
KF_MIN, PATH_MIN_M, COST_OK, LEADER_KF_MIN = 8, 8.0, 0.35, 15
STALE_ODOM_S, STALE_FEED_S = 2.0, 6.0
R_OBS_M, FOV_HALF_DEG, NEAR_OBS_M = 8.0, 34.5, 6.0
THRESHOLDS = dict(GAP_AMB=GAP_AMB, GAP_CONF=GAP_CONF, SEP_M=SEP_M, SEP_DEG=SEP_DEG, KF_MIN=KF_MIN, PATH_MIN_M=PATH_MIN_M,
                  COST_OK=COST_OK, LEADER_KF_MIN=LEADER_KF_MIN, STALE_ODOM_S=STALE_ODOM_S, STALE_FEED_S=STALE_FEED_S, tuned_on_ground_truth=False)


def _sep(a, b):
    if a['floor'] != b['floor']:
        return True, 'OTHER_FLOOR'
    if a['pose_px'] is None or b['pose_px'] is None:
        return True, 'POSE_UNKNOWN'
    d = math.hypot(a['pose_px'][0] - b['pose_px'][0], a['pose_px'][1] - b['pose_px'][1]) / a['ppm']
    if d > SEP_M:
        return True, f'{d:.1f}m_APART'
    if a['heading_deg'] is not None and b['heading_deg'] is not None:
        dh = abs((a['heading_deg'] - b['heading_deg'] + 180) % 360 - 180)
        if dh > SEP_DEG:
            return True, f'HEADING_{dh:.0f}deg'
    return False, 'SAME_PLACE'


def path_since(rows, t0):
    xy = [(r[2], r[3]) for r in rows if r[1] >= t0]
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(xy, xy[1:])))


def classify(e2, e3, feed, lio, rows, now):
    """-> dict(state, reasons[], metrics{}, top[]); every input may be None (then the honest state says so)."""
    reasons, m = [], {}
    odom_age = None if not lio else (lio.get('odom') or {}).get('last_rx_age_s')
    feed_last = ((feed or {}).get('last') or {})
    feed_age = None if not feed_last.get('wall') else now - float(feed_last['wall'])
    m.update(odom_last_rx_age_s=odom_age, feed_last_sample_age_s=None if feed_age is None else round(feed_age, 2))
    act_all = [h for h in (e3 or {}).get('hypotheses', []) if h['status'] == 'ACTIVE']
    backed = [h for h in act_all if h.get('evidence', 'EVIDENCE') == 'EVIDENCE']
    hyps = backed or act_all            # PENDING_EVIDENCE hypotheses never decide leader / competitor while backed ones exist
    m.update(n_pending_evidence=len(act_all) - len(backed), geometry=(e3 or {}).get('geometry'))
    allh = (e3 or {}).get('hypotheses', [])
    floors_total = (e2 or {}).get('floors_total')
    done = (e2 or {}).get('floors_done') or []
    failed = (e2 or {}).get('floors_failed') or {}
    first_complete = bool(e2) and ((e2.get('pass_id') or 0) > 1 or (e2.get('status') in ('DONE', 'IDLE') and e2.get('pass_id', 0) >= 1)
                                   or (floors_total and len(done) + len(failed) >= floors_total))
    m.update(floors_total=floors_total, floors_done=len(done), floors_failed=len(failed), e2_pass_id=(e2 or {}).get('pass_id'),
             e2_status=(e2 or {}).get('status'), n_active=len(hyps), n_alive=len(allh), search_complete=first_complete)
    req = (e3 or {}).get('request') or {}

    # ---------- RECOVERY_REQUIRED ----------
    rec = []
    if e3 and any(ev.get('kind') == 'session_changed' and now - ev.get('wall', 0) < 30 for ev in e3.get('events', [])):
        rec.append('ODOM_SESSION_CHANGED: hypotheses of the old FAST-LIO session were invalidated')
    if odom_age is None and (e3 or {}).get('session'):
        rec.append('ODOMETRY_UNAVAILABLE')
    elif odom_age is not None and odom_age > STALE_ODOM_S:
        rec.append(f'ODOMETRY_STALE {odom_age:.1f}s > {STALE_ODOM_S}s')
    if feed_age is not None and feed_age > STALE_FEED_S:
        rec.append(f'E2_FEED_STALE {feed_age:.1f}s > {STALE_FEED_S}s')
    if floors_total and len(failed) >= floors_total:
        rec.append('E2_ALL_FLOORS_FAILED')
    if req.get('why') == 'ALL_ACTIVE_DEGRADED' and now - req.get('wall', 0) < 30:
        rec.append(f"ALL_ACTIVE_DEGRADED best recent frame merit {req.get('best_recent')} m")
    if first_complete and not hyps and e3 is not None:
        rec.append('NO_ACTIVE_HYPOTHESIS_AFTER_FULL_SEARCH')
    if rec:
        return _out('RECOVERY_REQUIRED', rec, m, hyps, rows, now)
    if req.get('why') == 'GEOMETRY_INCONSISTENT' and now - req.get('wall', 0) < 30:
        return _out('GEOMETRY_INCONSISTENT', [f"GEOMETRY_INCONSISTENT {req.get('detail')}"], m, hyps, rows, now)

    # ---------- INITIALIZING ----------
    if not (e3 or {}).get('session'):
        return _out('INITIALIZING', ['NO_ODOM_SESSION (FAST-LIO / feed not producing samples)'], m, hyps, rows, now)
    if not e2 or e2.get('status') == 'WAITING_DATA':
        note = (e2 or {}).get('wait_note')
        return _out('INITIALIZING', [f'E2_WAITING_FOR_WINDOW (store span {note} s < E2 window+10 s)' if note else 'E2_WAITING_FOR_WINDOW'],
                    m, hyps, rows, now)
    coarse_done = ((e2.get('coarse') or {}).get('floors_done')) or []
    m['coarse_floors_done'] = len(coarse_done)
    if not done and not coarse_done and (e2.get('pass_id') or 0) <= 1:
        return _out('INITIALIZING', ['E2_FIRST_FLOOR_RUNNING (no floor result yet)'], m, hyps, rows, now)
    if (e2.get('pass_id') or 0) > 1 and e2.get('status') == 'RUNNING':
        m['research_running'] = f"pass {e2.get('pass_id')} ({e2.get('pass_why')}): {len(done)}/{floors_total} floors"

    # ---------- SEARCH_INCOMPLETE ----------
    if not first_complete:
        # Ambiguity can be judged among hypotheses that already carry LiDAR evidence even before the exact 9-floor pass
        # completes (2026-10-20 field: the exact pass never completed in 4 runs, so AMBIGUOUS could never fire; in the
        # featureless-corridor run B the existing rule finds 7F/8F/4F within 0.008-0.05 m in 52/95 evaluations).
        # Only AMBIGUOUS is decided early; a leader is never called DOMINANT/CONFIDENT while the search is incomplete.
        part = [h for h in hyps if h.get('n_kf', 0) >= KF_MIN]
        if len(part) >= 2:
            pt = sorted(part, key=lambda h: h['score'])
            pc = next((h for h in pt[1:] if _sep(pt[0], h)[0]), None)
            pg = None if pc is None else pc['score'] - pt[0]['score']
            near = sorted({h['floor'] for h in pt if h['score'] - pt[0]['score'] < GAP_AMB})
            m.update(partial_best_floor=pt[0]['floor'], partial_competitor_floor=None if pc is None else pc['floor'],
                     partial_gap=None if pg is None else round(pg, 4), partial_floors_within_gap_amb=near)
            if (pg is not None and pg < GAP_AMB) or len(near) >= 2:
                return _out('AMBIGUOUS', [f'partial search ({len(done)}/{floors_total} exact): {len(near)} floor(s) {near} / competitor '
                                          f'{None if pc is None else pc["floor"]}F within {pg if pg is None else round(pg, 3)} < {GAP_AMB} m '
                                          f'of {pt[0]["floor"]}F among LiDAR-verified hypotheses (>= {KF_MIN} keyframes)',
                                          'SEARCH_INCOMPLETE: not every floor has an exact result'], m, pt, rows, now)
        pend = (e2 or {}).get('floors_pending') or []
        return _out('SEARCH_INCOMPLETE', [f'E2 pass {e2.get("pass_id")}: {len(done)}/{floors_total} floors searched, pending {pend}'
                                          + (f' (coarse pre-ranking {len(coarse_done)}/{floors_total}, approximate)' if coarse_done else '')
                                          + (f' (held: {e2.get("held")})' if e2.get('held') else '')], m, hyps, rows, now)

    # ---------- AMBIGUOUS / DOMINANT / CONFIDENT ----------
    top = sorted(hyps, key=lambda h: h['score'])
    b = top[0]
    comp, why_sep = None, None
    for h in top[1:]:
        sep, w = _sep(b, h)
        if sep:
            comp, why_sep = h, w
            break
    gap = None if comp is None else comp['score'] - b['score']
    near = [h for h in top if h['score'] - b['score'] < GAP_AMB]
    floors_near = sorted({h['floor'] for h in near})
    pth = path_since(rows, b['fit_t']) if rows else 0.0
    m.update(best_id=b['id'], best_floor=b['floor'], best_score=b['score'], best_recent=b['recent'], best_n_kf=b['n_kf'],
             competitor_id=None if comp is None else comp['id'], competitor_floor=None if comp is None else comp['floor'],
             gap=None if gap is None else round(gap, 4), separation=why_sep, floors_within_gap_amb=floors_near,
             floor_diversity=len({h['floor'] for h in top}), path_since_best_fit_m=round(pth, 2),
             best_violations=b['violations'], best_twoway_mean=b['twoway_mean'],
             leader_streak_kf=(e3 or {}).get('leader_streak_kf'))
    if (gap is not None and gap < GAP_AMB) or len(floors_near) >= 2:
        r = []
        if gap is not None and gap < GAP_AMB:
            r.append(f'competitor {comp["id"]} ({why_sep}) within {gap:.3f} < {GAP_AMB} m of the best')
        if len(floors_near) >= 2:
            r.append(f'{len(floors_near)} floors {floors_near} explain the scans within {GAP_AMB} m')
        return _out('AMBIGUOUS', r, m, top, rows, now)
    conf_fail = []
    if gap is not None and gap < GAP_CONF:
        conf_fail.append(f'gap {gap:.3f} < {GAP_CONF}')
    if b['n_kf'] < KF_MIN:
        conf_fail.append(f'keyframes {b["n_kf"]} < {KF_MIN}')
    if pth < PATH_MIN_M:
        conf_fail.append(f'path since E2 window {pth:.1f} < {PATH_MIN_M} m')
    if b['recent'] is None or b['recent'] > COST_OK:
        conf_fail.append(f'recent frame merit {b["recent"]} > {COST_OK}')
    streak = (e3 or {}).get('leader_streak_kf') or 0
    if streak < LEADER_KF_MIN:
        conf_fail.append(f'leader stable for {streak} < {LEADER_KF_MIN} keyframes')
    if b['violations']:
        conf_fail.append(f'{b["violations"]} hard violations in recent keyframes')
    if conf_fail:
        return _out('DOMINANT_UNVERIFIED', ['leader without enough independent evidence: ' + '; '.join(conf_fail)]
                    + ([] if comp else ['no distinct competitor alive']), m, top, rows, now)
    return _out('CONFIDENT_GEOMETRY', [f'gap {gap if gap is not None else "inf"} >= {GAP_CONF}, {b["n_kf"]} keyframes, {pth:.1f} m path',
                                       'GEOMETRY ONLY: E6/E7 not connected, not a verified final pose'], m, top, rows, now)


DISPLAY_KO = dict(AMBIGUOUS='위치 모호성 발생', GEOMETRY_INCONSISTENT='기하 불일치 (재탐색)', RECOVERY_REQUIRED='복구 필요', INITIALIZING='초기화 중',
                  SEARCH_INCOMPLETE='층 탐색 중 (전체 층 미완료)', DOMINANT_UNVERIFIED='유력 후보 있음 (미검증)',
                  CONFIDENT_GEOMETRY='기하 일치 (최종 위치 아님)')


def candidate_situation(state, m):
    """Separate 'there is nothing to choose from' from 'there are several equally good answers'."""
    n = m.get('n_active') or 0
    if n == 0:
        return 'NO_CANDIDATES'
    if state == 'GEOMETRY_INCONSISTENT':
        return 'GEOMETRY_INCONSISTENT'
    if state == 'AMBIGUOUS':
        return 'AMBIGUOUS_CANDIDATES'
    return 'SINGLE_CANDIDATE' if n == 1 else 'MULTIPLE_CANDIDATES'


def _out(state, reasons, m, top, rows, now):
    return dict(state=state, display_ko=DISPLAY_KO.get(state, state), candidates=candidate_situation(state, m),
                ambiguous=state == 'AMBIGUOUS', reasons=reasons, metrics=m, top=[h['id'] for h in (top or [])[:8]],
                thresholds=THRESHOLDS, wall=now)


# ------------------------------------------------------------------ E4 -> SigLIP activation (request only)
SIGLIP_HOLD_S = 15.0          # stay ACTIVE this long after the last reason to look, so a flickering state does not thrash


FIX_CONTRACT = dict(
    e4_role="LiDAR geometry state + semantic need only; E4 never declares a final pose",
    semantic_request="allowed in every state where semantic_need().needed (incl. SEARCH_INCOMPLETE = MEDIUM, AMBIGUOUS = HIGH); "
                     "an incomplete E2 exact pass does NOT block E5 search / STOP / E6 observation",
    final_fix="e6e7_link.global_decision == GLOBAL_CONFIRMED only: a DECISIVE semantic reading (evidence gate: CAPTURED, sharp, "
              "AUTO_E5/OPERATOR_ASSISTED) CONSISTENT with hypothesis H AND E7 independent geometry unique+separated with best == H",
    not_a_fix="E2/E3 Top-1, E4 CONFIDENT_GEOMETRY, semantic alone, forced/offline E7")
MILESTONES = ("first_hypothesis", "first_semantic_request", "first_stop_request", "first_e6_precision_result", "first_e7_evaluation",
              "first_global_confirmed")


def update_milestones(ms, now, e3, need, state_dir, link_last):
    """First wall time of each stage (2026-10-21: time to candidate / semantic request / E7 / FIX were not separable).
    Only LIVE events of this run are read; whether a FIX was CORRECT is decided after the run against GT, never here."""
    sd = Path(state_dir)
    seen = dict(first_hypothesis=bool((e3 or {}).get("n_alive")), first_semantic_request=bool((need or {}).get("needed")),
                first_stop_request=(sd / "e6_observations.jsonl").exists() and (sd / "e6_observations.jsonl").stat().st_size > 0,
                first_e6_precision_result=(sd / "e6_hypothesis_updates.jsonl").exists(),
                first_e7_evaluation=bool((link_last or {}).get("observation_id")),
                first_global_confirmed=(link_last or {}).get("global_decision") == "GLOBAL_CONFIRMED")
    for k in MILESTONES:
        if seen[k] and ms.get(k) is None:
            ms[k] = round(now, 3)
    return ms


def siglip_request(e4, need, prev, now):
    """What E4 asks of the semantic stage. E4 never starts/stops a process: it writes this request; the observe side decides
    (and records) whether it obeys. ACTIVE while semantic evidence can change the answer (need.priority HIGH / MEDIUM),
    IDLE_OK only after SIGLIP_HOLD_S without such a reason. 'event' is set on the tick where the mode or the ambiguity
    changes (the E4 -> SigLIP activation event)."""
    prev = prev or {}
    want = need.get('priority') in ('HIGH', 'MEDIUM')
    last_reason = now if want else prev.get('last_reason_wall')
    mode = 'ACTIVE' if want or (last_reason is not None and now - last_reason < SIGLIP_HOLD_S) else 'IDLE_OK'
    event = None
    if e4['ambiguous'] and not prev.get('ambiguous'):
        event = 'AMBIGUITY_RAISED'
    elif prev.get('ambiguous') and not e4['ambiguous']:
        event = 'AMBIGUITY_CLEARED'
    elif mode != prev.get('mode'):
        event = 'SIGLIP_' + mode
    seq = int(prev.get('seq') or 0) + (1 if event else 0)
    return dict(mode=mode, priority=need.get('priority'), why=need.get('why'), e4_state=e4['state'], ambiguous=e4['ambiguous'],
                candidates=e4['candidates'], event=event, last_event=event or prev.get('last_event'), seq=seq, stamp=now, last_reason_wall=last_reason,
                since=now if event else prev.get('since', now), hold_s=SIGLIP_HOLD_S, source='LIVE2_E4')


# ------------------------------------------------------------------ semantic need (E4 -> E5)
class SemanticMaps:
    """Read-only semantic map packages (kci_handoff/<F>F/<F>F_semantic_map.json). Pixel frame must equal the localization
    map (checked by image size); otherwise the floor is NO_SEMANTIC_MAP."""

    def __init__(self, root, building, loc_sizes=None):
        self.root, self.building = Path(root), building
        self.maps, self.status = {}, {}
        self.loc_sizes = loc_sizes or {}

    def floor_dir(self, f):
        return f'{f}F' if self.building == 'kci' else {'sw': f'S{f}F', 'gong': f'G{f}F'}.get(self.building, f'{f}F')

    def get(self, f):
        if f in self.maps or f in self.status:
            return self.maps.get(f)
        name = self.floor_dir(f)
        p = self.root / name / f'{name}_semantic_map.json'
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            self.status[f] = 'NO_SEMANTIC_MAP'
            return None
        sz = d['map_meta']['image_size']
        if f in self.loc_sizes and tuple(self.loc_sizes[f]) != (sz['width'], sz['height']):
            self.status[f] = f"PIXEL_FRAME_MISMATCH sem {sz} loc {self.loc_sizes[f]}"
            return None
        lm = []
        for s in d.get('spaces', []):
            poly = np.asarray(s.get('polygon_px') or [], float)
            if s.get('room_number') and len(poly):
                lm.append(dict(kind='room_number_plate', label=str(s['room_number']), name=s.get('name'), poly=poly,
                               center=np.asarray(s['center_px'], float)))
            elif s.get('type') in ('stair', 'elevator', 'restroom') and len(poly):
                lm.append(dict(kind=s['type'] + '_sign', label=s['id'], name=s.get('name'), poly=poly,
                               center=np.asarray(s['center_px'], float)))
        self.maps[f] = dict(landmarks=lm, file=str(p), resolution=d['map_meta'].get('resolution_m_per_px'))
        self.status[f] = f'OK {len(lm)} landmarks'
        return self.maps[f]


def _nearest_on_poly(poly, p):
    best, bd = None, 1e18
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
        q = a + t * ab
        d = float(np.hypot(*(q - p)))
        if d < bd:
            best, bd = q, d
    return best


def _los(wall, a, b, skip_px):
    n = int(max(2, math.hypot(b[0] - a[0], b[1] - a[1])))
    xs = np.rint(np.linspace(a[0], b[0], n)).astype(int); ys = np.rint(np.linspace(a[1], b[1], n)).astype(int)
    keep = np.hypot(xs - b[0], ys - b[1]) > skip_px                    # the plate is ON the wall: ignore its own wall
    H, W = wall.shape
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    return not bool(wall[ys[ok & keep], xs[ok & keep]].any())


def semantic_need(e4, hyps, sem, walls):
    """Why semantic observation is needed now, and which map landmarks to look for first (priority order)."""
    st = e4['state']
    pri = dict(AMBIGUOUS='HIGH', RECOVERY_REQUIRED='HIGH', GEOMETRY_INCONSISTENT='HIGH', SEARCH_INCOMPLETE='MEDIUM', DOMINANT_UNVERIFIED='MEDIUM',
               CONFIDENT_GEOMETRY='LOW', INITIALIZING='MEDIUM').get(st, 'NONE')
    why = dict(AMBIGUOUS='distinct hypotheses explain the LiDAR geometry equally well; a readable sign can separate them',
               RECOVERY_REQUIRED='inputs or hypotheses are invalid; a sign is an independent re-anchor',
               GEOMETRY_INCONSISTENT='no hypothesis explains the LiDAR geometry better than the rest; a sign is an independent re-anchor',
               SEARCH_INCOMPLETE='the floor search is still running; a floor/room number would narrow it',
               DOMINANT_UNVERIFIED='a geometric leader exists but has no independent confirmation (E6/E7 not connected)',
               CONFIDENT_GEOMETRY='geometry is self-consistent; a sign would still be the only independent check',
               INITIALIZING='no geometry hypothesis yet: a floor / room sign is the fastest independent cue').get(st)
    act = [h for h in hyps if h['status'] == 'ACTIVE' and h['pose_px'] is not None]
    act = sorted(act, key=lambda h: h['score'])[:6]
    floors = sorted({h['floor'] for h in act})
    targets = []
    for h in act:
        sm = sem.get(h['floor'])
        if sm is None:
            continue
        p = np.asarray(h['pose_px'], float)
        hd = math.radians(h['heading_deg'] or 0.0)
        for lm in sm['landmarks']:
            q = _nearest_on_poly(lm['poly'], p)
            d_m = float(np.hypot(*(q - p))) / h['ppm']
            if d_m > R_OBS_M:
                continue
            v = q - p
            bearing = math.degrees(math.atan2(-v[1], v[0]) - hd)          # image y down: map angle = atan2(-dy, dx)
            bearing = (bearing + 180) % 360 - 180
            wall = walls.get(h['floor'])
            los = None if wall is None else _los(wall, p, q, 0.4 * h['ppm'])
            if los and abs(bearing) <= FOV_HALF_DEG and d_m <= NEAR_OBS_M:
                obs = 'OBSERVABLE_NOW'
            elif los:
                obs = 'NEEDS_OPERATOR_TURN'
            else:
                obs = 'NO_LINE_OF_SIGHT' if los is not None else 'LOS_UNKNOWN'
            discriminates = len(floors) >= 2 and lm['kind'] == 'room_number_plate'
            score = (2 if discriminates else 0) + dict(OBSERVABLE_NOW=2, NEEDS_OPERATOR_TURN=1).get(obs, 0) + (1 - d_m / R_OBS_M)
            targets.append(dict(hyp=h['id'], floor=h['floor'], kind=lm['kind'], label=lm['label'], name=lm['name'],
                                dist_m=round(d_m, 2), bearing_deg=round(bearing, 1), observability=obs,
                                discriminates_floors=discriminates, priority_score=round(score, 3),
                                target_px=[round(float(q[0]), 1), round(float(q[1]), 1)],
                                note='plate position = nearest room boundary to the hypothesis (door position unknown); '
                                     'bearing uses the hypothesis heading (frame contract unverified)'))
    targets.sort(key=lambda t: -t['priority_score'])
    return dict(needed=pri in ('HIGH', 'MEDIUM'), priority=pri, why=why, floors_in_question=floors,
                targets=targets[:8], wanted_labels=sorted({t['label'] for t in targets[:8] if t['kind'] == 'room_number_plate'}),
                semantic_maps={f: sem.status.get(f) for f in floors}, pantilt='NOT_COMMANDED (forbidden in LIVE2)')
