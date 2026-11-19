"""Estimate the planar scan-to-odometry rotation from overlapping observations.

No map, floor identifier, trajectory PCA, or ground truth is used. All frames
are supplied by the caller, which controls the causal time boundary.
"""
import numpy as np
from scipy.spatial import cKDTree


class CalibrationUnavailable(ValueError):
    """R1 (2026-10-16): β(센서 스캔-오돔 각도 정렬)를 이 관측으로 정할 수 없다. ValueError 를 이어받아 기존 호출부 호환.
    .result 에 typed 결과(status=CALIBRATION_UNAVAILABLE, reason, provenance)가 있다. 수치(228° 등)를 채워 넣지 않는다."""
    reason_code = 'CALIBRATION_UNAVAILABLE'

    def __init__(self, reason, provenance=None):
        super().__init__(reason)
        self.result = dict(status='CALIBRATION_UNAVAILABLE', beta=None, beta_null_reason=reason, quality=None,
                           provenance=dict(provenance or {}, method='estimate_beta(nn-registration)'))


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def estimate_beta(frames, pair_s=3.0, step_deg=2.0, min_disp=.3,
                  max_pair_error=.2, cap=500):
    """Robust nearest-neighbour registration, including odometry translation.

    A pair whose best possible alignment still exceeds max_pair_error metres
    cannot calibrate the frame transform. This rejects distorted startup scans
    rather than interpreting their unrelated points as a sensor rotation.
    """
    times = np.array([f['t'] for f in frames])
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise CalibrationUnavailable('Calibration needs at least two time-ordered scans',
                                     dict(n_frames=len(times), min_disp_m=min_disp, pair_s=pair_s))
    angles = np.deg2rad(np.arange(0., 360., step_deg))
    curves = {1: [], -1: []}
    intervals, turns = [], []
    for i, first in enumerate(frames):
        j = int(np.argmin(abs(times - (times[i] + pair_s))))
        if j <= i or abs(times[j] - times[i] - pair_s) > pair_s / 4:
            continue
        second = frames[j]
        delta = np.asarray(second['xy']) - first['xy']
        if np.linalg.norm(delta) < min_disp:
            continue
        p = np.asarray(first.get('near_pts', first['pts']), float)
        q = np.asarray(second.get('near_pts', second['pts']), float)
        p, q = p[np.isfinite(p).all(1)], q[np.isfinite(q).all(1)]
        if min(len(p), len(q)) < 50:
            continue
        if len(p) > cap: p = p[np.linspace(0, len(p)-1, cap).astype(int)]
        if len(q) > cap: q = q[np.linspace(0, len(q)-1, cap).astype(int)]
        tree = cKDTree(p)
        dyaw = second['yaw'] - first['yaw']
        for mirror in (1, -1):
            # In the original scan frame: M R(dyaw) M = R(mirror*dyaw).
            rotated = q @ rotation(mirror * dyaw).T
            costs = []
            for beta in angles:
                translation = (delta @ rotation(-first['yaw'] - beta).T) * [1, mirror]
                costs.append(np.median(tree.query(rotated + translation)[0]))
            curves[mirror].append(costs)
        intervals.append((float(times[i]), float(times[j])))
        turns.append(abs(np.arctan2(np.sin(dyaw), np.cos(dyaw))) >= np.deg2rad(10))
    if not intervals:
        raise CalibrationUnavailable('Not enough overlapping scans with translation to calibrate',
                                     dict(n_frames=len(times), n_moving_pairs=0, min_disp_m=min_disp, pair_s=pair_s,
                                          interval=[float(times[0]), float(times[-1])]))
    hypotheses = []
    for mirror, rows in curves.items():
        costs = np.asarray(rows)
        good = costs.min(axis=1) <= max_pair_error
        if not good.any(): continue
        objective = np.median(costs[good], axis=0)
        best = int(np.argmin(objective))
        # Straight travel cannot identify handedness. Evaluate turn pairs even
        # when a wrong handedness would reject them as bad observations.
        selection = np.asarray(turns) if sum(turns) >= 2 else np.ones(len(rows), bool)
        support = float(np.median(costs[selection, best]))
        hypotheses.append((support, mirror, best, objective, good))
    if not hypotheses:
        raise CalibrationUnavailable('No geometrically consistent scan pairs; collect more observations',
                                     dict(n_frames=len(times), n_moving_pairs=len(intervals), max_pair_error_m=max_pair_error,
                                          interval=[float(times[0]), float(times[-1])]))
    support, mirror, best, objective, good = min(hypotheses, key=lambda h: h[0])
    return float(angles[best]), dict(
        mirror=mirror, handedness_error_m=support, turn_pairs=int(sum(turns)),
        error_m=float(objective[best]), pairs=int(good.sum()),
        rejected_pairs=int((~good).sum()),
        interval=[float(times[0]), float(times[-1])],
        pair_intervals=[intervals[i] for i in np.flatnonzero(good)],
        step_deg=float(step_deg))


def estimate_beta_typed(frames, **kw):
    """R1: 예외 대신 typed 결과. status OK → beta(rad)·quality(pairs·error_m·handedness·turn_pairs)·provenance,
    CALIBRATION_UNAVAILABLE → beta=None + 이유. verified=False(이 창의 런타임 추정은 검증된 보정이 아니다)."""
    try:
        beta, q = estimate_beta(frames, **kw)
    except CalibrationUnavailable as e:
        return dict(e.result, verified=False)
    return dict(status='OK', beta=float(beta), quality=q, verified=False,
                provenance=dict(method='estimate_beta(nn-registration)', n_frames=len(frames), pairs=q.get('pairs'),
                                residual_m=q.get('error_m'), handedness=q.get('mirror'), turn_pairs=q.get('turn_pairs'),
                                interval=q.get('interval')))


class BetaProvider:
    """R1: β 재사용. **검증된** 보정(파일, verified=True)만 같은 계약(sensor·calibration_id·odom_session_id·설정)에서 창 사이에 다시 쓴다.
    계약이 다르거나(세션 바뀜·설정 바뀜) max_age_s 를 넘으면 무효로 하고 다시 쓰지 않는다. 런타임 추정은 그 창에만 쓴다(재사용 안 함).
    검증된 보정이 없고 이 창에서 추정도 안 되면 CALIBRATION_UNAVAILABLE(값을 지어내지 않는다).
    주의: 정지 상태의 초기 위치추정은 이것만으로 해결되지 않는다(사전 검증 보정 또는 별도 초기화 필요)."""
    CONTRACT_KEYS = ('sensor', 'calibration_id', 'odom_session_id', 'building', 'window_s')

    def __init__(self, verified=None, estimator=None, max_age_s=None, clock=None):
        import time as _t
        self.verified = verified            # dict(beta, mirror, contract{...}, provenance, verified_at)
        self.estimator = estimator          # (frames_or_ctx) -> typed result; None → caller-provided per call
        self.max_age_s = max_age_s
        self.clock = clock or _t.monotonic
        self.history = []
        self.calibration_id = (verified or {}).get('contract', {}).get('calibration_id') if verified else None

    @classmethod
    def from_file(cls, path):
        if not path:
            return cls()
        import json
        d = json.load(open(path))
        if not d.get('verified'):
            raise ValueError(f'{path}: verified=true 인 보정만 받는다(검증되지 않은 값을 보정으로 쓰지 않음)')
        return cls(verified=dict(d, loaded_mono=None))

    def _matches(self, contract):
        v = self.verified
        if not v:
            return False, 'NO_VERIFIED_CALIBRATION'
        vc = v.get('contract') or {}
        for k in self.CONTRACT_KEYS:
            if k in vc and vc[k] is not None and vc.get(k) != (contract or {}).get(k):
                return False, f'CONTRACT_MISMATCH:{k}'
        if self.max_age_s is not None and v.get('t_mono') is not None and self.clock() - v['t_mono'] > self.max_age_s:
            return False, 'STALE'
        return True, 'OK'

    def invalidate(self, why):
        if self.verified:
            self.history.append(dict(event='INVALIDATED', why=why, beta=self.verified.get('beta')))
        self.verified = None

    def get(self, contract, estimate_fn=None):
        """돌려줌: typed 결과 dict(status, beta, mirror, source=VERIFIED_CACHE|RUNTIME_ESTIMATE, provenance)."""
        ok, why = self._matches(contract)
        if ok:
            v = self.verified
            return dict(status='OK', beta=float(v['beta']), mirror=int(v.get('mirror', 1)), source='VERIFIED_CACHE', verified=True,
                        quality=v.get('quality'), provenance=dict(v.get('provenance') or {}, contract=v.get('contract')))
        if self.verified and why.startswith(('CONTRACT_MISMATCH', 'STALE')):
            self.invalidate(why)
        fn = estimate_fn or self.estimator
        if fn is None:
            return dict(status='CALIBRATION_UNAVAILABLE', beta=None, beta_null_reason=why, source=None, verified=False)
        r = dict(fn())
        if r.get('status') == 'OK':
            r.update(source='RUNTIME_ESTIMATE', mirror=int((r.get('quality') or {}).get('mirror', r.get('mirror', 1))))
            r.setdefault('verified', False)
        else:
            r.setdefault('beta', None)
            r['cache_status'] = why
        self.history.append(dict(event='GET', status=r.get('status'), source=r.get('source'), why=why))
        return r
