"""e8_map_data.py — what the semantic maps can and cannot say about door plates (E8 pair consistency).

KCI semantic_prior_package_v1 (kci_handoff/<F>F/<F>F_semantic_map.json) has room polygons, corridor polygons and the floor-plan
OCR anchor (text printed on the drawing) — NO physical plate position and NO plate facing. Nothing here invents them.

  PLATE_SCHEMA / validate_plates(map)  optional future field 'plate_landmarks' (surveyed plates). Today every floor ->
                                        BLOCKED_BY_MAP_DATA: plate-to-plate distance and plate yaw cannot be checked directly.
  corridor_wall(map, space)             DERIVED observable zone: the room-polygon edges facing a corridor polygon (a door plate is
                                        mounted on that wall). provenance DERIVED_ROOM_CORRIDOR_ADJACENCY, uncertainty = ADJ_M
                                        (+ the map resolution); UNVERIFIED — a diagnostic interval, never a fix by itself.
  wall_interval(map, a, b)              [min, max] distance between the derived corridor walls of two rooms (m).
"""
import numpy as np

import e8_pair as EP

PLATE_SCHEMA = dict(
    required=dict(id=str, floor=int, room_number=str, position_map_m=list, facing_rad=float, uncertainty_m=float,
                  provenance=str, verified=bool),
    provenance_allowed=("SURVEYED_ON_SITE", "PHOTO_REGISTERED", "CAD_DOOR_SIGN_LAYER"),
    note="a plate is usable for E8 only with verified=true and a stated uncertainty; derived anchors are never plates")


def validate_plates(floor_map):
    """-> dict(status OK | BLOCKED_BY_MAP_DATA | INVALID, n, problems). Reads only 'plate_landmarks' (absent today)."""
    plates = floor_map.get("plate_landmarks")
    if not plates:
        return dict(status="BLOCKED_BY_MAP_DATA", n=0,
                    problems=["no 'plate_landmarks' in the semantic map: physical plate position / facing unknown"])
    problems = []
    for i, p in enumerate(plates):
        for k, t in PLATE_SCHEMA["required"].items():
            if k not in p:
                problems.append(f"#{i} missing {k}")
            elif t is float and not isinstance(p[k], (int, float)):
                problems.append(f"#{i} {k} not a number")
            elif t is not float and not isinstance(p[k], t):
                problems.append(f"#{i} {k} not {t.__name__}")
        if p.get("provenance") not in PLATE_SCHEMA["provenance_allowed"]:
            problems.append(f"#{i} provenance {p.get('provenance')!r} not allowed")
        if p.get("verified") is not True:
            problems.append(f"#{i} not verified")
    ok = [p for i, p in enumerate(plates) if not any(x.startswith(f"#{i} ") for x in problems)]
    return dict(status="OK" if ok and not problems else ("INVALID" if not ok else "PARTIAL"), n=len(ok), problems=problems[:20])


def _res(floor_map):
    return float((floor_map.get("map_meta") or {}).get("resolution_m_per_px") or 0.0) or None


def corridor_wall(floor_map, space):
    """Room edges (m) whose midpoint lies within ADJ_M of a corridor polygon -> dict(segments, length_m, provenance, unc)."""
    res = _res(floor_map)
    cors = floor_map.get("corridors") or []
    if res is None or not cors:
        return dict(segments=[], length_m=0.0, provenance="NO_CORRIDOR_DATA", uncertainty_m=None)
    polys, _ = EP.corridor_components(cors, res)
    P = EP._poly_m(space, res)
    segs = []
    for a, b in zip(P, np.roll(P, -1, 0)):
        mid = (a + b) / 2
        d = min(min(EP._seg_dist(mid, c0, c1) for c0, c1 in zip(c, np.roll(c, -1, 0))) for c in polys)
        if d <= EP.ADJ_M:
            segs.append((a, b))
    return dict(segments=segs, length_m=round(float(sum(np.hypot(*(b - a)) for a, b in segs)), 2),
                provenance="DERIVED_ROOM_CORRIDOR_ADJACENCY" if segs else "NOT_ADJACENT_TO_CORRIDOR",
                uncertainty_m=round(EP.ADJ_M + res, 2) if segs else None)


def _sample(segs, step=0.5):
    pts = []
    for a, b in segs:
        n = max(2, int(np.hypot(*(b - a)) / step) + 1)
        pts += [a + (b - a) * t for t in np.linspace(0, 1, n)]
    return np.asarray(pts)


def wall_interval(floor_map, a, b):
    """[min, max] distance (m) between the derived corridor walls of rooms a and b, or None when either has none."""
    wa, wb = corridor_wall(floor_map, a), corridor_wall(floor_map, b)
    if not wa["segments"] or not wb["segments"]:
        return None
    A, B = _sample(wa["segments"]), _sample(wb["segments"])
    d = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1])
    return dict(interval_m=[round(float(d.min()), 2), round(float(d.max()), 2)], provenance="DERIVED_ROOM_CORRIDOR_ADJACENCY",
                uncertainty_m=round(max(wa["uncertainty_m"], wb["uncertainty_m"]), 2), verified=False)


def map_data_status(sf):
    """Per floor: plate landmarks (validator) + derived wall coverage, for the evaluation record and the tablet."""
    out = {}
    for f, m in (getattr(sf, "maps", None) or {}).items():
        rooms = [s for s in m.get("spaces", []) if s.get("room_number")]
        walls = sum(1 for s in rooms if corridor_wall(m, s)["segments"])
        out[f] = dict(plates=validate_plates(m)["status"], rooms=len(rooms), rooms_with_derived_wall=walls,
                      yaw_check="BLOCKED_BY_MAP_DATA" if validate_plates(m)["status"] != "OK" else "AVAILABLE")
    return out
