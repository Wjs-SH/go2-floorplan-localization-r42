"""e6_semantic.py — LIVE2 E6 semantic evidence: OCR / VLM reading -> label alternatives -> floors -> per-E3-hypothesis verdict.

Reuses the student code unchanged (imported, never copied):
  src/sci_text_reading/semantic.py        classify_text (room-number / poster / directory typing), fuse (OCR+VLM, OCR alone
                                          only >= 0.85 confidence, conflicts keep both alternatives)
  src/sci_semantic_localization/e5_e7/e6_region.py  norm_label, FloorMap.spaces_with_label (semantic maps, read-only)
Floor likelihood = student rule: sum_alt p_alt * (1 if the label exists on that floor else 0.02), normalised.
KCI: every room label starts with its floor digit and no label repeats across 1F-9F (semantic maps), so a reliable reading
is floor evidence; the PLACE is checked against each hypothesis' projection of the observed target (E6 target).
Verdicts per hypothesis (no probability is invented beyond the student rule):
  SEMANTIC_CONSISTENT  label on the hypothesis floor AND its projected target within PLACE_TOL_M of a labelled room
  PLACE_MISMATCH       label on that floor, projected target far from every room with that label
                       (2026-11-19, E8 pair path only: or farther across the wall than half the corridor — see HALF_CORRIDOR)
  FLOOR_MISMATCH       label not on that floor
  NO_LABEL             nothing usable was read (poster / directory / unreadable) -> no evidence
GT is never an input.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

CAND = Path(__file__).resolve().parents[1]
for _p in (CAND / "src" / "sci_text_reading", CAND / "src" / "sci_semantic_localization" / "e5_e7"):
    sys.path.insert(0, str(_p))

PLACE_TOL_M = 3.0
# 2026-11-19 half-corridor place rule (user decision; replaces the room-size question of 11-18 B5 with a value read from the map).
# A room's plate hangs on the room's corridor-facing wall, so under a right transform the projected plate lies on that wall line give or take
# the transform error.  ACROSS the wall the corridor itself bounds it: farther than half the local corridor width and the plate is nearer the
# rooms on the opposite side (or out of the corridor), whatever the room size.  ALONG the wall the corridor says nothing, so the distance to
# the wall-line segment keeps PLACE_TOL_M (3.0 m = largest pre-E7 plate error of the best candidate in 41 recorded pair checks).
#   wall line   = semantic_anchor.plate_wall_points (the room edge nearest a corridor polygon, moved halfway to it) — the same line the
#                 plate-anchored candidates are built on.
#   half width  = area / perimeter of the corridor polygon nearest the projected plate (= half of 2A/P, the mean width of a long corridor).
#   No corridor polygon, no wall line, or a facility category (cat:...) -> the room-polygon rule above, unchanged.
# Measured offline (41 recorded pair checks, 111 candidates, graded by a whole-run/window LiDAR reference, not the clicked start): removes
# 34/80 wrong candidates, 0/31 lost among candidates E7 actually brings to the reference; after E7 every right plate (43/43) is within it.
# Used only where a caller asks for it (e8_pair, the second-plate check of the path-anchored job); E6 single-plate verdicts are unchanged.
# Off: LIVE2_PLATE_HALF_CORRIDOR=0 -> the room-polygon rule exactly.
HALF_CORRIDOR = __import__("os").environ.get("LIVE2_PLATE_HALF_CORRIDOR", "1") == "1"
ABSENT_LL = 0.02                       # student e6_region label_likelihood for a label absent from a floor
import pwd as _pwd
SEM_ROOT = Path(_pwd.getpwuid(__import__("os").getuid()).pw_dir) / "sci_ws" / "maps" / "kci_handoff"   # real home (not $HOME)


CATEGORY_PREFIX = "cat:"          # R34: a facility category (e.g. restroom), not a room label: never decides a floor by itself
CATEGORY_MIN_CONF = 0.85
_CATEGORY_WORDS = {"restroom": ("TOILET", "TOILETS", "RESTROOM", "RESTROOMS", "WC", "화장실", "남자화장실", "여자화장실")}


def category_of(ocr_lines):
    """Facility word in the OCR text -> dict(category, label, original_text, confidence, source), or None. 11-01: 'TOILETS' 0.935 was thrown away because
    the student classifier only knows room numbers. Exact word match after removing spaces (no fuzzy guess)."""
    best = None
    for text, conf in ocr_lines or []:
        t = "".join(str(text).split()).upper()
        for cat, words in _CATEGORY_WORDS.items():
            if t in words and float(conf) >= CATEGORY_MIN_CONF and (best is None or float(conf) > best["confidence"]):
                best = dict(category=cat, label=CATEGORY_PREFIX + cat, original_text=str(text), confidence=float(conf), source="OCR")
    return best


def label_alternatives(ocr_lines, vlm=None, with_category=False):
    """ocr_lines = [(text, conf)], vlm = dict(sign_type, room_number) or None -> (alternatives [(label, p)], fusion dict).
    with_category (E8 pair path only): no room label but a facility word -> one 'cat:<category>' alternative, marked in info['category']."""
    import semantic as S
    oc = S.classify_text(ocr_lines) if ocr_lines else None
    fu = S.fuse(oc, vlm)
    alts = [(a[0] if isinstance(a, (list, tuple)) else a.get("label"), float(a[1] if isinstance(a, (list, tuple)) else a.get("p", 0)))
            for a in (fu.get("label_alternatives") or [])]
    alts = [(lab, p) for lab, p in alts if lab]
    info = dict(ocr=oc, fusion=fu)
    if with_category and not alts:
        c = category_of(ocr_lines)
        if c:
            alts, info["category"] = [(c["label"], c["confidence"])], c
    return alts, info


class SemanticFloors:
    def __init__(self, building="kci", floors=range(1, 10), root=SEM_ROOT):
        self.maps = {}
        for f in floors:
            p = root / f"{f}F" / f"{f}F_semantic_map.json"
            try:
                self.maps[f] = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass

    def spaces(self, floor, label):
        if str(label).startswith(CATEGORY_PREFIX):       # every location of that category on the floor
            return [s for s in (self.maps.get(floor) or {}).get("spaces", []) if s.get("type") == label[len(CATEGORY_PREFIX):] and s.get("polygon_px")]
        from e6_region import norm_label
        return [s for s in (self.maps.get(floor) or {}).get("spaces", [])
                if norm_label(str(s.get("room_number") or "")) == label and s.get("polygon_px")]

    def floor_likelihood(self, alts):
        ll = {f: sum(p * (1.0 if self.spaces(f, lab) else ABSENT_LL) for lab, p in alts) for f in self.maps}
        z = sum(ll.values())
        return {f: (v / z if z > 0 else 0.0) for f, v in ll.items()}


def _poly_dist_px(poly, q):
    """0 inside the room polygon (the plate is on its wall; projection error can land the target either side),
    otherwise the distance to its boundary."""
    poly = np.asarray(poly, float)
    x, y = float(q[0]), float(q[1])
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, 0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    if inside:
        return 0.0
    best = 1e18
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        ab = b - a
        t = float(np.clip(np.dot(q - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
        best = min(best, float(np.hypot(*(a + t * ab - q))))
    return best


def corridor_half_width_px(corridors, q):
    """Half the mean width (area / perimeter) of the corridor polygon nearest q (0 inside). -> px, or None without corridors."""
    best = None
    for c in corridors or []:
        c = np.asarray(c, float)
        if len(c) < 3:
            continue
        x, y = c[:, 0], c[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        perim = float(np.sum(np.hypot(*(np.roll(c, -1, 0) - c).T)))
        d = _poly_dist_px(c, q)
        if perim > 0 and (best is None or d < best[0]):
            best = (d, area / perim)
    return None if best is None else best[1]


def wall_line_px(room_poly, corridors):
    """The room's corridor-facing wall line (semantic_anchor.plate_wall_points end points) -> (a, b) px, or None."""
    if not corridors:
        return None
    import semantic_anchor as SA
    pts = SA.plate_wall_points(room_poly, corridors, 1.0, step_m=1e9)       # one step per edge -> exactly the two end points
    if len(pts) != 2:
        return None
    a, b = np.asarray(pts[0], float), np.asarray(pts[1], float)
    return (a, b) if float(np.hypot(*(b - a))) > 1.0 else None


def place_check(spaces, corridors, q, ppm, half_corridor=True):
    """Projected plate q (px) against the rooms carrying its label. -> (ok, info) for the best room (passing first, then nearest).
    info: dist_m (room polygon, as before), map_room_id/name, and with the half-corridor rule wall_seg_m, wall_perp_m, wall_along_m, half_corridor_m."""
    q = np.asarray(q, float); ppm = float(ppm); best = None
    for s_ in spaces:
        d_poly = _poly_dist_px(s_["polygon_px"], q) / ppm
        info = dict(dist_m=round(d_poly, 2), map_room_id=s_.get("id"), map_room_name=s_.get("name"), place_rule="ROOM_POLYGON")
        wl = wall_line_px(s_["polygon_px"], corridors) if half_corridor else None
        hw = corridor_half_width_px(corridors, q) if wl is not None else None
        if wl is None or hw is None:
            ok, key = d_poly <= PLACE_TOL_M, d_poly
        else:
            a, b = wl; L = float(np.hypot(*(b - a))); u = (b - a) / L; n = np.array([-u[1], u[0]]); t = float(np.dot(q - a, u))
            perp = abs(float(np.dot(q - a, n))) / ppm; along = max(0.0, -t, t - L) / ppm; seg = math.hypot(perp, along); half = hw / ppm
            ok, key = (seg <= PLACE_TOL_M and perp <= half), seg
            info.update(place_rule="HALF_CORRIDOR", wall_seg_m=round(seg, 2), wall_perp_m=round(perp, 2), wall_along_m=round(along, 2), half_corridor_m=round(half, 2))
        if best is None or (not ok, key) < (not best[0], best[1]):
            best = (ok, key, info)
    return (False, None) if best is None else (best[0], best[2])


def evaluate(alts, hyps, sf, target_odom_link_xy, half_corridor=False):
    """hyps = [{'id','floor','rec'}] (E3 alive), target_odom_link_xy = E6 target in the odom_link frame.
    -> dict(floor_likelihood, per_hypothesis {id: verdict dict}).
    half_corridor (E8 pair path, 2026-11-19): room-number labels use place_check's half-corridor rule; facility categories keep the room polygon."""
    import e6_core as G
    out = dict(alternatives=alts, floor_likelihood=sf.floor_likelihood(alts) if alts else None, per_hypothesis={})
    if not alts:
        for h in hyps:
            out["per_hypothesis"][h["id"]] = dict(verdict="NO_LABEL")
        return out
    for h in hyps:
        best = None
        if half_corridor:
            cor = [c["polygon_px"] for c in (sf.maps.get(h["floor"]) or {}).get("corridors") or [] if c.get("polygon_px")]
            ok_best = None
            for lab, p in alts:
                sp = sf.spaces(h["floor"], lab)
                if not sp:
                    continue
                px, py = G.map_projection(h["rec"], *target_odom_link_xy)
                ok, info = place_check(sp, cor, (px, py), h["rec"]["ppm"], half_corridor=not str(lab).startswith(CATEGORY_PREFIX))
                key = (not ok, info.get("wall_seg_m", info["dist_m"]))
                if ok_best is None or key < ok_best[0]:
                    ok_best = (key, ok, dict(info, label=lab, p=p, target_px=[round(px, 1), round(py, 1)], observed_room_id=lab))
            if ok_best is None:
                out["per_hypothesis"][h["id"]] = dict(verdict="FLOOR_MISMATCH", floor=h["floor"], labels=[a[0] for a in alts])
            else:
                _, ok, b = ok_best
                out["per_hypothesis"][h["id"]] = dict(b, floor=h["floor"], semantic_score=b["p"] if ok else 0.0,
                                                      verdict="SEMANTIC_CONSISTENT" if ok else "PLACE_MISMATCH")
            continue
        for lab, p in alts:
            sp = sf.spaces(h["floor"], lab)
            if not sp:
                continue
            px, py = G.map_projection(h["rec"], *target_odom_link_xy)
            ds = [(_poly_dist_px(s_["polygon_px"], np.array([px, py])) / float(h["rec"]["ppm"]), s_) for s_ in sp]
            d, sp_best = min(ds, key=lambda x: x[0])
            if best is None or d < best["dist_m"]:
                best = dict(label=lab, p=p, dist_m=round(d, 2), target_px=[round(px, 1), round(py, 1)],
                            observed_room_id=lab, map_room_id=sp_best.get("id"), map_room_name=sp_best.get("name"))
        if best is None:
            out["per_hypothesis"][h["id"]] = dict(verdict="FLOOR_MISMATCH", floor=h["floor"], labels=[a[0] for a in alts])
        else:
            out["per_hypothesis"][h["id"]] = dict(best, floor=h["floor"], semantic_score=best["p"] if best["dist_m"] <= PLACE_TOL_M else 0.0,
                                                  verdict="SEMANTIC_CONSISTENT" if best["dist_m"] <= PLACE_TOL_M else "PLACE_MISMATCH")
    return out
