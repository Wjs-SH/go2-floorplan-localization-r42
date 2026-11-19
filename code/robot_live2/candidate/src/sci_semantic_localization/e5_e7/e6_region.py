#!/usr/bin/env python3
"""e6_region.py — E6: semantic label + observation bearing/range + visibility -> floor hypotheses + 2-D regions.

NEW module (E6 did not exist; thesis deck status "미수행 (설계)"). Isolated: reads the canonical semantic maps
read-only, writes only to --out.

Input  : E5_OBSERVATION_PACKET (json, schema e5_observation_packet/v1, see e5_packet.py)
Output : E6 result json + one region mask PNG per surviving floor hypothesis (+ overlay PNG)

Model (all parameters fixed in DEFAULTS before any trial; GT is never read):
  1. Label -> floors: a floor is a hypothesis iff its semantic map contains a space whose normalised
     room_number equals one of the packet's label alternatives (probability from the packet).
  2. Sign support S (where a room plate can physically be): wall pixels (base_map < 50) within `support_band_px`
     of the room polygon AND within `support_band_px` of free pixels (> 200) outside the room polygon
     (= the wall between the room and the space it is seen from). The semantic map has no sign positions or
     wall normals (READ: maps/4F/4F_semantic_map.json), so this is an explicit model assumption.
  3. Robot positions p: free pixels with some s in S such that
       |p - s| / ppm  in  [r - k*sr, r + k*sr]   for some ppm in [ppm_lo, ppm_hi]
     and a clear line of sight from p to s (no wall pixel on the segment except the last `los_skip_px`).
  4. Heading for each p: psi = heading(p->s) - beta_body (beta from the RealSense pixel, geometry.py).
     The pan angle is NOT used as a map bearing.
  5. Scale: kci has no metric base_map.yaml; [ppm_lo, ppm_hi] is an explicit interval built from the documented,
     conflicting sources (semantic map handshake 1/0.057894737 = 17.27 px/m; floor_id self-test 24.144 px/m;
     floor_id SCALES 15..43). Region areas are reported as an interval in m^2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from . import geometry as G
except ImportError:  # run as a script
    import geometry as G  # type: ignore

# SCI: read the sci_ws copies (originals untouched); override with env SCI_EVENT_MAPS / SCI_FLOOR_ID
import os as _os
EVENT_MAPS = Path(_os.environ.get("SCI_EVENT_MAPS", str(Path.home() / "sci_ws" / "maps" / "kci_handoff")))
_FID = Path(_os.environ.get("SCI_FLOOR_ID", str(Path.home() / "sci_ws" / "external" / "floor_id"))) / "maps"
FLOORID_MAPS = {"kci": _FID / "kci_maps_perfloor", "sw": _FID / "sw_maps_perfloor", "gong": _FID / "gong_maps_perfloor"}

DEFAULTS = dict(
    ppm_lo=17.27 * 0.9,          # semantic-map handshake minus 10 %
    ppm_hi=24.144 * 1.1,         # floor_id kci 4F self-test plus 10 %
    k_sigma=2.0,                 # range/bearing tolerance in sigmas
    support_band_px=6,           # wall-to-room / wall-to-outside-free adjacency
    los_skip_px=4,               # ignore the last pixels of the ray (the sign's own wall)
    los_steps=32,
    max_support_pts=200,
    min_range_m=0.3,
    label_miss_eps=0.02,         # P(label observed | label not on this floor) — misread/map omission
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm_label(s: str) -> str:
    """'401호', ' 401 ', 'Room 401' -> '401'; keeps letter prefixes like 'B101'. Empty if no digits."""
    s = (s or "").upper().replace("호", " ")
    m = re.findall(r"[A-Z]?\d{2,4}(?:-\d+)?", s)
    return m[0] if m else ""


class FloorMap:
    def __init__(self, floor: str, root: Path = EVENT_MAPS):
        self.floor = floor
        self.dir = root / floor
        self.sem_path = self.dir / f"{floor}_semantic_map.json"
        self.img_path = self.dir / "base_map.png"
        self.sem = json.loads(self.sem_path.read_text(encoding="utf-8"))
        img = cv2.imread(str(self.img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.img_path)
        self.img = img
        self.wall = img < 50
        self.free = img > 200
        meta = self.sem.get("map_meta", {})
        sz = meta.get("image_size", {})
        self.meta = dict(frame_id=meta.get("frame_id"), resolution_m_per_px=meta.get("resolution_m_per_px"),
                         image_size=[sz.get("width"), sz.get("height")], schema=self.sem.get("schema_version"))
        if sz and (sz.get("width"), sz.get("height")) != (img.shape[1], img.shape[0]):
            raise ValueError(f"{floor}: semantic image_size {sz} != base_map {img.shape[::-1]}")

    def spaces_with_label(self, label: str):
        out = []
        for sp in self.sem.get("spaces", []):
            if norm_label(str(sp.get("room_number") or "")) == label and sp.get("polygon_px"):
                out.append(sp)
        return out

    def provenance(self) -> dict:
        p = dict(floor=self.floor, semantic_map=str(self.sem_path), semantic_map_sha256=sha256(self.sem_path),
                 base_map=str(self.img_path), base_map_sha256=sha256(self.img_path), **self.meta)
        return p


def frame_equivalence(building: str, floor: str) -> dict:
    """Prove (or refute) that the event semantic map pixels == floor_id localisation map pixels for this floor."""
    a = EVENT_MAPS / floor / "base_map.png"
    root = FLOORID_MAPS.get(building)
    num = re.sub(r"\D", "", floor)
    b = (root / f"{num}F" / "maps" / "base_map.png") if root else None
    res = dict(event_base_map=str(a), floor_id_base_map=str(b) if b else None)
    if not (a.exists() and b and b.exists()):
        res.update(status="UNVERIFIED_MISSING_FILE")
        return res
    ha, hb = sha256(a), sha256(b)
    res.update(event_sha256=ha, floor_id_sha256=hb)
    if ha == hb:
        res["status"] = "IDENTICAL_FILE"
        return res
    ia, ib = cv2.imread(str(a), 0), cv2.imread(str(b), 0)
    if ia.shape != ib.shape:
        res.update(status="DIFFERENT_SIZE", shapes=[ia.shape, ib.shape])
        return res
    wa, wb = ia < 50, ib < 50
    inter, union = int((wa & wb).sum()), int((wa | wb).sum())
    res.update(status="SAME_SIZE_DIFFERENT_CONTENT", wall_iou=inter / max(union, 1))
    return res


def sign_support(fm: FloorMap, space: dict, band: int) -> np.ndarray:
    room = np.zeros(fm.img.shape, np.uint8)
    cv2.fillPoly(room, [np.asarray(space["polygon_px"], np.int32)], 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    near_room = cv2.dilate(room, k).astype(bool)
    outside_free = fm.free & ~cv2.dilate(room, k).astype(bool)
    near_outside = cv2.dilate(outside_free.astype(np.uint8), k).astype(bool)
    return fm.wall & near_room & near_outside


def region_for(fm: FloorMap, support: np.ndarray, beta: float, r: float, sr: float, sb: float, P: dict):
    """Union of visibility annuli around sign-support points. Returns mask, heading sum (cos, sin), witness count."""
    H, W = fm.img.shape
    ys, xs = np.nonzero(support)
    n = len(xs)
    if n == 0:
        return np.zeros((H, W), bool), None, None, 0
    stride = max(1, int(math.ceil(n / P["max_support_pts"])))
    pts = np.c_[xs, ys][::stride].astype(float)
    a_m = max(r - P["k_sigma"] * sr, P["min_range_m"])
    b_m = r + P["k_sigma"] * sr
    a_px, b_px = a_m * P["ppm_lo"], b_m * P["ppm_hi"]
    mask = np.zeros((H, W), bool)
    cs = np.zeros((H, W), np.float32)
    sn = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.int32)
    R = int(math.ceil(b_px)) + 1
    t = np.linspace(0.0, 1.0, P["los_steps"], endpoint=False)
    for sx, sy in pts:
        x0, x1 = max(0, int(sx) - R), min(W, int(sx) + R + 1)
        y0, y1 = max(0, int(sy) - R), min(H, int(sy) + R + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - sx, yy - sy)
        cand = (d >= a_px) & (d <= b_px) & fm.free[y0:y1, x0:x1]
        if not cand.any():
            continue
        px, py = xx[cand].astype(float), yy[cand].astype(float)
        dd = d[cand]
        frac_end = np.clip(1.0 - P["los_skip_px"] / np.maximum(dd, 1e-6), 0.0, 1.0)
        tt = t[None, :] * frac_end[:, None]
        rx = np.rint(px[:, None] + tt * (sx - px[:, None])).astype(int)
        ry = np.rint(py[:, None] + tt * (sy - py[:, None])).astype(int)
        np.clip(rx, 0, W - 1, out=rx)
        np.clip(ry, 0, H - 1, out=ry)
        clear = ~fm.wall[ry, rx].any(axis=1)
        if not clear.any():
            continue
        px, py = px[clear].astype(int), py[clear].astype(int)
        psi = np.arctan2(-(sy - py), sx - px) - beta
        mask[py, px] = True
        cs[py, px] += np.cos(psi)
        sn[py, px] += np.sin(psi)
        cnt[py, px] += 1
    return mask, cs, sn, int(len(pts))


def components(mask: np.ndarray, cs, sn, P: dict, min_px: int = 4):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_px:
            continue
        sel = lab == i
        c, s = float(cs[sel].sum()), float(sn[sel].sum())
        R = math.hypot(c, s) / max(float(np.count_nonzero(sel)), 1.0)
        comps.append(dict(id=i, area_px=area,
                          area_m2=[area / P["ppm_hi"] ** 2, area / P["ppm_lo"] ** 2],
                          bbox_px=[int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3])],
                          centroid_px=[float(cent[i, 0]), float(cent[i, 1])],
                          heading_mean_deg=math.degrees(math.atan2(s, c)),
                          heading_resultant=R))
    comps.sort(key=lambda d: -d["area_px"])
    return comps


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _finite_pos(x) -> bool:
    return _finite(x) and float(x) > 0.0


def run_e6(packet: dict, floors, out_dir: Path, building: str = "kci", params: dict | None = None,
           candidate_poses: list | None = None, maps_root: Path = EVENT_MAPS) -> dict:
    """candidate_poses: optional E2/E3 hypotheses [{floor, x_px, y_px, yaw_rad(map metric)}] — scored, never required."""
    P = dict(DEFAULTS, **(params or {}))
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    obs = packet["observation"]
    if not (_finite_pos(obs.get("range_m")) and _finite(obs.get("bearing_body_rad"))
            and _finite_pos(obs.get("range_sigma_m")) and _finite_pos(obs.get("bearing_sigma_rad"))):
        # R1: 0/NaN/Inf/negative range is not clamped to min_range_m and no region is produced
        res = dict(schema="e6_result/v1", status="INPUT_INVALID", reason_code="GEOMETRY_UNAVAILABLE",
                   why=f"invalid observation range/bearing: {dict((k, obs.get(k)) for k in ('range_m', 'bearing_body_rad', 'range_sigma_m', 'bearing_sigma_rad'))}",
                   floor_hypotheses=[], packet_id=packet.get("packet_id"))
        (out_dir / "e6_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
        return res
    beta, r = float(obs["bearing_body_rad"]), float(obs["range_m"])
    sb, sr = float(obs["bearing_sigma_rad"]), float(obs["range_sigma_m"])
    labels = packet["semantic"]["label_alternatives"]          # [{label, p}]
    labels = [dict(label=norm_label(a["label"]), p=float(a["p"])) for a in labels if norm_label(a["label"])]
    hyps, evidence = [], []
    for fl in floors:
        fm = FloorMap(fl, maps_root)
        prov = fm.provenance()
        prov["frame_equivalence"] = frame_equivalence(building, fl) if maps_root == EVENT_MAPS \
            else dict(status="SYNTHETIC_MAP")
        like = 0.0
        best = None
        for alt in labels:
            sps = fm.spaces_with_label(alt["label"])
            like += alt["p"] * (1.0 if sps else P["label_miss_eps"])
            if sps and (best is None or alt["p"] > best[0]["p"]):
                best = (alt, sps)
        rec = dict(floor=fl, label_likelihood=like, provenance=prov)
        if best is None:
            rec.update(status="NO_LABEL_ON_FLOOR")
            evidence.append(rec)
            continue
        alt, sps = best
        support = np.zeros(fm.img.shape, bool)
        own_room = np.zeros(fm.img.shape, np.uint8)
        for sp in sps:
            support |= sign_support(fm, sp, P["support_band_px"])
            cv2.fillPoly(own_room, [np.asarray(sp["polygon_px"], np.int32)], 1)
        mask, cs, sn, nsup = region_for(fm, support, beta, r, sr, sb, P)
        # Rule: a room plate faces outward — the robot cannot read room X's plate from inside room X.
        b = P["support_band_px"]
        own = cv2.dilate(own_room, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * b + 1, 2 * b + 1))).astype(bool)
        mask &= ~own
        comps = components(mask, cs, sn, P) if mask.any() else []
        rec.update(label=alt["label"], spaces=[sp.get("id") for sp in sps], support_px=int(support.sum()),
                   support_pts_used=nsup, region_px=int(mask.sum()), components=comps,
                   status="REGION" if comps else "EMPTY_REGION")
        if comps:
            mpath = out_dir / f"e6_region_{fl}.png"
            cv2.imwrite(str(mpath), mask.astype(np.uint8) * 255)
            hp = np.full(mask.shape, np.nan, np.float32)
            hp[mask] = np.degrees(np.arctan2(sn[mask], cs[mask]))
            np.save(out_dir / f"e6_heading_deg_{fl}.npy", hp)
            ov = cv2.cvtColor(fm.img, cv2.COLOR_GRAY2BGR)
            ov[mask] = (0.5 * ov[mask] + [0, 0, 127]).astype(np.uint8)
            ov[support] = (255, 0, 0)
            cv2.imwrite(str(out_dir / f"e6_overlay_{fl}.png"), ov)
            rec.update(region_mask_png=str(mpath), heading_npy=str(out_dir / f"e6_heading_deg_{fl}.npy"))
            if candidate_poses:
                sc = []
                for c in candidate_poses:
                    if c.get("floor") != fl:
                        continue
                    x, y = int(round(c["x_px"])), int(round(c["y_px"]))
                    inside = bool(0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x])
                    dh = float(np.degrees(G.wrap(c["yaw_rad"] - math.radians(hp[y, x])))) if inside else None
                    sc.append(dict(c, inside_region=inside, heading_delta_deg=dh))
                rec["candidate_pose_check"] = sc
        hyps.append(rec) if comps else evidence.append(rec)
    Z = sum(h["label_likelihood"] for h in hyps) or 1.0
    for h in hyps:
        h["p_floor"] = h["label_likelihood"] / Z
    hyps.sort(key=lambda h: -h["p_floor"])
    res = dict(schema="e6_result/v1", packet_id=packet.get("packet_id"), building=building, params=P,
               observation=dict(bearing_body_deg=math.degrees(beta), range_m=r,
                                bearing_sigma_deg=math.degrees(sb), range_sigma_m=sr),
               floor_hypotheses=hyps, rejected=evidence,
               n_regions=sum(len(h["components"]) for h in hyps),
               status="OK" if hyps else "NO_HYPOTHESIS",
               sec=round(time.time() - t0, 3), gt_used=False)
    (out_dir / "e6_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description="E6 floor/region inference from an E5 observation packet")
    ap.add_argument("--packet", required=True)
    ap.add_argument("--floors", nargs="+", required=True, help="semantic-map floor folders, e.g. 1F 2F ... 9F")
    ap.add_argument("--building", default="kci")
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", default=None, help="optional json list of E2/E3 candidate poses")
    a = ap.parse_args()
    pk = json.loads(Path(a.packet).read_text(encoding="utf-8"))
    cands = json.loads(Path(a.candidates).read_text()) if a.candidates else None
    res = run_e6(pk, a.floors, Path(a.out), a.building, candidate_poses=cands)
    print(json.dumps({k: res[k] for k in ("status", "n_regions", "sec")}, ensure_ascii=False))
    for h in res["floor_hypotheses"]:
        print(f"  {h['floor']}: p={h['p_floor']:.3f} regions={len(h['components'])} px={h['region_px']}")


if __name__ == "__main__":
    main()
