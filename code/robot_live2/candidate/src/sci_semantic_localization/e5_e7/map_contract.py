"""map_contract.py — R1 (2026-10-16): one manifest format for the maps E2/UI/E6/E7 actually use, and the E7 live preflight.

Separated on purpose (R1 §9.3):
  content_version      = file sha256 (a 1-px door-gap edit changes it; it says nothing about the coordinate frame)
  shape                = PNG IHDR width/height (equal shape does NOT prove coordinate equivalence)
  origin/axis          = from a metric base_map.yaml if present, else the pixel-frame assumption, labelled as such
  scale_provenance     = metric yaml resolution if present, else UNKNOWN (no averaging of fitted ppm values)
  coordinate_frame     = PROVEN only by (a) byte-identical files or (b) an explicit proof record naming BOTH sha256s;
                         otherwise UNPROVEN.
Nothing here writes or rewrites a map.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

E7_LIVE_WIRED = False          # R1: E7 is NOT wired to the live pipeline in this stage (no automatic E7 run)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def png_shape(p: Path):
    """(height, width) from the PNG IHDR chunk (stdlib only); None if not a PNG."""
    with open(p, "rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", head[16:24])
    return [int(h), int(w)]


def _yaml_meta(png: Path):
    y = png.with_name("base_map.yaml")
    if not y.exists():
        return None
    meta = {}
    for line in y.read_text(errors="ignore").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


def entry(consumer: str, path) -> dict:
    p = Path(path)
    rec = dict(consumer=consumer, path=str(p), exists=p.exists())
    if not p.exists():
        rec.update(content_version=None, content_version_null_reason="MISSING_FILE", shape=None,
                   origin_axis=dict(status="UNKNOWN"), scale_provenance=dict(status="UNKNOWN"),
                   coordinate_frame=dict(status="UNPROVEN", why="missing file"))
        return rec
    meta = _yaml_meta(p)
    rec.update(sha256=_sha256(p), content_version=None, shape=png_shape(p),
               origin_axis=(dict(status="FROM_YAML", origin=meta.get("origin"), source=str(p.with_name("base_map.yaml")))
                            if meta else dict(status="ASSUMED_PIXEL_FRAME", convention="x right, y down, top-left origin",
                                              note="no metric base_map.yaml; pixel frame assumed, not proven")),
               scale_provenance=(dict(status="FROM_YAML", resolution_m_per_px=meta.get("resolution")) if meta and meta.get("resolution")
                                 else dict(status="UNKNOWN", note="no metric yaml; fitted ppm values are search results, not a scale")),
               coordinate_frame=dict(status="UNPROVEN", why="per-file entry; equivalence is decided pairwise"))
    rec["content_version"] = rec["sha256"][:16]
    return rec


def load_proofs(path) -> list:
    if not path or not Path(path).exists():
        return []
    d = json.loads(Path(path).read_text())
    return d if isinstance(d, list) else [d]


def compare(a: dict, b: dict, proofs=()) -> dict:
    """Pairwise coordinate-frame decision between two manifest entries."""
    res = dict(a=a.get("path"), b=b.get("path"), a_sha256=a.get("sha256"), b_sha256=b.get("sha256"),
               content_identical=bool(a.get("sha256") and a.get("sha256") == b.get("sha256")),
               shape_equal=bool(a.get("shape") and a.get("shape") == b.get("shape")))
    if not (a.get("exists") and b.get("exists")):
        res.update(coordinate_frame="UNPROVEN", reason="MISSING_FILE")
    elif res["content_identical"]:
        res.update(coordinate_frame="PROVEN", reason="IDENTICAL_FILE")
    else:
        hit = [p for p in proofs if {p.get("sha256_a"), p.get("sha256_b")} == {a.get("sha256"), b.get("sha256")}
               and p.get("verified")]
        if hit:
            res.update(coordinate_frame="PROVEN", reason="EXPLICIT_PROOF", proof=hit[0])
        else:
            res.update(coordinate_frame="UNPROVEN",
                       reason="CONTENT_DIFFERS_SHAPE_EQUAL_NO_PROOF" if res["shape_equal"] else "CONTENT_AND_SHAPE_DIFFER")
    return res


def build_manifest(consumers: dict) -> dict:
    """consumers: {name: path}. Returns {entries:[...], schema}."""
    return dict(schema="map_manifest/r1", entries=[entry(k, v) for k, v in consumers.items()])


def e7_live_preflight(map_comparisons=(), calibration=None, wired=None) -> dict:
    """Decide whether E7 may run on live data. Never runs E7. Blockers are listed separately:
      NOT_WIRED              E7 is not connected to the live pipeline (R1 default)
      MAP_CONTRACT_UNPROVEN  any E2-map vs E6/E7-map comparison not PROVEN
      CALIBRATION_UNAVAILABLE no usable sensor-odom beta for the observation window"""
    wired = E7_LIVE_WIRED if wired is None else wired
    blockers = []
    if not wired:
        blockers.append("NOT_WIRED")
    if not map_comparisons or any(c.get("coordinate_frame") != "PROVEN" for c in map_comparisons):
        blockers.append("MAP_CONTRACT_UNPROVEN")
    if not calibration or calibration.get("status") != "OK":
        blockers.append("CALIBRATION_UNAVAILABLE")
    return dict(status="ALLOWED" if not blockers else blockers[0], blockers=blockers, e7_run=False,
                map_comparisons=list(map_comparisons), calibration=calibration)
