#!/usr/bin/env python3
"""Render a FAST-LIO param file from the template + a calibration file into a NEW run directory.
  render_fastlio_cfg.py CALIB.yaml OUT.yaml --mode live|offline-plumbing [--extrinsic-est]
live:              refuses unless status == CALIBRATION_VERIFIED and live_walk_allowed: true
offline-plumbing:  accepts NOMINAL/PARTIAL values (domain-42 replay/plumbing tests only); stamps the output as such."""
import argparse, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "..", "config", "fastlio", "fastlio_alt_imu.TEMPLATE.yaml")


def load_calib(p):
    t = open(p).read()
    def grab(key):
        m = re.search(r"^%s:\s*(\[[^\]]*\]|[^\n#]+)" % key, t, re.M | re.S)
        if not m:
            sys.exit("REFUSED: calibration file lacks '%s'" % key)
        return m.group(1).strip()
    nums = lambda s: [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]
    c = dict(status=grab("status"), live=grab("live_walk_allowed").lower() == "true", R=nums(grab("R")), T=nums(grab("T")),
             dt=float(grab("time_offset_lidar_to_imu")))
    if len(c["R"]) != 9 or len(c["T"]) != 3:
        sys.exit("REFUSED: R needs 9 and T needs 3 numbers")
    if not all(math.isfinite(x) for x in c["R"] + c["T"] + [c["dt"]]):
        sys.exit("REFUSED: non-finite value")
    R = [c["R"][0:3], c["R"][3:6], c["R"][6:9]]
    RtR = [[sum(R[k][i] * R[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    orth = max(abs(RtR[i][j] - (1.0 if i == j else 0.0)) for i in range(3) for j in range(3))
    det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
           + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
    if orth > 1e-3 or abs(det - 1.0) > 1e-3:
        sys.exit("REFUSED: R not a rotation (|RtR-I|max=%.2e det=%.4f)" % (orth, det))
    if abs(c["dt"]) > 0.2:
        sys.exit("REFUSED: |time_offset| > 0.2 s is implausible")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib"); ap.add_argument("out")
    ap.add_argument("--mode", choices=["live", "offline-plumbing", "live-static-diag", "live-walk-diag", "live-e2-body"], required=True)
    ap.add_argument("--extrinsic-est", action="store_true")
    a = ap.parse_args()
    c = load_calib(a.calib)
    if a.mode == "live" and not (c["status"] == "CALIBRATION_VERIFIED" and c["live"]):
        sys.exit("REFUSED live: calibration status %s, live_walk_allowed=%s (need CALIBRATION_VERIFIED + true)" % (c["status"], c["live"]))
    # live-static-diag: STATIC diagnostic on the real robot only (live_static_diag.sh: bounded time, motion guard, auto-stop).
    # Accepts the dynamic-field candidate or the nominal file; never writes live_walk_allowed and never relaxes the 'live' gate.
    DIAG_OK = ("ROTATION_AND_DT_ESTIMATED_T_NOMINAL", "PARTIAL_ROLL_PITCH_ONLY", "CALIBRATION_VERIFIED")
    # live-walk-diag: ONE operator-driven short walk (live_walk_diag.sh: per-run approval file, divergence watchdog). Only a
    # data-derived candidate (not the URDF-nominal file) is accepted for walking.
    WALK_OK = ("ROTATION_AND_DT_ESTIMATED_T_NOMINAL", "CALIBRATION_VERIFIED")
    if a.mode == "live-static-diag" and c["status"] not in DIAG_OK:
        sys.exit("REFUSED live-static-diag: status %s not in %s" % (c["status"], DIAG_OK))
    if a.mode == "live-walk-diag" and c["status"] not in WALK_OK:
        sys.exit("REFUSED live-walk-diag: status %s not in %s" % (c["status"], WALK_OK))
    # live-e2-body: the body-IMU input mode of the LIVE2 'fastlio' module (scripts/body_fastlio_live2.sh -> l1_adapter_node -> E2..E8).
    # Same calibration requirement as the walk diagnostic (data-derived candidate); the status stays what it is (not 'calibrated').
    if a.mode == "live-e2-body" and c["status"] not in WALK_OK:
        sys.exit("REFUSED live-e2-body: status %s not in %s" % (c["status"], WALK_OK))
    if a.mode in ("live-static-diag", "live-walk-diag", "live-e2-body") and a.extrinsic_est:
        sys.exit("REFUSED %s: online extrinsic estimation is not part of the diagnostic" % a.mode)
    if os.path.exists(a.out):
        sys.exit("REFUSED: %s exists (render into a new run directory)" % a.out)
    t = open(TEMPLATE).read()
    fmt = lambda v: "[" + ", ".join(repr(float(x)) for x in v) + "]"   # always float literals (vector<double> params)
    t = (t.replace("__EXTRINSIC_R__", fmt(c["R"])).replace("__EXTRINSIC_T__", fmt(c["T"]))
          .replace("__TIME_OFFSET__", "%.6f" % c["dt"]).replace("__EXTRINSIC_EST__", "true" if a.extrinsic_est else "false"))
    if re.search(r"__(EXTRINSIC_R|EXTRINSIC_T|TIME_OFFSET|EXTRINSIC_EST)__", t):
        sys.exit("REFUSED: unfilled placeholder")
    t = ("# RENDERED %s from %s (status %s)\n" % (a.mode.upper(), os.path.abspath(a.calib), c["status"])) + \
        ("# STATIC DIAGNOSTIC ONLY — PROVISIONAL EXTRINSIC (translation unobserved). NOT A WALK PERMISSION. NOT FOR E2-E8.\n"
         if a.mode == "live-static-diag" else
         "# WALK DIAGNOSTIC (one short operator-driven walk) — PROVISIONAL EXTRINSIC (translation unobserved). NOT FOR E2-E8.\n"
         if a.mode == "live-walk-diag" else
         "# LIVE2 BODY-IMU INPUT MODE (outputs go through l1_adapter_node) — PROVISIONAL EXTRINSIC (translation unobserved, status kept).\n"
         if a.mode == "live-e2-body" else "") + t
    open(a.out, "w").write(t)
    print("rendered", a.out, "mode", a.mode, "status", c["status"])


if __name__ == "__main__":
    main()
