"""E5-RT (2026-10-17): real-time sign confirmation layer on top of the student deploy (kci_r1_20261017).

Pure-Python/OpenCV modules (no torch/ROS imports at module level) so they can be unit-tested without devices:
  contracts      cause codes, gate state, E4 contract, ObservationCandidate
  gates          observe / request / armed / operator-link / trial-logging / global-match separation
  tracker        bounded sparse-LK ROI tracker with forward-backward check (geometry evidence, not semantics)
  scout          multi-rate candidate discovery (downsampled scout + periodic full-resolution rescue)
  scheduler      per-track SigLIP verification scheduling, cache (never counted as new evidence), latest-slot queue
  evidence       temporal confirmation policies: legacy mean3, max (comparison only), EMA+hysteresis, quality-aware
  telemetry      explicit per-scene timing record helper (NOT yet wired; scene timing fields are added in frontend.py)
  status_writer  single-writer revisioned status file (no stale snapshot overwrite)
  handoff        ObservationCandidate for pan/tilt + reader (same target/frame/geometry provenance)
  live_render    live image = tracked boxes only (with age); inference panel = source frame + its own boxes
  home_policy    HOME = cleanup of this event's own action via the device owner (pure)
  rt_frontend    per-track SigLIP cache wrapper between legacy frontend and worker (opt-in)
(vision-only / lazy context / GPU normalisation live in ../siglip_worker.py)
"""
E5RT_VERSION = "e5rt-0.1-20261017"
