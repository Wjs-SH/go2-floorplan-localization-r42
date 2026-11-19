"""SEM1: data-integrity checks used by the label/ablation tools and tests.

verify_encoder_input(image_path, meta_tile): an image may be called an ENCODER INPUT only if it is the lossless 224x224
  uint8 RGB crop whose sha256 equals the dump record (a JPEG context preview / re-encoded image fails).
split_leaks(items): no physical_target_id (or capture session group) may appear in more than one split.
"""
from __future__ import annotations

import hashlib


def verify_encoder_input(image_path, meta_tile, size=224):
    import cv2
    p = str(image_path)
    if p.lower().endswith((".jpg", ".jpeg")):
        return False, "LOSSY_JPEG_IS_NOT_ENCODER_INPUT"
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        return False, "UNREADABLE"
    if im.shape[:2] != (size, size):
        return False, f"SHAPE_{im.shape[1]}x{im.shape[0]}_NOT_{size}"
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if hashlib.sha256(rgb.tobytes()).hexdigest() != meta_tile.get("rgb_uint8_sha256"):
        return False, "SHA_MISMATCH_WITH_DUMP_RECORD"
    return True, "EXACT_ENCODER_INPUT"


def split_leaks(items, group_key="physical_target_id", split_key="split"):
    seen = {}
    for it in items:
        g = it.get(group_key)
        if not g:
            continue
        seen.setdefault(g, set()).add(it.get(split_key))
    return {g: sorted(s) for g, s in seen.items() if len(s) > 1}
