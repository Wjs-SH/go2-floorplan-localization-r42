#!/usr/bin/env python3
"""On-board OCR: PP-OCRv4 (RapidOCR / onnxruntime CPU, thread-limited) for digits + Latin, EasyOCR (GPU) for Korean.

Bounded retry policy (max 3 attempts, stop early on a confident room number):
  1. PP-OCRv4 on the sign crop
  2. PP-OCRv4 on 2x upscale + CLAHE
  3. EasyOCR ko+en on the best variant (Korean names; also a second opinion on digits)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

EASY_DIR = Path.home() / "sci_ws" / "models" / "easyocr"
OCR_THREADS = int(os.environ.get("SCI_OCR_THREADS", "2"))      # keep CPU for FAST-LIO / RealSense


def clahe(x):
    l, a, b = cv2.split(cv2.cvtColor(x, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


PREPROCESS_DOC = "resize max_side 1280 (INTER_AREA) -> variants: ppocr_crop, ppocr_clahe (x2 INTER_CUBIC if max side < 640)"


class OcrEngine:
    """Engines are created only when this class is instantiated (nothing is loaded at import)."""

    def __init__(self, use_easy=True, easy_gpu=True):
        import rapidocr_onnxruntime as _ro
        from rapidocr_onnxruntime import RapidOCR
        t0 = time.time()
        self.rapid = RapidOCR(intra_op_num_threads=OCR_THREADS, inter_op_num_threads=1)
        self.easy = None
        if use_easy:
            import easyocr
            self.easy = easyocr.Reader(["ko", "en"], gpu=easy_gpu, model_storage_directory=str(EASY_DIR),
                                       download_enabled=False, verbose=False)
        self.load_s = time.time() - t0
        # R1: engine/provider strings come from what was actually instantiated (no EasyOCR / GPU claim when unused)
        self.info = dict(
            engine="RapidOCR PP-OCRv4" + (" + EasyOCR(ko,en)" if self.easy is not None else ""),
            rapidocr_version=getattr(_ro, "__version__", None),
            rapidocr_provider=f"onnxruntime CPU (RapidOCR default, use_cuda not set; intra_op_threads={OCR_THREADS})",
            easyocr=None if self.easy is None else dict(gpu=bool(easy_gpu), model_dir=str(EASY_DIR)),
            preprocess=PREPROCESS_DOC)

    def describe(self):
        return dict(self.info)

    def rapid_lines(self, img):
        res, _ = self.rapid(img)
        return [(r[1], float(r[2])) for r in (res or [])]

    def easy_lines(self, img):
        return [(t, float(c)) for _, t, c in self.easy.readtext(img)] if self.easy else []

    def read(self, img, classify, confident=0.85, max_attempts=3, max_side=1280):
        """Bounded: input limited to max_side; upscale only small crops; EasyOCR only if no confident room number."""
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            s_ = max_side / max(h, w)
            img = cv2.resize(img, (int(w * s_), int(h * s_)), interpolation=cv2.INTER_AREA)
        up = 2 if max(img.shape[:2]) < 640 else 1
        variants = [("ppocr_crop", lambda: self.rapid_lines(img)),
                    ("ppocr_clahe" + ("_x2" if up > 1 else ""),
                     lambda: self.rapid_lines(clahe(cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up > 1 else img)))]
        if self.easy is not None:
            variants.append(("easyocr_ko_en", lambda: self.easy_lines(clahe(img))))
        attempts, merged = [], []
        for name, fn in variants[:max_attempts]:
            t0 = time.time()
            lines = fn()
            c = classify(lines)
            attempts.append(dict(engine=name, sec=round(time.time() - t0, 3), lines=lines, sign_type=c["sign_type"], rooms=c["rooms"]))
            merged += lines
            if c["rooms"] and max(r[1] for r in c["rooms"]) >= confident:
                break                                   # confident room number: stop retrying
        return dict(attempts=attempts, merged=merged, result=classify(merged), input_px=list(img.shape[:2]))


def sign_crop(img, center=None, frac=0.5):
    """LEGACY centre crop. R1 reader no longer uses it (verified target ROI + margin, else the full image)."""
    h, w = img.shape[:2]
    cx, cy = center if center else (w / 2, h / 2)
    hw, hh = w * frac / 2, h * frac / 2
    x0, y0, x1, y1 = int(max(0, cx - hw)), int(max(0, cy - hh)), int(min(w, cx + hw)), int(min(h, cy + hh))
    return img[y0:y1, x0:x1]
