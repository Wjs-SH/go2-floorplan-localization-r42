#!/usr/bin/env python3
"""선배 SCI 판독기(reader.py)에 우리 VLM 질문을 얹는다. R1(2026-10-16) 수정판.

  - 팬틸트 사진(IMX477_PANTILT)을 'OCR 결과와 상관없이 항상 VLM' 으로 보내던 덮어쓰기를 없앴다. SCI 의 OCR 우선 규칙
    (OCR 이 확실하지 않을 때만 VLM)을 그대로 쓴다. OCR 결과는 VLM 과 따로 바로 공개된다(r1_jobs).
  - VLM 은 사진 전체를 우리 질문(read_map.PROMPT)으로 읽는다. 크기는 config/r1_reader.json vlm_max_side(기본 768:
    메모리 측정이 있는 해상도). 1280px 는 메모리 측정이 없어서 admission 이 거부한다(resource_profile_unverified).
  - 답은 SCI 융합 형식으로 옮기고(read_map.to_sci) 우리 답은 'ours' 로 같이 남긴다. 기대 정답·층은 질문에 넣지 않는다.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCI_SRC = Path(os.environ.get("SCI_WS", str(Path.home() / "sci_ws"))) / "src"
sys.path.insert(0, str(SCI_SRC / "sci_text_reading"))
sys.path.insert(0, str(HERE))

import reader as R            # noqa: E402  SCI reader (R1 판)
import vlm as V               # noqa: E402
from read_map import PROMPT, to_sci  # noqa: E402


class OurVlm(V.VlmServer):
    def ask(self, bgr, prompt=None, max_side=None, max_tokens=96, timeout=120):
        r = super().ask(bgr, prompt=PROMPT, max_side=max_side or 768, max_tokens=max_tokens, timeout=timeout)
        r["ours"] = r.get("parsed")
        r["parsed"] = to_sci(r.get("parsed"))
        r["prompt"] = "kci_hires_sign_judge"
        return r


class OurReader(R.Reader):
    """VLM 서버만 우리 질문판으로 바꾼다. want_vlm 은 덮어쓰지 않는다(OCR 우선)."""

    def make_vlm(self):
        return OurVlm()


R.VlmServer = OurVlm             # 옛 코드가 이 이름을 직접 쓰는 경우를 위해 남김

if __name__ == "__main__":
    print("reader_onboard(R1): OCR 우선 · VLM 은 OCR 이 확실하지 않고 자원이 될 때만(작업별 유한 대기) · 우리 VLM 질문", flush=True)
    R.main(reader_cls=OurReader)
