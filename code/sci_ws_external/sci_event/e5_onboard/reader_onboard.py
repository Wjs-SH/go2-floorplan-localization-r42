#!/usr/bin/env python3
"""선배 SCI 판독기(reader.py: OCR → 필요하면 Qwen3-VL llama.cpp → 융합 → E5 패킷)에 우리 판독 방식을 얹는다(2026-10-15).

  - 팬틸트 고화질 사진(IMX477_PANTILT)은 OCR 결과와 상관없이 항상 VLM 으로도 읽는다(우리 PC 판처럼 '사진이 나오면 바로 판독').
    단 SCI 메모리 안전장치(가용 메모리·GPU·온도)는 그대로 — 안 되면 OCR 결과만 먼저 내고 15 초마다 다시 시도.
  - VLM 에는 SCI 처럼 가운데 50% 를 자른 그림이 아니라 사진 전체(1280 폭 그대로)를 우리 질문으로 준다.
    답은 SCI 융합 형식으로 옮기고(read_map.to_sci) 우리 답은 'ours' 로 같이 남긴다.
  - OCR(가운데 50%)·융합 규칙(VLM 혼자 읽은 번호는 위치 단서로 안 넘김)·E5 패킷·E6 은 SCI 그대로.
"""
import os
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
SCI_SRC = Path(os.environ.get("SCI_WS", str(Path.home() / "sci_ws"))) / "src"
sys.path.insert(0, str(SCI_SRC / "sci_text_reading"))
sys.path.insert(0, str(HERE))

import reader as R            # noqa: E402  SCI 원본
import vlm as V               # noqa: E402
from read_map import PROMPT, to_sci  # noqa: E402

VLM_MAX_SIDE = int(os.environ.get("SCI_VLM_MAX_SIDE", "1280"))


class OurVlm(V.VlmServer):
    full = None                  # 지금 판독 중인 팬틸트 사진 전체(없으면 SCI 가 넘긴 그림)

    def ask(self, bgr, prompt=None, max_side=None, max_tokens=96, timeout=120):
        img = self.full if self.full is not None else bgr
        r = super().ask(img, prompt=PROMPT, max_side=max_side or VLM_MAX_SIDE, max_tokens=max_tokens, timeout=timeout)
        r["ours"] = r.get("parsed")
        r["parsed"] = to_sci(r.get("parsed"))
        r["prompt"] = "kci_hires_sign_judge"
        return r


R.VlmServer = OurVlm             # SCI Reader.__init__ 이 이 이름으로 만든다(상주 모드도 우리 것 하나만 뜸)


class OurReader(R.Reader):
    _src = None

    def want_vlm(self, ocr_res):
        if self._src == "IMX477_PANTILT":
            return True, "pantilt_capture_always"
        return super().want_vlm(ocr_res)

    def process(self, job_path):
        import json
        job = json.loads(Path(job_path).read_text())
        self._src = job.get("capture_source")
        if isinstance(self.vlm, OurVlm):
            self.vlm.full = cv2.imread(job["image"]) if (job.get("image") and self._src == "IMX477_PANTILT") else None
        try:
            return super().process(job_path)
        finally:
            if isinstance(self.vlm, OurVlm):
                self.vlm.full = None
            self._src = None


R.Reader = OurReader

if __name__ == "__main__":
    print(f"reader_onboard: 팬틸트 사진은 항상 VLM(전체 그림 최대 {VLM_MAX_SIDE}px, 우리 질문) · 그 밖은 SCI 그대로", flush=True)
    R.main()
