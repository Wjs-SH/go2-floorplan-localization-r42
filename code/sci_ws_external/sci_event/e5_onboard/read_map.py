"""우리 판독 질문(PC 판 live_read_worker 와 같은 문장)과, 그 답을 SCI 융합 형식으로 옮기는 표(2026-10-15)."""
import re

# 408호 앞 팬틸트 사진 11장 중 10장을 '408' 로 읽은 질문(scripts/live_read_worker.py, hires_sign_judge)
PROMPT = ("Look at the sign or object closest to the image center. Classify it as exactly one of: "
          "room_plate (a room number or room name plate), restroom (toilet sign), elevator, exit (emergency exit sign), "
          "fire (fire alarm or extinguisher box), notice (paper notice or poster), none. "
          'Answer only JSON with two keys: {"type": "<one of the types>", "text": "<text you can read on that sign>"}. '
          "If you cannot read any text, use an empty string. Do not guess text that is not visible.")

TYPE_TO_SCI = {"room_plate": "room_number_plate", "elevator": "elevator_sign_or_button", "notice": "poster_or_notice"}
ROOM = re.compile(r"(?<![0-9A-Za-z])([A-Za-z]?\d{3,4}(?:-\d)?[A-Za-z]?)(?![0-9])")


def to_sci(ours):
    """{"type","text"} → SCI {"sign_type","room_number","text"}. 번호는 호실판일 때만 글자에서 뽑는다."""
    if not isinstance(ours, dict):
        return None
    t = str(ours.get("type") or "").strip()
    text = str(ours.get("text") or "")
    room = ""
    if t == "room_plate":
        m = ROOM.search(text)
        room = m.group(1) if m else ""
    return {"sign_type": TYPE_TO_SCI.get(t, "other"), "room_number": room, "text": text, "ours": ours}
