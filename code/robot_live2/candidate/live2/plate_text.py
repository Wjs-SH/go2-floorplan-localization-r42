#!/usr/bin/env python3
"""plate_text.py (R38): what a door plate says besides its number. The floor-independent evidence the 2-room rule was missing.

R37 confirmed '505 + 502' when OCR and VLM misread the same digit on both plates, because every other check depends only on those digits. A plate also carries a department and a person's name, read separately from the number. In this
building the departments differ by floor (map check in tests: 3F materials, 4F robotics, 5F electrical, 6F electronics, 1F automotive; 7-9F computer-related). So a misread digit contradicts the rest of its own plate: 505 with 로봇공학과 / 가나다 is rejected.

Only text read from the plate crop counts: the VLM invents departments on a whole frame (정신건강, 내과) and reads them from a crop (로봇공학과 on 4 of 4 readable field plates). Whole-frame VLM text is ignored, in both directions. The lexicon was fixed from the semantic map, not from the field photos."""
import re

# (pattern, floors). A word on several floors supports none of them, but it can still contradict others.
DEPT = ((re.compile(r"로봇|로보트|로보공|로부공"), {4}), (re.compile(r"전기\s*공"), {5, 6}), (re.compile(r"전자\s*공"), {6}), (re.compile(r"신소재"), {3}), (re.compile(r"자동차"), {1}),
        (re.compile(r"정보통신"), {7}), (re.compile(r"소프트웨어"), {8, 9}), (re.compile(r"컴퓨터\s*공"), {6, 7, 8}))
_NAME = re.compile(r"[가-힣]{3}$")
GENERIC = {"공학과", "강의실", "실험실", "연구실", "화회실", "관리실", "운영실", "실습실", "세미나", "공과대", "학습실", "프린터", "편의점", "기계실", "발전기"}


def person_names(maps):
    """{3-syllable last word of a room name: {(floor, room id)}} from the semantic maps. Only names that appear once in the building count."""
    seen = {}
    for f, m in (maps or {}).items():
        for s in m.get("spaces") or []:
            words = str(s.get("name") or "").split()
            if s.get("room_number") and len(words) >= 2 and _NAME.fullmatch(words[-1]) and words[-1] not in GENERIC and not re.search(r"[실심신과터래]$", words[-1]):
                seen.setdefault(words[-1], set()).add((int(f), s.get("id")))
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


def plate_texts(o):
    """[(text, source)] read from THIS plate, the room number excluded."""
    out = [(str(t), "OCR") for t, c in (o.get("ocr_lines") or []) if float(c) >= 0.5 and not re.fullmatch(r"[\d\-.\s]*", str(t))]
    v, meta = o.get("vlm"), ((o.get("reading") or {}).get("vlm") or {})
    if isinstance(v, dict) and meta.get("crop"):
        for k in ("text", "department", "name"):
            if v.get(k):
                out.append((str(v[k]), "VLM"))
    return out


VLM_ONLY_MIN_SHARP = 20.0     # field 11-03: readable plates 25.7-44.3, blurred photos 3.0-3.2 where the VLM still invents a plate (O008 '405', O010 '410')


def vlm_only_room(o):
    """R40 (X20261103_001810 O002: the photo shows '407 로봇공학과 차카타', OCR found nothing, the VLM read all of it, and it was thrown away as VLM-only; 407 + 405 were two real plates and no FIX).
    A room number from the VLM alone counts when the same crop's department belongs on that number's floor, the crop is sharp and OCR read no other number. A bare VLM number ('202', '403') never does. -> room or None."""
    v, meta = o.get("vlm"), ((o.get("reading") or {}).get("vlm") or {})
    if not (isinstance(v, dict) and meta.get("crop") and v.get("sign_type") == "room_number_plate"):
        return None
    room = str(v.get("room_number") or "").strip()
    if not re.fullmatch(r"\d{3}", room) or float((o.get("image_quality") or {}).get("roi_laplacian_var") or 0.0) < VLM_ONLY_MIN_SHARP:
        return None
    if any(re.search(r"\d{3}", str(t)) and float(c) >= 0.5 for t, c in (o.get("ocr_lines") or [])):
        return None
    fl = int(room[0])
    ev = [e for e in plate_evidence(o, None, {}) if e["kind"] == "DEPT" and e["source"] == "VLM"]
    return room if ev and all(fl in e["floors"] for e in ev) else None


def plate_evidence(o, maps=None, names=None):
    """-> [dict(kind DEPT | NAME, token, floors, source, text)] for one observation."""
    names = person_names(maps) if names is None else names
    ev = []
    for text, src in plate_texts(o):
        for rx, floors in DEPT:
            m = rx.search(text)
            if m:
                ev.append(dict(kind="DEPT", token=m.group(0), floors=sorted(floors), source=src, text=text[:40]))
        for w in re.findall(r"[가-힣]{3}", text):
            if w in names:
                ev.append(dict(kind="NAME", token=w, floors=[names[w][0]], room=names[w][1], source=src, text=text[:40]))
    return ev


def judge(plates, maps=None):
    """plates: [(observation, digit floor)]. -> dict(conflicts, support {kind: [...]}, ok, why). Conflict: a plate's own text belongs on other floors than its digits say."""
    names = person_names(maps)
    conflicts, dept, name = [], [], []
    for o, f in plates:
        ev = plate_evidence(o, maps, names)
        for e in ev:
            if f not in e["floors"]:
                conflicts.append(dict(e, observation_id=o.get("observation_id"), digit_floor=f))
        if any(e["kind"] == "DEPT" and e["floors"] == [f] for e in ev):
            dept.append(o.get("observation_id"))
        if any(e["kind"] == "NAME" and e["floors"] == [f] for e in ev):
            name.append(o.get("observation_id"))
    return dict(conflicts=conflicts, dept_plates=dept, name_plates=name, ok=not conflicts and (len(set(dept)) >= 2 or bool(name)))
