#!/usr/bin/env python3
"""Sign-type classification of OCR/VLM text and OCR-VLM fusion.

Only a reading that is (a) a room-number-like token and (b) NOT contradicted by poster / floor-directory / elevator
evidence becomes a location label. Numbers on posters, elevator buttons and floor directories are never labels.
When OCR and VLM disagree, both alternatives are kept with split probability and status CONFLICT (no automatic pick).
"""
from __future__ import annotations

import re

ROOM_RE = re.compile(r"^(?:[A-Z]{1,2}-?)?B?\d{3,4}(?:-\d{1,2})?(?:호|호실)?$")
FLOOR_TOKEN_RE = re.compile(r"^(?:B?\d{1,2}F|\d{1,2}층|B\d)$", re.I)
DATE_RE = re.compile(r"(?:20\d\d|\d{1,2}[./]\d{1,2}(?:[./]\d{1,2})?|\d{1,2}:\d\d)")
POSTER_WORDS = ("모집", "신청", "행사", "세미나", "특강", "대회", "기간", "일시", "장소", "접수", "공모", "안내문", "프로그램",
                "http", "www", ".com", ".kr", "@", "toeic", "event", "seminar", "contest", "apply", "notice")
LAB_WORDS = ("연구실", "실험실", "강의실", "교수", "사무실", "센터", "학과", "사업단", "lab", "office", "center", "room", "실")
ELEV_WORDS = ("엘리베이터", "승강기", "elevator", "lift", "층별", "층 안내", "floor guide", "directory", "현재층")

OCR_ALONE_MIN = 0.85          # OCR-only room number accepted as a label only above this recogniser confidence
SIGN_TYPES = ("room_number_plate", "lab_or_office_name_sign", "floor_directory", "elevator_sign_or_button",
              "poster_or_notice", "other")


def norm(t: str) -> str:
    t = t.strip().strip("'\"`.,:;[](){}").replace(" ", "").upper()
    return t.replace("O", "0") if re.fullmatch(r"[0-9O]{3,4}", t) else t


def classify_text(lines):
    """lines: [(text, conf)] -> dict(sign_type, scores, rooms=[(label, conf)], evidence)."""
    texts = [(norm(t), float(c)) for t, c in lines if t and t.strip()]
    low = " ".join(t for t, _ in lines).lower()
    rooms = [(t.replace("호실", "").replace("호", ""), c) for t, c in texts if ROOM_RE.match(t)]
    floor_tok = sum(1 for t, _ in texts if FLOOR_TOKEN_RE.match(t))
    single_digits = sum(1 for t, _ in texts if re.fullmatch(r"B?\d{1,2}", t))
    poster = sum(w in low for w in POSTER_WORDS) + len(DATE_RE.findall(low)) + (1 if len(texts) >= 8 else 0)
    elev = sum(w in low for w in ELEV_WORDS) + (1 if floor_tok >= 2 else 0) + (1 if single_digits >= 4 else 0)
    lab = sum(w in low for w in LAB_WORDS)
    sc = {"poster_or_notice": poster, "elevator_sign_or_button": elev if floor_tok < 3 else 0,
          "floor_directory": elev if floor_tok >= 3 else 0, "lab_or_office_name_sign": lab,
          "room_number_plate": 2 * len(rooms)}
    best = max(sc, key=sc.get) if max(sc.values()) > 0 else "other"
    if rooms and best != "room_number_plate" and sc[best] < sc["room_number_plate"] + 1:
        best = "room_number_plate"
    return dict(sign_type=best, scores=sc, rooms=rooms, n_lines=len(texts),
                evidence=dict(floor_tokens=floor_tok, single_digits=single_digits))


def is_location_type(t):
    return t in ("room_number_plate", "lab_or_office_name_sign")


def fuse(ocr: dict | None, vlm: dict | None):
    """ocr/vlm: classify_text-like dicts (vlm may carry sign_type from the model). Returns label_alternatives + status."""
    o_type = ocr["sign_type"] if ocr else None
    v_type = vlm.get("sign_type") if vlm else None
    o_room = max(ocr["rooms"], key=lambda r: r[1]) if ocr and ocr["rooms"] else None
    v_room = (vlm.get("room_number") or None) if vlm else None
    v_room = norm(v_room) if v_room and ROOM_RE.match(norm(v_room)) else None
    bad = [t for t in (o_type, v_type) if t and t in ("poster_or_notice", "floor_directory", "elevator_sign_or_button")]
    if bad and not (o_room and v_room and o_room[0] == v_room):
        return dict(status="NOT_A_LOCATION_SIGN", sign_type=bad[0], label_alternatives=[], ocr_type=o_type, vlm_type=v_type)
    if o_room and v_room:
        if o_room[0] == v_room:
            return dict(status="AGREE", sign_type="room_number_plate",
                        label_alternatives=[dict(label=v_room, p=max(0.9, o_room[1]))], ocr_type=o_type, vlm_type=v_type)
        return dict(status="CONFLICT", sign_type="room_number_plate", ocr_type=o_type, vlm_type=v_type,
                    label_alternatives=[dict(label=o_room[0], p=0.5), dict(label=v_room, p=0.5)])
    if o_room:
        if o_room[1] >= OCR_ALONE_MIN:
            return dict(status="OCR_ONLY", sign_type=o_type, label_alternatives=[dict(label=o_room[0], p=round(0.8 * o_room[1], 3))],
                        ocr_type=o_type, vlm_type=v_type)
        return dict(status="OCR_ONLY_UNCONFIRMED", sign_type=o_type, label_alternatives=[], unconfirmed=o_room[0],
                    ocr_type=o_type, vlm_type=v_type)
    if v_room:
        # VLM hallucinated a room plate "201" on a glossy reflection (bench_vlm.py, 2026-10-14): never forward an
        # uncorroborated VLM number as a location cue
        return dict(status="VLM_ONLY_UNCONFIRMED", sign_type=v_type, label_alternatives=[], unconfirmed=v_room,
                    ocr_type=o_type, vlm_type=v_type)
    return dict(status="NO_ROOM_LABEL", sign_type=v_type or o_type or "other", label_alternatives=[], ocr_type=o_type, vlm_type=v_type)


def label_schema(lines, fusion):
    """R1 schema slots for future general semantic matching. The extraction rule is NOT widened: label_alternatives
    are exactly the fusion output (whole-line room tokens only); raw_text keeps every OCR line so lab/office names and
    mixed lines ('418호 연구실') are preserved for later, separately validated rules."""
    return dict(raw_text=[t for t, _ in (lines or [])], label_alternatives=list((fusion or {}).get("label_alternatives", [])),
                label_type=(fusion or {}).get("sign_type"), spans=None,
                spans_null_reason="span extraction not implemented in R1 (no parser widening)",
                parser="semantic.ROOM_RE whole-line (unchanged)")
