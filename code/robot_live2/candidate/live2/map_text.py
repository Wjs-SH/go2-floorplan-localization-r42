#!/usr/bin/env python3
"""map_text.py (student staging, 2026-11-12): floor evidence from the text printed on the guide map itself. No hand-written department table.

plate_text.DEPT lists this building's departments by hand (로봇 -> 4F, 전기공 -> 5/6F ...). That list cannot move to another building. Here the same kind of evidence is
derived from the map alone: the printed text of every room (re-read from the guide-map photo, one room crop at a time; map_text_rooms.json) is cut into jamo trigrams.
A trigram that repeats on one floor and nowhere else votes for that floor; one found on every floor (공학과, 강의실) votes for none. A plate's text gets a score per floor and
its evidence is the MARGIN of the first floor over the second. Room digits are removed before scoring, so this stays independent of the room number (the point of R38).

    n_f(g)  rooms of floor f whose text contains trigram g
    p(f|g)  (n_f(g) + B/F) / (sum_f n_f(g) + B)            a trigram seen once is weak
    I(g)    1 - H(p(.|g)) / log F                           0 on every floor, near 1 when repeated on one floor only
    S_f(q)  sum over g in q of I(g) * p(f|g)                margin = S_first - S_second

Switch LIVE2_MAP_TEXT:  unset / 0  never called, R42 behaviour unchanged
                        shadow     judged and recorded under 'map_text' of the shadow record, decides nothing
                        1          used INSTEAD of plate_text.DEPT: floor evidence MAP_TEXT_ON_2_PLATES, MAP_TEXT_CONFLICT, and the VLM-only room number (R40)

Measured offline on stored field photos of 4F (43 VLM readings with the department, 19 invented or empty readings): all 43 first on 4F with margin 1.68-2.83, invented <= 1.04.
PLATE_MIN sits between those two; it comes from ONE floor of ONE building and is provisional. Map-against-map leave-one-room-out on three buildings: see the student notes."""
import json
import math
import os
import re
from pathlib import Path

B = 1.0
PLATE_MIN = 1.5          # a plate's text supports (or contradicts) a floor from this margin on
MIN_PLATES = 2           # plates with DIFFERENT room numbers
_LEX = {}


def mode():
    v = str(os.environ.get("LIVE2_MAP_TEXT", "")).strip().lower()
    return "use" if v in ("1", "use", "on", "true") else "shadow" if v == "shadow" else ""


def norm(s):
    """Letters only (Hangul, Latin, CJK, kana); digits, spaces and punctuation go."""
    s = str(s or "").upper()
    return "".join(ch for ch in s if ("가" <= ch <= "힣") or ("A" <= ch <= "Z") or ("一" <= ch <= "鿿") or ("぀" <= ch <= "ヿ"))


def jamo(s):
    out = []
    for ch in s:
        if "가" <= ch <= "힣":
            c = ord(ch) - 0xAC00
            out += ["c%d" % (c // 588), "v%d" % ((c % 588) // 28)] + (["t%d" % (c % 28)] if c % 28 else [])
        else:
            out.append(ch)
    return out


def grams(s):
    j = jamo(norm(s))
    return {"".join(j[i:i + 3]) for i in range(len(j) - 2)}


class Lex:
    def __init__(self, floors):
        """floors: {floor int: [(room number or None, text)]}"""
        self.floors = sorted(int(f) for f in floors)
        self.rooms = {}                       # room number -> {floors}
        self.n = {}                           # trigram -> {floor: rooms}
        for f, rows in floors.items():
            for num, text in rows:
                if num:
                    self.rooms.setdefault(str(num), set()).add(int(f))
                for g in grams(text):
                    d = self.n.setdefault(g, {})
                    d[int(f)] = d.get(int(f), 0) + 1

    def score(self, text):
        F = len(self.floors)
        S = {f: 0.0 for f in self.floors}
        G = grams(text)
        hit = 0
        for g in G:
            cnt = self.n.get(g)
            if not cnt:
                continue
            hit += 1
            tot = sum(cnt.values())
            p = {f: (cnt.get(f, 0) + B / F) / (tot + B) for f in self.floors}
            info = max(0.0, 1 - (-sum(v * math.log(v) for v in p.values() if v > 0)) / math.log(F))
            for f in self.floors:
                S[f] += info * p[f]
        o = sorted(S, key=lambda f: (-S[f], f))
        return dict(top=o[0], second=o[1], margin=round(S[o[0]] - S[o[1]], 3), score=round(S[o[0]], 3), grams=len(G), known=hit, floors={f: round(v, 3) for f, v in S.items()})


def lexicon(path=None):
    """Lex of the building, or None when the room-text file is missing (then nothing here is used)."""
    p = Path(path or os.environ.get("LIVE2_MAP_TEXT_FILE") or Path(__file__).with_name("map_text_rooms.json"))
    key = str(p)
    if key not in _LEX:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            _LEX[key] = Lex({int(f): [(r[0], r[1]) for r in rows] for f, rows in d["floors"].items()})
        except Exception:
            _LEX[key] = None
    return _LEX[key]


def plate_text(o, sources=("OCR", "VLM")):
    """The words of THIS plate as plate_text.plate_texts gives them (crop only, number excluded), joined."""
    import plate_text as PT
    return " ".join(t for t, src in PT.plate_texts(o) if src in sources)


def judge(plates, lex=None):
    """plates: [(observation, digit floor, room label)]. A plate supports its digit floor when its own text scores that floor first by >= PLATE_MIN, and contradicts it when
    another floor is first by >= PLATE_MIN. ok: no contradiction and >= MIN_PLATES supporting plates with different room numbers. -> dict"""
    lex = lexicon() if lex is None else lex
    if lex is None:
        return dict(available=False, ok=False, support=[], support_rooms=[], conflicts=[], per_plate=[])
    per, support, rooms, conflicts = [], [], [], []
    for o, f, label in plates:
        text = plate_text(o)
        if not norm(text):
            continue
        r = lex.score(text)
        row = dict(observation_id=o.get("observation_id"), room=label, digit_floor=f, text=text[:40], top=r["top"], second=r["second"], margin=r["margin"], on_digit_floor=r["floors"].get(f))
        per.append(row)
        if r["margin"] >= PLATE_MIN and r["top"] == f:
            support.append(o.get("observation_id"))
            rooms.append(str(label))
        elif r["margin"] >= PLATE_MIN:
            conflicts.append(row)
    distinct = sorted(set(rooms))
    return dict(available=True, ok=not conflicts and len(distinct) >= MIN_PLATES, support=support, support_rooms=distinct, conflicts=conflicts, per_plate=per, plate_min=PLATE_MIN)


def vlm_only_room(o, lex=None):
    """plate_text.vlm_only_room (R40) without the department table: a room number read by the VLM alone counts when the same sharp crop's words score, by >= PLATE_MIN,
    a floor on which the map has that room number, and OCR read no other number. -> room or None"""
    import plate_text as PT
    lex = lexicon() if lex is None else lex
    v, meta = o.get("vlm"), ((o.get("reading") or {}).get("vlm") or {})
    if lex is None or not (isinstance(v, dict) and meta.get("crop") and v.get("sign_type") == "room_number_plate"):
        return None
    room = str(v.get("room_number") or "").strip()
    if not re.fullmatch(r"\d{3}", room) or float((o.get("image_quality") or {}).get("roi_laplacian_var") or 0.0) < PT.VLM_ONLY_MIN_SHARP:
        return None
    if any(re.search(r"\d{3}", str(t)) and float(c) >= 0.5 for t, c in (o.get("ocr_lines") or [])):
        return None
    text = plate_text(o, sources=("VLM",))
    if not norm(text):
        return None
    r = lex.score(text)
    return room if r["margin"] >= PLATE_MIN and r["top"] in lex.rooms.get(room, ()) else None
