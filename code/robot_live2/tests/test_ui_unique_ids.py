"""STAGING (2026-10-30): every id in web_v2/index.html is unique. A duplicate id made $("e6m") return the new E6-mode <select>
instead of the existing E6 manual-focus card, so the card could never be shown (found in the 16:xx audit, fixed: select -> e6mode)."""
import collections, re, sys
from pathlib import Path
p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "candidate/live2/web_v2/index.html"
ids = re.findall(r'\bid="([^"]+)"', p.read_text())
dup = {k: v for k, v in collections.Counter(ids).items() if v > 1}
print(f"{p}: {len(ids)} ids, duplicates: {dup or 'none'}")
sys.exit(1 if dup else 0)
