"""The date a page states for itself: when its content is current, not when it was crawled."""
from __future__ import annotations

import re
from datetime import date

from textnorm import MONTHS

_DATE = (rf"(?:(?P<mon>{'|'.join(MONTHS)})\s+(?P<d1>\d{{1,2}}),\s+(?P<y1>\d{{4}}))"
         r"|(?:(?P<m2>\d{1,2})/(?P<d2>\d{1,2})/(?P<y2>\d{4}|\d{2}))"
         r"|(?:(?P<y3>\d{4})-(?P<m3>\d{2})-(?P<d3>\d{2}))")
# "Rates are effective as of 03/15/2031", "Last updated: March 3, 2026": a date right after a word like "effective" or "updated".
_STATED = re.compile(rf"(?:effective|updated|revised|last modified|published|posted)\b[^.\n]{{0,25}}?(?:{_DATE})", re.I)
_ANY = re.compile(_DATE)
DATELINE_WINDOW = 800  # a press release prints its date near the top


def _iso(match: re.Match) -> str:
    groups = match.groupdict()
    try:
        if groups.get("mon"):
            return date(int(groups["y1"]), MONTHS.index(groups["mon"].title()) + 1, int(groups["d1"])).isoformat()
        if groups.get("m2"):
            year = int(groups["y2"]) + (2000 if len(groups["y2"]) == 2 else 0)
            return date(year, int(groups["m2"]), int(groups["d2"])).isoformat()
        return date(int(groups["y3"]), int(groups["m3"]), int(groups["d3"])).isoformat()
    except (ValueError, TypeError):
        return ""


def page_date(markdown: str) -> str:
    """ISO date of the page: the latest keyword-anchored date, else the dateline near the top, else ''."""
    stated = [d for d in (_iso(m) for m in _STATED.finditer(markdown)) if d]
    if stated:
        return max(stated)
    top = _ANY.search(markdown[:DATELINE_WINDOW])
    return _iso(top) if top else ""


def dates_in(text: str) -> list[str]:
    """Every ISO date written in `text` (month names, M/D/YYYY, ISO)."""
    return [d for d in (_iso(m) for m in _ANY.finditer(text)) if d]
