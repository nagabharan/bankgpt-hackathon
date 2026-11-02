"""Dates written inside a question ("as of May 2026", "in March") as a snapshot date."""
from __future__ import annotations

import calendar
import re
from typing import Optional

from textnorm import MONTHS

# A month needs a year or a preposition: a bare capitalised "May" is usually the verb ("May I apply?").
# "since"/"from" are not here: they ask what changed, not what was true then.
_DATE_PHRASE = re.compile(
    rf"\b(?P<prep>(?:as of|in|during)\s+)?(?P<month>{'|'.join(MONTHS)})(?:\s+(?P<year>\d{{4}}))?\b")


def split_date(question: str, default_year: int) -> tuple[str, Optional[str]]:
    """(question without the date phrase, last day of that month as 'YYYY-MM-DD' or None).

    A month without a year takes `default_year` (the year of the newest snapshot)."""
    match = next((m for m in _DATE_PHRASE.finditer(question) if m.group("prep") or m.group("year")), None)
    if match is None:
        return question, None
    year = int(match.group("year") or default_year)
    month = MONTHS.index(match.group("month")) + 1
    last_day = calendar.monthrange(year, month)[1]
    stripped = (question[:match.start()] + question[match.end():]).replace("  ", " ").strip()
    return stripped, f"{year}-{month:02d}-{last_day:02d}"
