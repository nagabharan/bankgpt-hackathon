"""Dates written inside a question ("as of May 2026", "in March") as a snapshot date."""
from __future__ import annotations

import calendar
import re
from datetime import date
from typing import Optional

from textnorm import MONTHS

# A month needs a year or a preposition: a bare capitalised "May" is usually the verb ("May I apply?").
# "since"/"from" are not here: they ask what changed, not what was true then.
_DATE_PHRASE = re.compile(
    rf"\b(?P<prep>(?:as of|in|during|on|around|by|at|for)\s+)?(?P<month>{'|'.join(MONTHS)})(?:\s+(?P<day>\d{{1,2}})(?:(?:st|nd|rd|th)\b|(?=,?\s+\d{{4}}\b)|(?!\s*-?\s*(?:month|mo\b|year|yr|day|week|%|percent|k\b|\d))),?)?(?:\s+(?P<year>\d{{4}}))?\b")
# "2026-05-12" and "5/12/2026" name one day.
_NUMERIC_DATE = re.compile(r"\b(?:(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})|(?P<m2>\d{1,2})/(?P<d2>\d{1,2})/(?P<y2>\d{4}))\b")
# "as of 6/9": a month/day with a date preposition and no year takes the default year.
_SHORT_DATE = re.compile(r"\b(?:as of|on|around|by|since|for)\s+(?P<m>\d{1,2})/(?P<d>\d{1,2})\b(?![/\d])", re.I)


# The last day of each season: "in spring" asks for what the site said by the end of May.
_SEASON_END = {"spring": (5, 31), "summer": (8, 31), "fall": (11, 30), "autumn": (11, 30), "winter": (2, 28)}
_SEASON_PHRASE = re.compile(r"\b(?:as of|in|during)\s+(?:(?:the|last|this|past)\s+)?(?P<season>spring|summer|fall|autumn|winter)\b", re.I)


def long_date(iso: str) -> str:
    """'2026-06-09' -> 'June 9, 2026'."""
    day = date.fromisoformat(iso)
    return f"{day:%B} {day.day}, {day.year}"


def split_date(question: str, default_year: int) -> tuple[str, Optional[str]]:
    """(question without the date phrase, last day of that month as 'YYYY-MM-DD' or None).

    A month without a year takes `default_year` (the year of the newest snapshot). A named day
    ("on May 12, 2026", "2026-05-12") is that day itself."""
    numeric = _NUMERIC_DATE.search(question)
    if numeric:
        year, month, day = (int(numeric.group(k) or numeric.group(k2)) for k, k2 in (("y", "y2"), ("m", "m2"), ("d", "d2")))
        if 1 <= month <= 12 and 1 <= day <= calendar.monthrange(year, month)[1]:
            stripped = (question[:numeric.start()] + question[numeric.end():]).replace("  ", " ").strip()
            return stripped, f"{year}-{month:02d}-{day:02d}"
    short = _SHORT_DATE.search(question)
    if short and 1 <= int(short.group("m")) <= 12 and 1 <= int(short.group("d")) <= calendar.monthrange(default_year, int(short.group("m")))[1]:
        stripped = (question[:short.start()] + question[short.end():]).replace("  ", " ").strip()
        return stripped, f"{default_year}-{int(short.group('m')):02d}-{int(short.group('d')):02d}"
    match = next((m for m in _DATE_PHRASE.finditer(question) if m.group("prep") or m.group("year")), None)
    if match is None:
        season = _SEASON_PHRASE.search(question)
        if season is None:
            return question, None
        month, day = _SEASON_END[season.group("season").lower()]
        stripped = (question[:season.start()] + question[season.end():]).replace("  ", " ").strip()
        return stripped, f"{default_year}-{month:02d}-{day:02d}"
    year = int(match.group("year") or default_year)
    month = MONTHS.index(match.group("month")) + 1
    last_day = calendar.monthrange(year, month)[1]
    if match.group("day") and 1 <= int(match.group("day")) <= last_day:
        last_day = int(match.group("day"))
    stripped = (question[:match.start()] + question[match.end():]).replace("  ", " ").strip()
    return stripped, f"{year}-{month:02d}-{last_day:02d}"
