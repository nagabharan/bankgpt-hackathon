"""Is an offer still live? A block describing an ended offer must not be answered as a current deal."""
from __future__ import annotations

import re

from pagedate import dates_in

_ENDED_WORDS = re.compile(
    r"\b(?:offer|promotion|promo|special|deal|rate)\b[^.\n]{0,40}\b(?:has ended|have ended|ended|has expired|expired|is over|no longer available)\b"
    r"|\b(?:this|the) (?:offer|promotion|special) (?:has|is) (?:ended|over|expired)\b", re.I)
# "valid through 5/31/2026", "ends June 30, 2026", "expires 2026-07-01": a deadline keyword right before a date.
_DEADLINE = re.compile(r"\b(?:valid (?:through|until|thru)|through|until|thru|ends?|expires?|expired|ended)\b[:\s]+(?:on\s+)?", re.I)
_STARTS_WITH_DATE = re.compile(r"(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\b)")
_NAMED_EXPIRED = re.compile(r"expired|offer-ended|ended-offer", re.I)


def page_looks_expired(url: str, title: str) -> bool:
    return bool(_NAMED_EXPIRED.search(url) or _NAMED_EXPIRED.search(title))


def is_expired(text: str, snapshot_date: str) -> bool:
    """True when the text says the offer ended, or states an end date before `snapshot_date`."""
    if _ENDED_WORDS.search(text):
        return True
    for match in _DEADLINE.finditer(text):
        rest = text[match.end():match.end() + 28]
        if _STARTS_WITH_DATE.match(rest):  # the date must follow the keyword directly, not later in the sentence
            found = dates_in(rest)
            if found and found[0] < snapshot_date:
                return True
    return False
