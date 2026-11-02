"""Opening hours that differ between branches: collect the distinct schedules found on the site.

Hours live in branch announcements, one schedule per page. "Do all branches keep the same lobby
hours?" is answered by listing the different schedules, not by picking one branch's.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from bm25 import Index
from blocks import Block
from textnorm import clean, content_tokens

CLOCK = re.compile(r"\d\s*(?::\d\d)?\s*[ap]\.?m|\bnoon\b", re.I)
_COMPARES = re.compile(r"\b(?:same|differ\w*|vary|varies|each|every|all)\b", re.I)
_NOT_TOPIC = frozenset("hour same differ different vary each every all dupaco branch keep open close time".split())
DEFAULT_TOPIC = "lobby"  # with no kind named, the branch lobby hours
MAX_SCHEDULES = 3
MAX_LINES = 8


# Wordings that ask for opening hours without the word "hours".
OPENING = re.compile(r"\bwhat time\b|\bwhen (?:is|are)\b[^?]*\bopen\b|\bopen and close\b|\bopening\b|\bclosing time\b", re.I)
_CAPITALISED = re.compile(r"\b[A-Z][\w'-]*")


def _names_a_place(question: str) -> bool:
    """A capitalised word after the first that is not Dupaco ("Galena", "Hy-Vee"): hours of one place."""
    return any(m.start() > 0 and m.group(0).lower() != "dupaco" for m in _CAPITALISED.finditer(question))


def asks_hours(question: str) -> bool:
    return "hour" in content_tokens(clean(question)) or bool(OPENING.search(question))


def asks_schedules(question: str) -> bool:
    """Opening hours of no particular place, or a comparison between places: show the schedules found."""
    return asks_hours(question) and (bool(_COMPARES.search(question)) or not _names_a_place(question))


def distinct_schedules(index: Index, question: str) -> list[Block]:
    """The clock-time lines of up to three distinct page schedules, most common first."""
    topic = [t for t in content_tokens(clean(question)) if t not in _NOT_TOPIC] or [DEFAULT_TOPIC]
    by_page: dict[str, list[Block]] = defaultdict(list)
    for doc_id in sorted(index.showing("clock", lambda text: bool(CLOCK.search(text)))):
        block = index.blocks[doc_id]
        by_page[block.url].append(block)
    schedules: Counter = Counter()
    lines_of: dict[tuple[str, ...], list[Block]] = {}
    for url in sorted(by_page):
        blocks = by_page[url]
        if topic and not any(t in set(content_tokens(b.text)) for b in blocks for t in topic):
            continue  # this page's hours are not the asked kind (lobby, drive-up...)
        key = tuple(b.text for b in blocks[:MAX_LINES])
        schedules[key] += 1
        lines_of.setdefault(key, blocks[:MAX_LINES])
    ranked = sorted(schedules, key=lambda key: (-schedules[key], key))[:MAX_SCHEDULES]
    return [block for key in ranked for block in lines_of[key]]
