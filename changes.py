"""Questions about what changed between snapshots ("did the rate change since May?").

They are answered from two snapshots: the newest and an earlier reference one, and both values
are printed with their dates, because the question is about the difference.
"""
from __future__ import annotations

import calendar
import re
from typing import Optional

from contract import Source
from matching import MONEY_WORDS, select_blocks
from textnorm import clean, content_tokens
from views import SnapshotView

_CHANGE_WORDS = re.compile(
    r"\b(?:chang(?:e|ed|es|ing)|since|versus|vs\.?|compared? (?:to|with)|increase[ds]?|decrease[ds]?|"
    r"went (?:up|down)|rise[sn]?|rose|dropp?(?:ed)?|fell|used to|previously|differen(?:t|ce))\b", re.I)
_MONTHS = list(calendar.month_name)[1:]
# "since May", "from March 2026", "in May", "vs May": the month the question compares with.
_REFERENCE = re.compile(rf"\b(?:since|from|in|as of|during|versus|vs\.?|compared? (?:to|with))\s+"
                        rf"(?P<month>{'|'.join(_MONTHS)})(?:\s+(?P<year>\d{{4}}))?\b")
_MONTH_WORD = re.compile(rf"\b(?:{'|'.join(_MONTHS)})(?:\s+\d{{4}})?\b")


def is_change_question(question: str) -> bool:
    """A change word together with a rate/fee word ("how do I change my PIN" is not one)."""
    return bool(_CHANGE_WORDS.search(question)) and bool(MONEY_WORDS.intersection(content_tokens(clean(question))))


def _reference_view(views: list[SnapshotView], question: str) -> Optional[SnapshotView]:
    """The first snapshot taken in or after the month the question names, else the oldest one;
    never the newest itself."""
    older = views[:-1]
    match = _REFERENCE.search(question)
    if match and older:
        year = int(match.group("year") or views[-1].taken_on[:4])
        start = f"{year}-{_MONTHS.index(match.group('month')) + 1:02d}-01"
        later = [v for v in older if v.taken_on >= start]
        return later[0] if later else older[-1]
    return older[0] if older else None


def _topic(question: str) -> str:
    """The question without the change wording and the month it compares with."""
    question = _REFERENCE.sub(" ", question)
    question = _MONTH_WORD.sub(" ", question)
    return _CHANGE_WORDS.sub(" ", question)


def change_answer(views: list[SnapshotView], question: str) -> Optional[tuple[str, Source]]:
    """('Now: ... / As of <date>: ...', source) or None when either snapshot has no answer."""
    reference = _reference_view(views, question)
    if reference is None:
        return None
    topic = _topic(question)
    now = select_blocks(views[-1].index, topic)
    then = select_blocks(reference.index, topic)
    if not now or not then:
        return None
    source = Source(now[0].url, views[-1].snapshot_id, now[0].text[:200])
    if now[0].text == then[0].text:
        return f"No change since {reference.taken_on}: {now[0].text}", source
    return f"Now: {now[0].text}\nAs of {reference.taken_on}: {then[0].text}", source
