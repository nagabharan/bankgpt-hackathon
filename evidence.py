"""Questions that ask for a particular kind of fact, and the evidence a block must show to answer them.

"How can I update my contact information?" wants a phone number, "who is eligible to join?" wants
where members live, "what is the Home program?" wants its dollar amount. A block without that
evidence cannot answer, however well its words match.
"""
from __future__ import annotations

import re
from typing import Callable, NamedTuple, Optional

_PHONE = re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b|\(\d{3}\)\s*\d{3}[-.\s]\d{4}")
_DOLLAR_AMOUNT = re.compile(r"\$\d")
_STATES = frozenset(
    "Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii Idaho "
    "Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota "
    "Mississippi Missouri Montana Nebraska Nevada Ohio Oklahoma Oregon Pennsylvania Tennessee Texas Utah "
    "Vermont Virginia Washington Wisconsin Wyoming".split()
) | {"New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
     "Rhode Island", "South Carolina", "South Dakota", "West Virginia"}
_STATE_WORD = re.compile(r"\b(?:" + "|".join(sorted(_STATES, key=len, reverse=True)) + r")\b")


class Kind(NamedTuple):
    name: str
    question: re.Pattern
    shows: Callable[[str], bool]  # does the block's text carry the evidence?
    filler: frozenset[str]  # words that name the action, not the topic: not required to appear
    min_coverage: float
    skip_articles: bool  # blogs and press releases are not where this fact is stated


def _shows_places(text: str) -> bool:
    return len(set(_STATE_WORD.findall(text))) >= 2


KINDS = (
    Kind("phone", re.compile(r"\b(?:contact|phone|call|reach|hotline)\b", re.I), lambda t: bool(_PHONE.search(t)),
         frozenset("update change information info detail contact phone call reach number speak talk".split()),
         0.6, True),
    Kind("places", re.compile(r"\b(?:eligib\w*|who can (?:join|become)|where (?:do|are|can) you (?:serve|operate))\b", re.I),
         _shows_places, frozenset("eligible eligibility become who can".split()), 0.6, True),
    Kind("amount", re.compile(r"\bwhat(?:'s| is| are)\b.*\b(?:program|promotion|grant|assistance|bonus)\b", re.I),
         lambda t: bool(_DOLLAR_AMOUNT.search(t)), frozenset(), 0.6, False),
)


def phone_in(text: str) -> str:
    """The first phone number in `text` (digits only), or ''."""
    match = _PHONE.search(text)
    return re.sub(r"\D", "", match.group(0)) if match else ""


def kind_for(question: str) -> Optional[Kind]:
    return next((k for k in KINDS if k.question.search(question)), None)
