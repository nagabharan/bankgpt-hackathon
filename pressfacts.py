"""Dated statements in press releases: counts ("22 branch office locations") and ordinals
("This was the fifth time Forbes recognized Dupaco").

Press releases repeat the same boilerplate for years, so the same count shows up with
different values on pages of different dates; the dateline says which is current.
"""
from __future__ import annotations

import re
from datetime import date
from typing import NamedTuple

_MONTHS = "January February March April May June July August September October November December".split()
# "June 18, 2024": the publication date printed near the top of a release.
_DATELINE = re.compile(rf"\b({'|'.join(_MONTHS)})\s+(\d{{1,2}}),\s+(\d{{4}})\b")
# "22 branch office locations" / "22 branches" (not "1,000 branches": the lookbehind skips digits and commas).
_BRANCH_COUNT = re.compile(r"(?<![\d,])(\d{1,3})\s+branch(?:es| office locations?)\b", re.I)
_ORDINALS = "first second third fourth fifth sixth seventh eighth ninth tenth".split()
# "fifth time Forbes recognized": the verb anchor keeps out "first-time homebuyer" and the like.
_ORDINAL_RECOGNITION = re.compile(rf"\b({'|'.join(_ORDINALS)}) time\s+(\w+)\s+(?:recognized|named|ranked)\b", re.I)


class DatedFact(NamedTuple):
    label: str  # what is being counted/ranked, e.g. "branch count", "forbes recognition"
    value: str  # the statement's value as printed, e.g. "22 branch office locations", "fifth"
    date: str  # ISO publication date of the page
    url: str


def dateline(text: str) -> str:
    """ISO date of the first 'Month DD, YYYY' in the page, or '' when it has none."""
    match = _DATELINE.search(text)
    if not match:
        return ""
    month, day, year = match.groups()
    return date(int(year), _MONTHS.index(month) + 1, int(day)).isoformat()


_HOW_MANY_BRANCHES = re.compile(r"\b(?:how many|number of|count of)\b[^?]*\bbranch", re.I)
_RECOGNITION_WORDS = re.compile(r"recogni|award|honou?r|named|ranked|ranking", re.I)


def answer_from_facts(question: str, facts: list[DatedFact]) -> DatedFact | None:
    """The latest-dated fact that the question asks about, or None.

    "How many branches...?" asks for the branch count; "what recognition ... from Forbes?"
    asks for the ordinal for that organisation. Old pages keep repeating superseded values,
    so the value on the newest page wins."""
    wanted = None
    if _HOW_MANY_BRANCHES.search(question):
        wanted = "branch count"
    elif _RECOGNITION_WORDS.search(question):
        orgs = {f.label.removesuffix(" recognition") for f in facts if f.label.endswith(" recognition")}
        asked = [org for org in sorted(orgs) if re.search(rf"\b{re.escape(org)}\b", question, re.I)]
        wanted = f"{asked[0]} recognition" if asked else None
    candidates = [f for f in facts if f.label == wanted]
    return max(candidates, key=lambda f: (f.date, f.value)) if candidates else None


def sentence_for(fact: DatedFact) -> str:
    """One plain sentence stating only the winning value."""
    if fact.label == "branch count":
        return f"Dupaco has {fact.value}."
    org = fact.label.removesuffix(" recognition").title()
    return f"{org} has recognized Dupaco for the {fact.value} time."


def dated_facts(url: str, markdown: str) -> list[DatedFact]:
    published = dateline(markdown)
    if not published:
        return []
    facts = [DatedFact("branch count", f"{n} branch office locations", published, url)
             for n in sorted(set(_BRANCH_COUNT.findall(markdown)))]
    facts += [DatedFact(f"{org.lower()} recognition", ordinal.lower(), published, url)
              for ordinal, org in sorted(set(_ORDINAL_RECOGNITION.findall(markdown)))]
    return facts
