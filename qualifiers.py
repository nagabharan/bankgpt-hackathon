"""Qualifiers a question can carry (term, model year, tier word) and how well a block fits them.

A rate row for the wrong term, model-year range or tier is a different fact, not a
duplicate, so each qualifier either rules a block out or scales its score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from blocks import Block
from textnorm import stem

# "60 Months", "30-Year", "17 Month": number then a unit; stored in months.
_TERM = re.compile(r"(?<![\d.])\b(\d{1,3})\s*-?\s*(month|mo|year|yr)s?\b(?!\s*old)", re.I)
# Model-year ranges "2024-27" / "2023-2026", and bare years in a question (not "2026-09-25").
YEAR_SPAN = re.compile(r"\b(20\d{2})\s*-\s*(\d{4}|\d{2})\b(?!-\d)")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
# Words that name a different tier of the same product; the plain tier is the default.
TIER_WORDS = frozenset({"jumbo", "business", "roth", "traditional", "relief"})

WRONG_TIER_FACTOR = 0.5  # block names a tier the question did not ask for
MISSING_TIER_FACTOR = 0.3  # question names a tier the block lacks
NO_YEAR_RANGE_FACTOR = 0.5  # question gives a model year but the block has no year range


def terms_in(text: str) -> frozenset[int]:
    """Loan/certificate terms in `text`, in months."""
    found = set()
    for number, unit in _TERM.findall(text):
        months = int(number) * (12 if unit.lower() in ("year", "yr") else 1)
        found.add(months)
    return frozenset(found)


def year_spans_in(text: str) -> list[tuple[int, int]]:
    spans = []
    for lo, hi in YEAR_SPAN.findall(text):
        hi_year = int(hi) if len(hi) == 4 else int(lo[:2] + hi)
        spans.append((int(lo), hi_year))
    return spans


def tier_words_in(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z]+", text.lower()) if w in TIER_WORDS)


@dataclass(frozen=True)
class QuestionInfo:
    terms: frozenset[int]
    years: tuple[int, ...]
    tiers: frozenset[str]
    consumed: frozenset[str]  # tokens the qualifiers already account for (digits, units)

    def fit(self, block: Block, ignore_tiers: bool = False) -> float:
        """0 rules the block out; otherwise a factor to scale its score by.

        `ignore_tiers` is for list questions, which want every tier of the product."""
        if self.terms and not self.terms & terms_in(block.text):
            return 0.0
        factor = 1.0
        if self.years:
            spans = year_spans_in(block.text)
            if not spans:
                factor *= NO_YEAR_RANGE_FACTOR
            elif not any(lo <= y <= hi for y in self.years for lo, hi in spans):
                return 0.0
        if ignore_tiers:
            return factor
        block_tiers = tier_words_in(block.text)
        factor *= WRONG_TIER_FACTOR ** len(block_tiers - self.tiers)
        factor *= MISSING_TIER_FACTOR ** len(self.tiers - block_tiers)
        return factor


def parse_question(text: str) -> QuestionInfo:
    """`text` is the cleaned question."""
    consumed = set()
    for number, unit in _TERM.findall(text):
        consumed.update((number, stem(unit.lower())))
    spans = {m.group(0) for m in YEAR_SPAN.finditer(text)}
    years = tuple(sorted({int(y) for y in _YEAR.findall(text) if not any(y in s for s in spans)}))
    consumed.update(str(y) for y in years)
    return QuestionInfo(terms_in(text), years, tier_words_in(text), frozenset(consumed))
