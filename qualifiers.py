"""Qualifiers a question can carry (term, model year, tier word) and how well a block fits them.

A rate row for the wrong term, model-year range or tier is a different fact, not a
duplicate, so each qualifier either rules a block out or scales its score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from blocks import Block
from textnorm import content_tokens, stem

# "60 Months", "30-Year", "17 Month": number then a unit; stored in months.
_TERM = re.compile(r"(?<![\d.])\b(\d{1,3})\s*-?\s*(month|mo|year|yr)s?\b(?!\s*old)", re.I)
# Model-year ranges "2024-27" / "2023-2026", and bare years in a question (not "2026-09-25").
YEAR_SPAN = re.compile(r"\b(20\d{2})\s*-\s*(\d{4}|\d{2})\b(?!-\d)")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
# Words that name a different tier of the same product; the plain tier is the default.
TIER_WORDS = frozenset({"jumbo", "business", "roth", "traditional", "relief"})
EXCEPTIONAL_TIERS = frozenset({"relief"})  # one-off programs: never part of "the rates"
EXCEPTIONAL_TIER_FACTOR = 0.2

WRONG_TIER_FACTOR = 0.5  # block names a tier the question did not ask for
MISSING_TIER_FACTOR = 0.3  # question names a tier the block lacks
NO_YEAR_RANGE_FACTOR = 0.5  # question gives a model year but the block has no year range


# Caption words that name no product, so sharing them proves nothing.
GENERIC_CAPTION_WORDS = frozenset({"loan", "rate", "account", "special", "certificate"})


def caption_words(block: Block) -> frozenset[str]:
    return frozenset(t for t in content_tokens(block.caption) if t not in GENERIC_CAPTION_WORDS)


def fact_key(block: Block) -> tuple[str, str]:
    """What a table row is a value of: its product word and its tier label with model-year ranges
    removed ("2023-26 model year | 60 months" and "2021-23 model year | 60 months" are the same
    fact on a stale and on a current page)."""
    words = sorted(caption_words(block))
    return (words[0] if words else "", YEAR_SPAN.sub("", block.label.lower()))


def terms_in(text: str) -> frozenset[int]:
    """Loan/certificate terms in `text`, in months."""
    found = set()
    for number, unit in _TERM.findall(text):
        months = int(number) * (12 if unit.lower() in ("year", "yr") else 1)
        found.add(months)
    return frozenset(found)


_OLDER = re.compile(r"\b(20\d{2})\s+model\s+year\s*\(and older\)", re.I)


def year_spans_in(text: str) -> list[tuple[int, int]]:
    spans = [(0, int(year)) for year in _OLDER.findall(text)]  # "2016 Model Year (and older)" = up to 2016
    for lo, hi in YEAR_SPAN.findall(text):
        hi_year = int(hi) if len(hi) == 4 else int(lo[:2] + hi)
        spans.append((int(lo), hi_year))
    return spans


_MONTH_NAME = re.compile(r"\b(January|February|March|April|June|July|August|September|October|November|December)\b", re.I)
_MONEY_AMOUNT = re.compile(r"\$(\d+(?:\.\d+)?)")
# Amount tiers: "$10000 - $24999.99", "$25000+", "$300000 or less".
_AMOUNT_BAND = re.compile(r"\$(\d+(?:\.\d+)?)\s*-\s*\$(\d+(?:\.\d+)?)|\$(\d+(?:\.\d+)?)\s*(\+|or less)")
NO_AMOUNT_BAND_FACTOR = 0.2  # every amount in the question lies outside the block's amount tiers


def amount_bands_in(text: str) -> list[tuple[float, float]]:
    bands = []
    for lo, hi, single, kind in _AMOUNT_BAND.findall(text):
        if single:
            bands.append((float(single), float("inf")) if kind == "+" else (0.0, float(single)))
        else:
            bands.append((float(lo), float(hi)))
    return bands


def tier_words_in(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z]+", text.lower()) if w in TIER_WORDS)


@dataclass(frozen=True)
class QuestionInfo:
    terms: frozenset[int]
    years: tuple[int, ...]
    tiers: frozenset[str]
    consumed: frozenset[str]  # tokens the qualifiers already account for (digits, units)
    amounts: tuple[float, ...] = ()  # dollar amounts the question names (a balance, a loan size)

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
        if self.amounts:
            bands = amount_bands_in(block.text)
            if bands and not all(any(lo <= a <= hi for lo, hi in bands) for a in self.amounts):
                factor *= NO_AMOUNT_BAND_FACTOR
        block_tiers = tier_words_in(block.text)
        if ignore_tiers:  # a list wants every ordinary tier, but not a one-off program's rate
            return factor * EXCEPTIONAL_TIER_FACTOR ** len((block_tiers & EXCEPTIONAL_TIERS) - self.tiers)
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
    consumed.update(m.lower() for m in _MONTH_NAME.findall(text))  # a month is a date qualifier, not a topic word
    raw_amounts = _MONEY_AMOUNT.findall(text)
    consumed.update(raw_amounts)
    amounts = tuple(float(a) for a in raw_amounts)
    return QuestionInfo(terms_in(text), years, tier_words_in(text), frozenset(consumed), amounts)
