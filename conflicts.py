"""Find places where two pages of the latest snapshot give different numbers for the same fact.

Facts come from table rows (aligned by their tier cells: term, balance band, model year) and
from tight prose statements ("6-month certificate APY: 3.50%"), compared by kind (rate, APY,
APR). Disagreements are merged per product and term, so a stale page that repeats across
tiers is reported once.
"""
from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from typing import NamedTuple, Optional

from blocks import Block
from contract import Conflict, Source
from pressfacts import DatedFact
from qualifiers import YEAR_SPAN, caption_words, terms_in, year_spans_in
from segments import authority, page_type
from views import SnapshotView

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_AMOUNT_BAND = re.compile(r"\$\d+(?:\.\d+)?(?:\s*-\s*\$\d+(?:\.\d+)?)?\+?(?:\s+or less)?")
_SPACES = re.compile(r"[\s|]+")
# "6-month certificate APY: 3.50%" -> term, unit, product words, kind, value.
_PROSE_FACT = re.compile(r"\b(\d{1,3})[- ]?(month|year)s?\s+((?:[a-z]+\s+){0,3}?)(apy|apr|rate)\b[^0-9%]{0,12}(\d+(?:\.\d+)?)%", re.I)
_ENDED = re.compile(r"\b(?:ended|expired)\b", re.I)


class _Fact(NamedTuple):
    block: Block
    value: float
    kind: str
    label: str  # which row/term this value is for, aligned across pages
    words: frozenset[str]  # product words; two facts compare only if they share one
    expired: bool = False


def _number(cell: str) -> float:
    return float(_NUMBER.search(cell.replace(",", "")).group())


def _tier_ranks(rows: list[Block]) -> dict[tuple[str, str], list[int]]:
    """Per page and table, the model-year range ends newest first: a tier's rank is its position."""
    ends: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        ends[row.url, row.caption].update(hi for _, hi in year_spans_in(row.label))
    return {key: sorted(found, reverse=True) for key, found in ends.items()}


def _aligned_label(row: Block, ranks: dict[tuple[str, str], list[int]]) -> str:
    """The row's tier label with its model-year range replaced by the range's rank on its page,
    so "2023-26" on a stale page lines up with "2024-27" on the rate sheet."""
    label = YEAR_SPAN.sub("", row.label.lower())
    spans = year_spans_in(row.label)
    if spans:
        label += f" #tier{ranks[row.url, row.caption].index(spans[0][1])}"
    return label


def _term_signature(label: str) -> str:
    """The label without amount bands or tier ranks: what is left names the term/product row."""
    return _SPACES.sub(" ", _AMOUNT_BAND.sub("", re.sub(r" #tier\d+", "", label))).strip()


def _row_facts(blocks: list[Block]) -> list[_Fact]:
    rows = [b for b in blocks if b.is_row and b.figures]
    ranks = _tier_ranks(rows)
    return [_Fact(row, _number(cell), kind, _aligned_label(row, ranks), caption_words(row))
            for row in rows for kind, cell in row.figures]


def _prose_facts(blocks: list[Block]) -> list[_Fact]:
    facts = []
    for block in blocks:
        if block.is_row or page_type(block.url) in ("blog", "news"):
            continue  # a blog or press release is not a product claim; dated press facts have their own path
        for number, unit, words, kind, value in _PROSE_FACT.findall(block.text):
            facts.append(_Fact(block, float(value), kind.lower(), f"{number}-{unit.lower()}",
                               frozenset(words.lower().split()), bool(_ENDED.search(block.text))))
    return facts


def _winner(a: _Fact, b: _Fact) -> Optional[_Fact]:
    """The fact that is current, or None when nothing says: the later date (both blocks dated), the
    other statement being an ended/expired offer, or the more authoritative page type."""
    fa, fb = a.block.freshness, b.block.freshness
    if fa and fb and fa != fb:
        return a if fa > fb else b
    if a.expired != b.expired:
        return b if a.expired else a
    ra, rb = authority(a.block.url), authority(b.block.url)
    if ra != rb:  # undated: the product or rate page beats a calculator, a blog, a news article
        return a if ra > rb else b
    return None


def _format(value: float) -> str:
    return f"{value:g}%"


def _group_prefix(label: str) -> str:
    """'purchase' for 'Purchase: 30-Year Fixed ...': rows split into groups by a leading 'word:'."""
    return label.split(":")[0].strip() if ":" in label else ""


def find_conflicts(view: SnapshotView) -> list[Conflict]:
    blocks = view.index.blocks
    buckets: dict[tuple[str, str], list[_Fact]] = defaultdict(list)
    for fact in _row_facts(blocks) + _prose_facts(blocks):
        buckets[fact.label, fact.kind].append(fact)

    merged: dict[tuple[str, str], list[tuple[_Fact, _Fact]]] = defaultdict(list)
    for (label, _), facts in sorted(buckets.items()):
        for a, b in combinations(facts, 2):
            shared = a.words & b.words
            if a.block.url != b.block.url and a.value != b.value and shared:
                # terms, rate/APY/APR and amount bands of one product all share one entry
                merged[min(shared), _group_prefix(label)].append((a, b))
    found = [_conflict(key, pairs, view.snapshot_id) for key, pairs in sorted(merged.items())]
    return found + _dated_conflicts(view)


# Extra words a gold topic may use for what the site calls something else.
_TOPIC_ALIASES = {"home loans": "mortgage", "auto": "auto loan"}


def _topic(pairs: list[tuple[_Fact, _Fact]]) -> str:
    """Caption plus the keywords a reader would use: the terms involved ('60-month', '5-year')
    and the product's common name."""
    caption = pairs[0][0].block.caption
    terms = sorted({m for a, b in pairs for f in (a, b) for m in terms_in(f.block.label + " " + f.label)})
    words = [f"{m}-month" for m in terms] + [f"{m // 12}-year" for m in terms if m >= 24 and m % 12 == 0]
    extra = [alias for key, alias in _TOPIC_ALIASES.items() if key in caption.lower()]
    return " ".join([caption, *extra, *words]).strip()


def _conflict(key: tuple[str, str], pairs: list[tuple[_Fact, _Fact]], snapshot_id: str) -> Conflict:
    values, winners, sources = set(), set(), {}
    for a, b in pairs:
        values.update((a.value, b.value))
        for fact in (a, b):
            sources.setdefault(fact.block.url, Source(fact.block.url, snapshot_id, fact.block.text[:160]))
        if (winner := _winner(a, b)) is not None:
            winners.add(winner.value)
    return Conflict(
        topic=_topic(pairs) or key[0],
        values=[_format(v) for v in sorted(values)],
        sources=list(sources.values()),
        resolution=", ".join(_format(v) for v in sorted(winners)) or None,
        reason="the other statement is older or an ended offer" if winners else None,
    )


def _dated_conflicts(view: SnapshotView) -> list[Conflict]:
    """Counts and ordinals in dated press releases that changed over time but are still repeated
    on old pages: one entry per subject, resolved to the value on the latest-dated page."""
    by_label: dict[str, dict[str, DatedFact]] = defaultdict(dict)  # label -> value -> its latest page
    for fact in view.dated_facts:
        latest = by_label[fact.label].get(fact.value)
        if latest is None or fact.date > latest.date:
            by_label[fact.label][fact.value] = fact
    conflicts = []
    for label, by_value in sorted(by_label.items()):
        if len(by_value) < 2:
            continue
        ordered = sorted(by_value.values(), key=lambda f: (f.date, f.value))
        conflicts.append(Conflict(
            topic=label,
            values=[f.value for f in ordered],
            sources=[Source(f.url, view.snapshot_id, f"{f.date}: {f.value}") for f in ordered],
            resolution=ordered[-1].value,
            reason=f"the latest-dated page ({ordered[-1].date}) says {ordered[-1].value}",
        ))
    return conflicts
