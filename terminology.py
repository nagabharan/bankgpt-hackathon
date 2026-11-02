"""A model-year bracket the site has renamed: a question that still says "2023-26" means the bracket
that is called "2024-27" in the snapshot being read.

A table's brackets are ranked newest first on every snapshot; the same rank in the same table is the
same bracket under its name of the time. A bracket named in the question that the snapshot's rates page
no longer uses is replaced by its current name (the snapshot that last used the old name tells the rank).
"""
from __future__ import annotations

import re
from conflicts import _tier_ranks
from qualifiers import YEAR_SPAN, caption_words, year_spans_in
from segments import page_type
from textnorm import clean, content_tokens
from views import SnapshotView

_CANONICAL = frozenset({"rates"})  # the rate sheet names brackets first; stale copies on product pages keep old names


def _brackets(view: SnapshotView) -> dict[tuple[frozenset[str], int], tuple[int, int]]:
    """(table words, rank) -> model-year span for each table of the rates page."""

    def build() -> dict[tuple[frozenset[str], int], tuple[int, int]]:
        rows = [b for b in view.index.blocks if b.is_row and b.figures and page_type(b.url) in _CANONICAL and year_spans_in(b.label)]
        ranks = _tier_ranks(rows)
        return {(caption_words(b), ranks[b.url, b.caption].index(year_spans_in(b.label)[0][1])): year_spans_in(b.label)[0] for b in rows}

    return view.index.cached("brackets", build)


def current_names(question: str, earlier: list[SnapshotView], view: SnapshotView) -> str:
    """`question` with each model-year bracket the snapshot no longer uses renamed to its current name.
    Only the table the question is about counts (an ATV table keeps a "2023-26" the auto table dropped)."""
    if not YEAR_SPAN.search(question):
        return question
    now = _brackets(view)
    topic = set(content_tokens(clean(question)))

    def rename(match: re.Match) -> str:
        span = year_spans_in(match.group(0))[0]
        for old_view in reversed(earlier):
            for (words, rank), old_span in _brackets(old_view).items():
                if old_span != span or not words & topic:
                    continue
                for (new_words, new_rank), new_span in now.items():
                    if new_rank == rank and words & new_words:
                        return match.group(0) if new_span == span else f"{new_span[0]}-{str(new_span[1])[2:]}"
        return match.group(0)

    return YEAR_SPAN.sub(rename, question)
