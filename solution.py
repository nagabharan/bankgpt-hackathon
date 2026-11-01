"""Entry point for the grader: build() returns the knowledge system.

`contract` is importable because the scorer puts harness/ on sys.path before loading
this folder; always return the harness's own Answer type (the repeat-consistency check
uses isinstance on it).

Keep @dataclass definitions out of this file: the scorer loads it without registering it
in sys.modules, which breaks dataclasses under `from __future__ import annotations`.
"""
from __future__ import annotations

import re
from typing import Optional

from bm25 import Index
from blocks import Block
from contract import Answer, Conflict, Snapshot, Source
from textnorm import MONEY_MARK, clean, content_tokens
from views import SnapshotView, build_view

# Refusal text must contain no digits: a refusal fails if it adds any number the
# question did not contain.
NOT_LISTED = "That isn't listed on our website; please contact us."

# Share of the question's idf mass the best block must cover, else we decline.
MIN_COVERAGE = 0.6
CANDIDATES = 10
_MONEY_OR_PERCENT = re.compile(f"%|{MONEY_MARK.pattern}")
_CLOCK_TIME = re.compile(r"\d\s*(?::\d\d)?\s*[ap]\.?m", re.I)
# A question word asks for a value of this shape; a block without one cannot answer it.
VALUE_SHAPES = {
    **dict.fromkeys(("rate", "apr", "apy", "interest", "fee", "cost", "charge", "price"), _MONEY_OR_PERCENT),
    "hour": _CLOCK_TIME,
}


def coverage(index: Index, query: list[str], block: Block) -> float:
    """Fraction of the question's idf-weighted tokens that appear in the block."""
    present = set(content_tokens(block.text))
    wanted = {t: index.idf(t) for t in sorted(set(query))}  # sorted: float sums stay hash-seed independent
    total = sum(wanted.values())
    return sum(w for t, w in wanted.items() if t in present) / total if total else 0.0


def is_answer(index: Index, query: list[str], block: Block) -> bool:
    """The block must cover the question's entities and hold the kind of value it asks for."""
    if coverage(index, query, block) < MIN_COVERAGE:
        return False
    return all(shape.search(block.text) for word, shape in VALUE_SHAPES.items() if word in query)


class KnowledgeBase:
    def __init__(self) -> None:
        self.views: list[SnapshotView] = []  # oldest first

    def ingest(self, snapshot: Snapshot) -> None:
        self.views.append(build_view(snapshot))

    def _view_for(self, as_of: Optional[str]) -> Optional[SnapshotView]:
        """Newest snapshot taken on or before `as_of` (newest overall when None)."""
        eligible = [v for v in self.views if as_of is None or v.taken_on <= as_of]
        return eligible[-1] if eligible else None

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        view = self._view_for(as_of)
        query = content_tokens(clean(question))
        if view is None or not query:
            return Answer(text=NOT_LISTED, confident=False)
        for _, doc_id in view.index.search(query, CANDIDATES):
            block = view.index.blocks[doc_id]
            if is_answer(view.index, query, block):
                source = Source(url=block.url, snapshot_id=view.snapshot_id, quote=block.text[:200])
                return Answer(text=block.text, sources=[source])
        return Answer(text=NOT_LISTED, confident=False)

    def conflicts(self) -> list[Conflict]:
        return []


def build() -> KnowledgeBase:
    return KnowledgeBase()
