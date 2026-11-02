"""Small in-memory BM25 index over blocks (stdlib only)."""
from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict

from blocks import Block
from evidence import phone_in
from textnorm import content_tokens

K1 = 1.2
B = 0.75
CONTEXT_TF = 0.15  # a word of the page title/URL/headings counts this much of a word in the block text


class Index:
    def __init__(self, blocks: list[Block]) -> None:
        self.blocks = blocks
        self._postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._text_df: Counter = Counter()  # blocks whose own text has the word (the context does not count)
        self._lengths: list[int] = []
        for doc_id, block in enumerate(blocks):
            counts: Counter = Counter(content_tokens(block.text))
            self._text_df.update(counts.keys())
            self._lengths.append(sum(counts.values()))  # the context does not lengthen the block
            for token in set(content_tokens(block.context)):
                counts[token] += CONTEXT_TF
            for token, tf in counts.items():
                self._postings[token].append((doc_id, tf))
        self._avg_length = (sum(self._lengths) / len(blocks)) if blocks else 0.0
        self._showing: dict[str, frozenset[int]] = {}
        self.phone_counts: Counter = Counter(p for p in (phone_in(b.text) for b in blocks) if p)  # the main line is repeated

    def idf(self, token: str) -> float:
        df = self._text_df.get(token, 0)
        return math.log(1 + (len(self.blocks) - df + 0.5) / (df + 0.5))

    def showing(self, name: str, shows) -> frozenset[int]:
        """Doc ids whose text passes `shows` (cached by `name`): the blocks that carry some kind of evidence."""
        if name not in self._showing:
            self._showing[name] = frozenset(i for i, b in enumerate(self.blocks) if shows(b.text))
        return self._showing[name]

    def search(self, query_tokens: list[str], limit: int = 10, rows_only: bool = False,
               only: frozenset[int] | None = None) -> list[tuple[float, int]]:
        """Top (score, doc_id) pairs; ties broken by doc id so results are deterministic.

        `rows_only` ranks table rows against each other: long rows would otherwise lose to the
        short passages that mention the same words."""
        scores: dict[int, float] = defaultdict(float)
        for token in sorted(set(query_tokens)):
            weight = self.idf(token)
            for doc_id, tf in self._postings.get(token, ()):
                if rows_only and not self.blocks[doc_id].is_row:
                    continue
                if only is not None and doc_id not in only:
                    continue
                norm = K1 * (1 - B + B * self._lengths[doc_id] / self._avg_length)
                scores[doc_id] += weight * tf * (K1 + 1) / (tf + norm)
        top = heapq.nlargest(limit, ((score, -doc_id) for doc_id, score in scores.items()))
        return [(score, -neg_doc_id) for score, neg_doc_id in top]
