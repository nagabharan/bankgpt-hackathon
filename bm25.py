"""Small in-memory BM25 index over blocks (stdlib only)."""
from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict

from blocks import Block
from textnorm import content_tokens

K1 = 1.2
B = 0.75


class Index:
    def __init__(self, blocks: list[Block]) -> None:
        self.blocks = blocks
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths: list[int] = []
        for doc_id, block in enumerate(blocks):
            counts = Counter(content_tokens(block.text))
            self._lengths.append(sum(counts.values()))
            for token, tf in counts.items():
                self._postings[token].append((doc_id, tf))
        self._avg_length = (sum(self._lengths) / len(blocks)) if blocks else 0.0

    def idf(self, token: str) -> float:
        df = len(self._postings.get(token, ()))
        return math.log(1 + (len(self.blocks) - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: list[str], limit: int = 10) -> list[tuple[float, int]]:
        """Top (score, doc_id) pairs; ties broken by doc id so results are deterministic."""
        scores: dict[int, float] = defaultdict(float)
        for token in sorted(set(query_tokens)):
            weight = self.idf(token)
            for doc_id, tf in self._postings.get(token, ()):
                norm = K1 * (1 - B + B * self._lengths[doc_id] / self._avg_length)
                scores[doc_id] += weight * tf * (K1 + 1) / (tf + norm)
        top = heapq.nlargest(limit, ((score, -doc_id) for doc_id, score in scores.items()))
        return [(score, -neg_doc_id) for score, neg_doc_id in top]
