"""Small in-memory BM25 index over blocks (stdlib only)."""
from __future__ import annotations

import heapq
import math
from array import array
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
        # token -> (doc ids, term frequencies) as parallel typed arrays: a posting costs 12 bytes,
        # not the ~90 of a (doc_id, tf) tuple, and postings are most of the memory after ingest.
        postings: dict[str, tuple[array, array]] = defaultdict(lambda: (array("i"), array("d")))
        self._text_df: Counter = Counter()  # blocks whose own text has the word (the context does not count)
        self._lengths: list[int] = []
        context_tokens: dict[str, set[str]] = {}  # every block of a page section shares its context
        for doc_id, block in enumerate(blocks):
            counts: Counter = Counter(content_tokens(block.text))
            self._text_df.update(counts.keys())
            self._lengths.append(sum(counts.values()))  # the context does not lengthen the block
            if block.context not in context_tokens:
                context_tokens[block.context] = set(content_tokens(block.context))
            for token in context_tokens[block.context]:
                counts[token] += CONTEXT_TF
            for token, tf in counts.items():
                doc_ids, tfs = postings[token]
                doc_ids.append(doc_id)
                tfs.append(tf)
        self._postings = dict(postings)
        self._avg_length = (sum(self._lengths) / len(blocks)) if blocks else 0.0
        self._cache: dict[str, object] = {}
        self.page_prior: dict[str, float] = {}  # url -> prior, set by the view that owns this index
        self.acronyms: dict[str, tuple[str, ...]] = {}  # acronym -> words it stands for, set by the view
        self.phone_counts: Counter = Counter(p for p in (phone_in(b.text) for b in blocks) if p)  # the main line is repeated

    def document_frequency(self, token: str) -> int:
        return self._text_df.get(token, 0)

    def df(self, token: str) -> int:
        return self._text_df.get(token, 0)

    def idf(self, token: str) -> float:
        df = self._text_df.get(token, 0)
        return math.log(1 + (len(self.blocks) - df + 0.5) / (df + 0.5))

    def cached(self, name: str, compute):
        """compute() on first use, then the stored result: derived data over all blocks, built per index
        once (callers must not mutate it)."""
        if name not in self._cache:
            self._cache[name] = compute()
        return self._cache[name]

    def showing(self, name: str, shows) -> frozenset[int]:
        """Doc ids whose text passes `shows` (cached by `name`): the blocks that carry some kind of evidence."""
        return self.cached("showing:" + name, lambda: frozenset(i for i, b in enumerate(self.blocks) if shows(b.text)))

    def search(self, query_tokens: list[str], limit: int = 10, rows_only: bool = False,
               only: frozenset[int] | None = None) -> list[tuple[float, int]]:
        """Top (score, doc_id) pairs; ties broken by doc id so results are deterministic.

        `rows_only` ranks table rows against each other: long rows would otherwise lose to the
        short passages that mention the same words."""
        scores: dict[int, float] = defaultdict(float)
        for token in sorted(set(query_tokens)):
            weight = self.idf(token)
            doc_ids, tfs = self._postings.get(token, ((), ()))
            for doc_id, tf in zip(doc_ids, tfs):
                if rows_only and not self.blocks[doc_id].is_row:
                    continue
                if only is not None and doc_id not in only:
                    continue
                norm = K1 * (1 - B + B * self._lengths[doc_id] / self._avg_length)
                scores[doc_id] += weight * tf * (K1 + 1) / (tf + norm)
        top = heapq.nlargest(limit, ((score, -doc_id) for doc_id, score in scores.items()))
        return [(score, -neg_doc_id) for score, neg_doc_id in top]
