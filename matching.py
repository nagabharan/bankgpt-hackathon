"""Pick the block that answers a question: retrieve, check, apply qualifiers, prefer fresh."""
from __future__ import annotations

import re
from typing import Optional

from bm25 import Index
from blocks import Block
from qualifiers import QuestionInfo, parse_question
from textnorm import MONEY_MARK, clean, content_tokens

CANDIDATES = 30
# Share of the question's idf mass the block must cover, else it does not answer the question.
MIN_COVERAGE = 0.6
ROW_BOOST = 1.5  # a table row beats prose when the question asks for a rate or fee
NEAR_TIE = 0.9  # candidates within this share of the best score compete on freshness

# Other words the site uses for the same thing ("Home Loans - Purchase" rows never say mortgage).
ALIASES = {"mortgage": ("home",), "cd": ("certificate",)}
# Words that say which value to pick, not what the topic is; no block contains "highest".
SUPERLATIVES = frozenset("best highest higher largest lowest lower maximum minimum most least top".split())
_MONEY_OR_PERCENT = re.compile(f"%|{MONEY_MARK.pattern}")
_CLOCK_TIME = re.compile(r"\d\s*(?::\d\d)?\s*[ap]\.?m", re.I)
# A question word asks for a value of this shape; a block without one cannot answer it, and
# one with it needn't repeat the word itself (hours are written "8:30 a.m.", not "hours").
VALUE_SHAPES = {
    **dict.fromkeys(("rate", "apr", "apy", "interest", "fee", "cost", "charge", "price"), _MONEY_OR_PERCENT),
    "hour": _CLOCK_TIME,
}


def coverage(index: Index, required: list[str], block: Block) -> float:
    """Fraction of the required tokens' idf mass that appears in the block (aliases count)."""
    present = set(content_tokens(block.text))
    wanted = {t: index.idf(t) for t in sorted(set(required))}  # sorted: float sums stay hash-seed independent
    total = sum(wanted.values())
    covered = sum(w for t, w in wanted.items() if t in present or present.intersection(ALIASES.get(t, ())))
    return covered / total if total else 0.0


def with_aliases(query: list[str]) -> list[str]:
    return query + [alias for t in query for alias in ALIASES.get(t, ())]


def has_asked_value(query: list[str], block: Block) -> bool:
    return all(shape.search(block.text) for word, shape in VALUE_SHAPES.items() if word in query)


def best_block(index: Index, question: str) -> Optional[Block]:
    text = clean(question)
    info: QuestionInfo = parse_question(text)
    query = content_tokens(text)
    required = [t for t in query if t not in info.consumed and t not in VALUE_SHAPES and t not in SUPERLATIVES]
    asks_value = any(t in VALUE_SHAPES for t in query)

    scored = []
    for score, doc_id in index.search(with_aliases(query), CANDIDATES):
        block = index.blocks[doc_id]
        factor = info.fit(block)
        if not factor or coverage(index, required, block) < MIN_COVERAGE or not has_asked_value(query, block):
            continue
        scored.append((score * factor * (ROW_BOOST if asks_value and block.is_row else 1.0), block))
    if not scored:
        return None
    scored.sort(key=lambda pair: -pair[0])  # stable: equal scores keep index order
    top = scored[0][0]
    contenders = [block for score, block in scored if score >= top * NEAR_TIE]
    return max(contenders, key=lambda block: block.effective)  # first of equals = best scored
