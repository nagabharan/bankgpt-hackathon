"""Pick the blocks that answer a question: retrieve, check, apply qualifiers, prefer fresh,
then gather the sibling blocks that belong in the same answer."""
from __future__ import annotations

import re

from bm25 import Index
from blocks import Block
from qualifiers import QuestionInfo, parse_question
from textnorm import MONEY_MARK, clean, content_tokens, plural_nouns

CANDIDATES = 100  # wide enough that table rows survive the prose that mentions the same words
# Share of the question's idf mass the block must cover, else it does not answer the question.
MIN_COVERAGE = 0.6
NEAR_TIE = 0.9  # candidates within this share of the best score compete on freshness
# Share of the best score a sibling block needs to join the answer, by question kind.
HOURS_TIE = 0.6
LIST_TIE = 0.5
MAX_BLOCKS = 6
MAX_LINKS = 3

# Other words the site uses for the same thing ("Home Loans - Purchase" rows never say mortgage).
ALIASES = {"mortgage": ("home",), "cd": ("certificate",)}
# Words that say which value to pick, not what the topic is; no block contains "highest".
SUPERLATIVES = frozenset("best highest higher largest lowest lower maximum minimum most least top".split())
# Words that ask for a link or for every item, not topics.
LINK_WORDS = frozenset({"link", "url", "website", "site"})
LIST_WORDS = frozenset({"all", "every", "each", "list", "offer", "option", "type", "kind"})
PLURAL_EXEMPT = frozenset({"hour"})  # plural nouns that do not mean "several items"
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


def has_anchor(index: Index, required: list[str], block: Block) -> bool:
    """The question's rarest topic word (the thing it names) must itself appear in the block;
    otherwise a block about something else can pass on the common words alone."""
    words = [t for t in required if not any(c.isdigit() for c in t)]  # amounts/terms are matched by qualifiers
    if not words:
        return True
    anchor = max(sorted(set(words)), key=index.idf)
    present = set(content_tokens(block.text))
    return anchor in present or bool(present.intersection(ALIASES.get(anchor, ())))


def with_aliases(query: list[str]) -> list[str]:
    return query + [alias for t in query for alias in ALIASES.get(t, ())]


def has_asked_value(query: list[str], block: Block) -> bool:
    return all(shape.search(block.text) for word, shape in VALUE_SHAPES.items() if word in query)


def _host(url: str) -> str:
    return url.split("//", 1)[-1].split("/", 1)[0]


def links_of(blocks: list[Block]) -> list[str]:
    """The blocks' link targets, keeping only same-host ones when the page has any."""
    found = list(dict.fromkeys(link for block in blocks for link in block.links))
    own = [link for link in found if _host(link) == _host(blocks[0].url)]
    return (own or found)[:MAX_LINKS]


def asks_for_link(question: str) -> bool:
    return bool(LINK_WORDS.intersection(content_tokens(clean(question))))


def wants_list(text: str, query: list[str]) -> bool:
    """A list word (all, each, offer...) or a plural topic noun (cards, rates): several rows are wanted.
    "Hours" is plural by nature and has its own band."""
    return bool(LIST_WORDS.intersection(query)) or bool(plural_nouns(text) - PLURAL_EXEMPT)


def _qualifying(index: Index, info: QuestionInfo, query: list[str], several: bool,
                rows_only: bool) -> list[tuple[float, int]]:
    """(score, doc_id) for the searched blocks that answer the question, best first."""
    wants_link = bool(LINK_WORDS.intersection(query))
    skipped = info.consumed.union(VALUE_SHAPES, SUPERLATIVES, LINK_WORDS, LIST_WORDS)
    required = [t for t in query if t not in skipped]

    scored = []
    for score, doc_id in index.search(with_aliases(query), CANDIDATES, rows_only):
        block = index.blocks[doc_id]
        factor = info.fit(block, ignore_tiers=several)
        if not factor or coverage(index, required, block) < MIN_COVERAGE or not has_anchor(index, required, block):
            continue
        if not (block.links if wants_link else has_asked_value(query, block)):
            continue
        scored.append((score * factor, doc_id))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))  # ties keep index order: deterministic
    return scored


def _score_candidates(index: Index, info: QuestionInfo, query: list[str], several: bool) -> list[tuple[float, int]]:
    """A rate or fee question is answered by a table row when one qualifies; otherwise by any block."""
    if any(t in VALUE_SHAPES for t in query):
        rows = _qualifying(index, info, query, several, rows_only=True)
        if rows:
            return rows
    return _qualifying(index, info, query, several, rows_only=False)


def select_blocks(index: Index, question: str) -> list[Block]:
    """The best block plus its siblings (same page, date and kind) that belong in the answer."""
    text = clean(question)
    query = content_tokens(text)
    several = wants_list(text, query)
    scored = _score_candidates(index, parse_question(text), query, several)
    if not scored:
        return []
    top = scored[0][0]
    blocks = index.blocks
    contenders = [doc_id for score, doc_id in scored if score >= top * NEAR_TIE]
    best = blocks[max(contenders, key=lambda d: blocks[d].effective)]  # first of equals = best scored

    tie = LIST_TIE if several else HOURS_TIE if "hour" in query else NEAR_TIE
    group = [doc_id for score, doc_id in scored
             if score >= top * tie and blocks[doc_id].url == best.url
             and blocks[doc_id].effective == best.effective and blocks[doc_id].is_row == best.is_row]
    kept = sorted(group[:MAX_BLOCKS])  # the highest scored, shown in page order
    return list(dict.fromkeys(blocks[d] for d in kept))
