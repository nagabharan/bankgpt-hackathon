"""Sentence windows: the finer index that finds the one sentence of a long passage that answers.

A passage of several sentences ranks as a whole, so the sentence that holds the answer is diluted by
the ones around it. Every sentence of such a passage is indexed on its own, with the passage's page
context and its "Label:" heading as searchable context.

A question that asks for a kind of value ("how much", "how long", "when") is answered only by a
sentence that shows that kind of value; that is what lets a sentence be trusted when the question's
casual wording shares few words with it. Questions that do not ask for a value keep the passages the
main index found, and take a sentence only where it covers the question clearly better.
"""
from __future__ import annotations

import re
from array import array
from dataclasses import replace
from functools import lru_cache
from typing import NamedTuple, Optional

from bm25 import Index
from blocks import Block
from matching import (
    CANDIDATES, EXPIRED_FACTOR, LINK_WORDS, LIST_WORDS, MIN_COVERAGE, NEAR_TIE, RATE_WORDS, SUPERLATIVES,
    VALUE_SHAPES, _CODE, _EXISTENCE, _entities, _question, _without_framing_words, affirmative_factor, block_words, lacks_asked_tier, lacks_named_product,
    coverage, has_anchor, has_asked_value, is_present, with_aliases,
)
from segments import authority, is_article, page_type, segment_factor, url_match_factor
from textnorm import content_tokens

PASSAGE_MIN_CHARS = 140  # a shorter passage is already about the size of a sentence
SECOND_SENTENCE_SCORE = 0.6  # a second, separate sentence of the same passage joins when it scores this well
SHAPED_COVERAGE = 0.35  # a sentence that shows the asked kind of value needs this much of the question's words
INFORMATIVE_IDF = 3.0  # words rarer than this tell topics apart ("dupaco", "how", "much" do not)
INFORMATIVE_WORDS = 2  # a sentence must hold this many of them: one named word alone does not make an answer
RARE_IDF = 5.0
ANCHORLESS_WORDS = 3  # without the rarest word, a sentence needs this many telling words (it may use a synonym)
RARE_COVERAGE = 0.6  # of the question's rare words (by idf mass) this much must be in the sentence
SHAPE_BONUS = 0.15  # score gain per asked kind of value the sentence shows
COVERAGE_GAIN = 0.1  # without a value to look for, a sentence replaces the passage only if it covers this much more

_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"'(])|\n+")
# A full stop after one of these ends an abbreviation, not a sentence.
_ABBREVIATION = re.compile(r"(?:\b[A-Z]|\b[A-Za-z]\.[A-Za-z]|\b(?:St|Ave|Dr|Mr|Mrs|Ms|Inc|Co|No|vs|etc|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)|[ap]\.m)\.$")
_LABEL = re.compile(r"^([^:\n]{1,80}):\s")

# What a question asks for -> how that value looks in text.
_NUMBER = re.compile(r"\d")
_MONEY = re.compile(r"[$€£¥]\s?\d|\d\s?(?:dollars|percent|%)", re.I)
_DURATION = re.compile(r"\b\d+(?:\s*(?:-|to|or)\s*\d+)?\s*(?:business\s+)?(?:days?|weeks?|months?|years?|hours?|minutes?)\b", re.I)
_DATE = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d|\b(?:19|20)\d\d\b|\b\d{1,2}/\d{1,2}\b", re.I)
_AGE = re.compile(r"\bage\s+\d|\b\d+\s+(?:years?\s+old|or older|and older)|\bat\s+\d\d\b", re.I)
_PHONE = re.compile(r"\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b")
_SHAPES: tuple[tuple[re.Pattern, re.Pattern], ...] = (
    (re.compile(r"\bhow much\b|\bmax(?:imum)?\b|\bmin(?:imum)?\b|\blimit\b|\bworth\b|\bprize\b|\bsalary\b|\bhow big\b|\bwhat(?:'s| is| was)? (?:the )?(?:cost|price|fee|charge|amount)\b", re.I), _MONEY),
    (re.compile(r"\bhow many\b|\bnumber of\b|\bwhat range\b|\bhow high\b", re.I), _NUMBER),
    (re.compile(r"\bhow long\b|\bhow soon\b|\bhow quickly\b", re.I), _DURATION),
    (re.compile(r"\bwhat age\b|\bat what age\b|\bhow old\b", re.I), _AGE),
    (re.compile(r"\bphone\b|\bwhat number\b|\bnumber to call\b|\bwho do i call\b", re.I), _PHONE),
    (re.compile(r"\bwhen\b|\bwhat (?:date|day|year)\b|\bwhich year\b", re.I), _DATE),
)


class Window(NamedTuple):
    parent: int  # position of the passage in the block list
    sentence: int  # which sentence of that passage
    label: str  # the passage's "Label:" heading, "" when it has none


@lru_cache(maxsize=1 << 12)
def sentences_of(text: str) -> tuple[str, ...]:
    """Sentences of a passage, not splitting after an abbreviation ("U.S.", "9 a.m.")."""
    found: list[str] = []
    for piece in _BREAK.split(text):
        if not piece.strip():
            continue
        if found and _ABBREVIATION.search(found[-1]):
            found[-1] += " " + piece
        else:
            found.append(piece)
    return tuple(found)


class Windows:
    """The sentences of every long passage as a read-only sequence of blocks that are built when asked
    for, so the index holds postings and positions, not a second copy of each passage as block objects."""

    def __init__(self, passages: list[Block]) -> None:
        self._passages = passages
        self._parent, self._sentence = array("i"), array("H")
        self._labels: dict[int, str] = {}  # passage position -> its "Label:" heading
        for position, block in enumerate(passages):
            if block.is_row or len(block.text) < PASSAGE_MIN_CHARS:
                continue
            count = len(sentences_of(block.text))
            if count < 2:
                continue
            heading = _LABEL.match(block.text)
            if heading:
                self._labels[position] = heading.group(1)
            self._parent.extend([position] * count)
            self._sentence.extend(range(count))

    def __len__(self) -> int:
        return len(self._parent)

    def window(self, i: int) -> Window:
        return Window(self._parent[i], self._sentence[i], self._labels.get(self._parent[i], ""))

    def __getitem__(self, i: int) -> Block:
        w = self.window(i)
        passage = self._passages[w.parent]
        return Block(passage.url, sentences_of(passage.text)[w.sentence], passage.effective, (), False, "", (), "",
                     f"{passage.context} {w.label}".strip(), passage.page_date, passage.expired)

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def build_index(passages: list[Block], page_prior: dict[str, float], acronyms: dict[str, tuple[str, ...]]) -> Optional[Index]:
    windows = Windows(passages)
    if not len(windows):
        return None
    index = Index(windows)  # type: ignore[arg-type]
    index.page_prior = page_prior
    index.acronyms = acronyms
    index.windows = windows  # type: ignore[attr-defined]
    return index


def asked_shapes(question: str) -> list[re.Pattern]:
    """The kinds of value the wording asks for ("how much" -> an amount, "when" -> a date)."""
    return [shows for asks, shows in _SHAPES if asks.search(question)]


def _informative_present(index: Index, required: list[str], block: Block) -> int:
    return sum(1 for t in set(required) if index.idf(t) >= INFORMATIVE_IDF and is_present(index, t, block))


def _rare_coverage(index: Index, required: list[str], block: Block) -> float:
    """Share of the question's rare words (what it names, not what it does) that the sentence has."""
    rare = {t: index.idf(t) for t in set(required) if index.idf(t) >= RARE_IDF and not any(c.isdigit() for c in t)}
    total = sum(rare.values())
    return sum(w for t, w in sorted(rare.items()) if is_present(index, t, block)) / total if total else 1.0


def _candidates(index: Index, question: str, shapes: list[re.Pattern]) -> list[tuple[float, int]]:
    """(score, window) for the sentences that may answer, best first. The checks of the main index apply
    (tier and year fit, named places and codes, "does Dupaco have X"); a question that asks for a value is
    held to the weaker word coverage that the value itself makes up for."""
    q = _question(question)
    skipped = q.info.consumed.union(VALUE_SHAPES, SUPERLATIVES, LINK_WORDS, LIST_WORDS)
    required = [t for t in q.query if t not in skipped]
    hits = index.search(with_aliases(index, q.query), CANDIDATES)
    required = _without_framing_words(index, required, hits)
    topic_words = {t for t in required if not any(c.isdigit() for c in t)}
    entities = _entities(index, q.raw)
    codes = [t for t in required if _CODE.match(t)]
    existence = bool(_EXISTENCE.search(q.raw))
    topic_required = [t for t in required if t != "dupaco"]
    scored: list[tuple[float, int]] = []
    anchorless: list[tuple[float, int]] = []  # reserve: sentences without the rarest word but with three telling ones
    for score, doc_id in hits:
        block = index.blocks[doc_id]
        factor = q.info.fit(block)
        if not factor or (RATE_WORDS.intersection(q.query) and is_article(block.url)):
            continue
        if q.asks_money and page_type(block.url) == "calculator":
            continue
        shown = sum(bool(shape.search(block.text)) for shape in shapes)
        anchored = True
        if shapes and not shown:
            continue  # asked for an amount, a length of time, a date: a sentence without one cannot answer
        if shapes:
            if (_informative_present(index, required, block) < INFORMATIVE_WORDS
                    or coverage(index, required, block) < SHAPED_COVERAGE
                    or _rare_coverage(index, required, block) < RARE_COVERAGE):
                continue
            anchored = has_anchor(index, required, block)
            if not anchored and _informative_present(index, required, block) < ANCHORLESS_WORDS:
                continue
        elif coverage(index, required, block) < MIN_COVERAGE or not has_anchor(index, required, block):
            continue
        if not has_asked_value(q.query, block):
            continue
        if codes and not set(codes) <= block_words(block):
            continue
        if entities and not entities <= block_words(block):
            continue
        if lacks_asked_tier(q, block) or (q.asks_money and lacks_named_product(index, required, block)):
            continue
        if existence and coverage(index, topic_required, block) < 1.0:
            continue
        (scored if anchored else anchorless).append((score * factor * (1 + SHAPE_BONUS * shown) * segment_factor(q.intent, block.url)
                       * index.page_prior.get(block.url, 1.0) * affirmative_factor(q.raw, block)
                       * url_match_factor(topic_words, block.url) * (EXPIRED_FACTOR if block.expired else 1.0), doc_id))
    found = scored or anchorless
    found.sort(key=lambda pair: (-pair[0], pair[1]))
    return found


def select_sentences(sentences: Index, question: str) -> list[Block]:
    """The best sentence and, when its passage has another well-scoring sentence elsewhere, that one too.
    Among near-equal sentences the one on the better page, and then the newer one, wins."""
    q = _question(question)
    scored = _candidates(sentences, question, asked_shapes(question))
    if not scored:
        return []
    top = scored[0][0]
    blocks, window = sentences.blocks, sentences.windows.window
    contenders = [d for score, d in scored if score >= top * NEAR_TIE]
    in_url = lambda d: sum(word in blocks[d].url.lower() for word in set(q.query))
    fit = lambda d: round(q.info.fit(blocks[d]), 1)
    best = max(contenders, key=lambda d: (fit(d), authority(blocks[d].url), blocks[d].freshness, in_url(d)))
    chosen = [best]
    for score, d in scored:
        if (d != best and window(d).parent == window(best).parent and blocks[d].freshness == blocks[best].freshness
                and score >= top * SECOND_SENTENCE_SCORE):
            chosen.append(d)
            break
    chosen.sort(key=lambda d: window(d).sentence)
    return [_labelled(blocks[d], window(d), q.query) for d in chosen]


def _labelled(block: Block, window: Window, query: list[str]) -> Block:
    """A sentence from the middle of a passage keeps the passage's heading when the heading names what was asked."""
    if window.sentence == 0 or not window.label or not set(content_tokens(window.label)).intersection(query):
        return block
    return replace(block, text=f"{window.label}: {block.text}")


def _required(question: str) -> list[str]:
    q = _question(question)
    skipped = q.info.consumed.union(VALUE_SHAPES, SUPERLATIVES, LINK_WORDS, LIST_WORDS)
    return sorted({t for t in q.query if t not in skipped and not any(c.isdigit() for c in t)})


def better_than(index: Index, blocks: list[Block], found: list[Block], question: str) -> bool:
    """Whether the sentences answer where the passages found do not. With nothing found otherwise only a
    sentence that shows the asked value is trusted; against passages, a sentence must either show the value
    that no sentence of the passages shows, or cover the question clearly better."""
    if not found:
        return False
    shapes = asked_shapes(question)
    if not blocks:
        return bool(shapes)
    if blocks[0].is_row or any(block.text in other.text for block in found for other in blocks):
        return False  # a table row, or a passage that already holds the sentence
    if all(len(b.text) <= 100 and not b.text.rstrip().endswith((".", "!", "?")) for b in blocks) and {f.url for f in found} <= {b.url for b in blocks}:
        return False  # only the page's headline was found: the paragraphs under it are added, a lone sentence would cut them out
    required = _required(question)
    if shapes:
        return not any(shape.search(piece) and coverage(index, required, replace(b, text=piece)) >= SHAPED_COVERAGE
                       for b in blocks for piece in sentences_of(b.text) for shape in shapes)
    if not required:
        return False
    return max(coverage(index, required, b) for b in found) > max(coverage(index, required, b) for b in blocks) + COVERAGE_GAIN
