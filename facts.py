"""Typed facts of the thing they are about: a phone number, a street address, a fee amount or a date.

A question that asks for one kind of value ("what is the phone number of the X", "how much is the Y
fee", "when does Z close") is answered from the one fact of that kind whose subject is the thing the
question names. Subject words come from the fact's own sentence and, less strongly, from its page's
title, URL and headings, so a sentence that never names its place still belongs to the right place.
When two different facts fit equally well the store stays silent and the ordinary search answers.
"""
from __future__ import annotations

import re
from typing import NamedTuple, Optional

from bm25 import Index
from blocks import Block
from evidence import phone_in
from matching import RATE_WORDS, aliases_of, root
from sentences import sentences_of
from textnorm import MONTHS, clean, content_tokens

_ADDRESS = re.compile(
    r"\b\d{1,5}\s+(?:[NSEW]\.?\s+)?[A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,2}\s+"
    r"(?:Ave|Avenue|St|Street|Rd|Road|Blvd|Boulevard|Hwy|Highway|Dr|Drive|Way|Pkwy|Ln|Lane|Ct|Court)\b\.?"
    r"|\bP\.?\s?O\.?\s+Box\s+\d+")
_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
_DATE = re.compile(rf"\b(?:{'|'.join(MONTHS)})\.?\s+\d{{1,2}}\b")

# What each kind of question asks, and the words that only say so (they are not the subject).
_ASKS = (
    ("phone", re.compile(r"\b(?:phone|telephone|call|dial)\b", re.I), frozenset("phone telephone call dial number".split())),
    ("address", re.compile(r"\baddress\b|\bwhere (?:is|are)\b|\bwhere's\b|\bstreet\b|\bwhere .{0,30}\blocated\b", re.I),
     frozenset("address street mailing located location where find st ave rd blvd hwy dr".split())),
    ("money", re.compile(r"\bhow much\b|\b(?:fee|cost|charge|price)s?\b", re.I), frozenset("fee cost charge price pay amount money need much dollar".split())),
    ("date", re.compile(r"\bwhen\b|\bwhat (?:date|day)\b|\bapplication (?:period|window)\b|\bdeadline\b", re.I),
     frozenset("date day application period window deadline".split())),
)
_EARNS = frozenset("pay pays earn earns yield dividend".split())  # what an account pays is a rate
_IGNORED = frozenset("dupaco credit union way".split())  # on nearly every page: no help in telling things apart
RARE_IDF = 4.0  # a question word this rare must be found in the fact's sentence or page
MIN_MASS = 0.7  # share of the subject words' weight the fact must carry
CONTEXT_WEIGHT = 0.6  # a subject word found only in the page's title/URL/headings counts less
AMBIGUITY = 0.9  # a different value scoring this close to the best leaves the question to the search


class Fact(NamedTuple):
    kind: str
    value: str  # what to look for in another answer: digits of a phone number, the street, the amount, the date
    sentence: str
    block: Block  # the sentence as a block of its page (url, context)
    words: frozenset[str]  # words of the sentence
    roots: frozenset[str]
    context: frozenset[str]  # words of the page's title, URL and headings
    context_roots: frozenset[str]


def _teaser(text: str) -> bool:
    return "Keep reading" in text or text.rstrip().endswith("...")  # a listing page's start of another page


def _values(sentence: str, is_row: bool) -> list[tuple[str, str]]:
    found = []
    phone = phone_in(sentence)
    if phone:
        found.append(("phone", phone))
    address = _ADDRESS.search(sentence)
    if address:
        found.append(("address", re.sub(r"[\s,]", "", address.group(0)).lower()))
    money = _MONEY.search(sentence)
    if money:
        found.append(("money", re.sub(r"[\s,]", "", money.group(0))))
    date = _DATE.search(sentence)
    if date and not is_row:
        found.append(("date", re.sub(r"[\s,]", "", date.group(0)).lower()))
    return found


def build_facts(index: Index) -> list[Fact]:
    facts = []
    for block in index.blocks:
        if block.expired or _teaser(block.text):
            continue
        context = frozenset(content_tokens(block.context))
        context_roots = frozenset(root(c) for c in context)
        pieces = [block.text] if block.is_row else [s for line in block.text.split("\n") for s in sentences_of(line)]
        for sentence in pieces:
            for kind, value in _values(sentence, block.is_row):
                words = frozenset(content_tokens(sentence))
                piece = Block(block.url, sentence, block.effective, (), block.is_row, block.label, (), block.caption,
                              block.context, block.page_date, block.expired)
                facts.append(Fact(kind, value, sentence.strip(), piece, words, frozenset(root(w) for w in words), context, context_roots))
    return facts


def asked_kind(question: str) -> Optional[tuple[str, frozenset[str]]]:
    """(kind of value, words that only say so) for a question that asks for one, else None."""
    for kind, pattern, cues in _ASKS:
        if pattern.search(question):
            return kind, cues
    return None


def _present(index: Index, token: str, words: frozenset[str], roots: frozenset[str]) -> bool:
    return token in words or root(token) in roots or bool(words.intersection(aliases_of(index, token)))


def typed_fact(index: Index, question: str) -> Optional[Fact]:
    """The one fact that answers the question, or None when no fact fits or two different ones fit alike."""
    asked = asked_kind(question)
    if asked is None:
        return None
    kind, cues = asked
    tokens = content_tokens(clean(question))
    if kind == "money" and (RATE_WORDS | _EARNS).intersection(tokens):
        return None  # rates have their own tables and tiers
    subject = [t for t in dict.fromkeys(tokens) if t not in cues and t not in _IGNORED and not t[0].isdigit()]
    if len(subject) < (2 if kind in ("money", "date") else 1):
        return None  # one word names too many things to pick a fee or a date by
    numbers = [t for t in tokens if t[0].isdigit()]  # "24-month": a different number is a different thing
    weight = {t: index.idf(t) for t in subject}
    total = sum(weight.values())
    facts = index.cached("typed_facts", lambda: build_facts(index))
    scored: dict[str, tuple[float, Fact]] = {}
    for fact in facts:
        if fact.kind != kind:
            continue
        if not all(n in fact.words for n in numbers):
            continue
        own = sum(w for t, w in weight.items() if _present(index, t, fact.words, fact.roots))
        if not own:
            continue  # the sentence itself must be about the subject
        near = {t: kind != "money" and _present(index, t, fact.context, fact.context_roots) for t in subject}  # a fee line names its own thing
        seen = sum(w for t, w in weight.items() if _present(index, t, fact.words, fact.roots) or near[t])
        if seen / total < MIN_MASS or any(w >= RARE_IDF and not (_present(index, t, fact.words, fact.roots) or near[t])
                                          for t, w in weight.items()):
            continue
        focus = sum(1 for t in subject if _present(index, t, fact.words, fact.roots)) / max(len(fact.words), 1)
        score = (own + CONTEXT_WEIGHT * (seen - own)) * (1 + focus)  # a short line about the subject beats a long one that mentions it
        if fact.value not in scored or score > scored[fact.value][0]:
            scored[fact.value] = (score, fact)
    if not scored:
        return None
    ranked = sorted(scored.values(), key=lambda pair: (-pair[0], pair[1].block.url, pair[1].sentence))
    if len(ranked) > 1 and ranked[1][0] >= AMBIGUITY * ranked[0][0]:
        return None
    return ranked[0][1]


def warm_facts(index: Index) -> None:
    """Build the typed facts at ingest so that no answer pays for them."""
    index.cached("typed_facts", lambda: build_facts(index))
