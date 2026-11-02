"""One searchable view per crawl snapshot."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Optional

from bm25 import Index
from blocks import Block, parse_page
from contract import Snapshot
from expiry import is_expired, page_looks_expired
from facts import warm_facts
from matching import warm
from pressfacts import DatedFact, dated_facts
from segments import page_prior
from sentences import build_index as build_sentence_index
from textnorm import STOPWORDS, clean, content_tokens, strip_links

# A block repeated on this many pages is site chrome (forms, country lists), not an answer.
BOILERPLATE_MIN_PAGES = 8


@dataclass
class SnapshotView:
    snapshot_id: str
    taken_on: str
    index: Index
    dated_facts: list[DatedFact]  # read from the raw pages, before boilerplate removal
    twin_urls: dict[str, tuple[str, ...]]  # block text -> every page carrying exactly that text
    sentences: Optional[Index] = None  # sentence windows of the long passages, for questions the whole passage buries


def canonical_url(url: str) -> str:
    """Drop query, fragment and trailing slash so crawl variants of one page collapse."""
    return url.split("#")[0].split("?")[0].rstrip("/")


def build_view(snapshot: Snapshot) -> SnapshotView:
    seen: set[str] = set()
    blocks: list[Block] = []
    pages_per_text: dict[str, set[str]] = defaultdict(set)
    facts: list[DatedFact] = []
    texts: list[str] = []  # every page's cleaned text, navigation included: where acronyms get spelled out
    for page in snapshot.pages:
        url = canonical_url(page.url)
        if url in seen:
            continue
        seen.add(url)
        cleaned = clean(page.markdown)  # the costliest step of ingest: once per page
        texts.append(cleaned)
        facts += dated_facts(url, cleaned)
        page_expired = page_looks_expired(url, page.title)
        for block in parse_page(url, page.markdown, page.title, cleaned):
            if page_expired or is_expired(block.text, snapshot.taken_on):
                block = replace(block, expired=True)  # an ended offer, judged at this snapshot's date
            pages_per_text[block.text].add(url)
            blocks.append(block)
    kept = [b for b in blocks if len(pages_per_text[b.text]) < BOILERPLATE_MIN_PAGES]
    twins = {text: tuple(sorted(urls)) for text, urls in pages_per_text.items() if 1 < len(urls) < BOILERPLATE_MIN_PAGES}
    index = Index(kept)
    index.page_prior = _page_priors(blocks, seen)
    index.acronyms = acronyms_defined(texts, index)
    sentences = build_sentence_index(kept, index.page_prior, index.acronyms)
    warm(index)
    warm_facts(index)
    return SnapshotView(snapshot.snapshot_id, snapshot.taken_on, index, facts, twins, sentences)


def _page_priors(blocks: list[Block], pages: set[str]) -> dict[str, float]:
    """Prior for every page: its type, and how many other pages link to it."""
    linked_from: dict[str, set[str]] = defaultdict(set)
    for block in blocks:
        for link in block.links:
            target = canonical_url(link)
            if target in pages and target != block.url:
                linked_from[target].add(block.url)
    return {url: page_prior(url, len(linked_from.get(url, ()))) for url in pages}


# "Home Equity Line of Credit (HELOC)", "Health Savings (HSA)": an acronym the site spells out once.
_ACRONYM_DEFINITION = re.compile(r"((?:[A-Za-z][\w'-]*\s+){1,7})\(([A-Z]{2,6})\)")
ACRONYM_MAX_BLOCKS = 100  # an acronym the site itself uses widely (PIN, APR) needs no expansion


def _spelled_out(words: list[str], acronym: str) -> list[str]:
    """The shortest run of trailing words whose initials spell the acronym, or spell all but its
    last letter ("Health Savings (HSA)"); [] when none does."""
    for start in range(len(words) - 1, -1, -1):
        initials = "".join(w[0] for w in words[start:]).upper()
        if initials == acronym or (len(initials) >= 2 and acronym[:-1] == initials):
            return words[start:]
        if len(initials) > len(acronym):
            break
    return []


_ACRONYM_MARK = re.compile(r"\(([A-Z]{2,6})\)")
_DEFINITION_REACH = 400  # characters before the acronym that can hold its seven words


def _definitions(text: str) -> list[tuple[str, str]]:
    """The (words, acronym) pairs of _ACRONYM_DEFINITION, found by looking only just before each
    "(ACR)" instead of scanning every word of the page."""
    found = []
    for mark in _ACRONYM_MARK.finditer(text):
        start = max(0, mark.start() - _DEFINITION_REACH)
        if start:  # begin at a word start, never inside a word
            gap = re.search(r"\s", text[start:mark.start()])
            start = start + gap.end() if gap else mark.start()
        window = text[start:mark.end()]
        found += [(m.group(1), m.group(2)) for m in _ACRONYM_DEFINITION.finditer(window) if m.end() == len(window)]
    return found


def acronyms_defined(texts: list[str], index: Index) -> dict[str, tuple[str, ...]]:
    """acronym token -> the words it stands for, for acronyms the blocks rarely use on their own."""
    found: dict[str, tuple[str, ...]] = {}
    for text in texts:
        for words, acronym in _definitions(strip_links(text)[0]):
            expansion = tuple(t for t in content_tokens(" ".join(_spelled_out(words.split(), acronym))) if t not in STOPWORDS)
            token = acronym.lower()
            if len(expansion) >= 2 and token not in found and token not in STOPWORDS \
                    and index.document_frequency(token) <= ACRONYM_MAX_BLOCKS:
                found[token] = expansion
    return found
