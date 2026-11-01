"""One searchable view per crawl snapshot."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from bm25 import Index
from blocks import Block, parse_page
from contract import Snapshot
from pressfacts import DatedFact, dated_facts
from textnorm import clean

# A block repeated on this many pages is site chrome (forms, country lists), not an answer.
BOILERPLATE_MIN_PAGES = 8


@dataclass
class SnapshotView:
    snapshot_id: str
    taken_on: str
    index: Index
    dated_facts: list[DatedFact]  # read from the raw pages, before boilerplate removal
    twin_urls: dict[str, tuple[str, ...]]  # block text -> every page carrying exactly that text


def canonical_url(url: str) -> str:
    """Drop query, fragment and trailing slash so crawl variants of one page collapse."""
    return url.split("#")[0].split("?")[0].rstrip("/")


def build_view(snapshot: Snapshot) -> SnapshotView:
    seen: set[str] = set()
    blocks: list[Block] = []
    pages_per_text: dict[str, set[str]] = defaultdict(set)
    facts: list[DatedFact] = []
    for page in snapshot.pages:
        url = canonical_url(page.url)
        if url in seen:
            continue
        seen.add(url)
        facts += dated_facts(url, clean(page.markdown))
        for block in parse_page(url, page.markdown, page.title):
            pages_per_text[block.text].add(url)
            blocks.append(block)
    kept = [b for b in blocks if len(pages_per_text[b.text]) < BOILERPLATE_MIN_PAGES]
    twins = {text: tuple(sorted(urls)) for text, urls in pages_per_text.items() if 1 < len(urls) < BOILERPLATE_MIN_PAGES}
    return SnapshotView(snapshot.snapshot_id, snapshot.taken_on, Index(kept), facts, twins)
