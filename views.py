"""One searchable view per crawl snapshot."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from bm25 import Index
from blocks import Block, parse_page
from contract import Snapshot

# A block repeated on this many pages is site chrome (forms, country lists), not an answer.
BOILERPLATE_MIN_PAGES = 8


@dataclass
class SnapshotView:
    snapshot_id: str
    taken_on: str
    index: Index


def canonical_url(url: str) -> str:
    """Drop query, fragment and trailing slash so crawl variants of one page collapse."""
    return url.split("#")[0].split("?")[0].rstrip("/")


def build_view(snapshot: Snapshot) -> SnapshotView:
    seen: set[str] = set()
    blocks: list[Block] = []
    pages_per_text: dict[str, set[str]] = defaultdict(set)
    for page in snapshot.pages:
        url = canonical_url(page.url)
        if url in seen:
            continue
        seen.add(url)
        for block in parse_page(url, page.markdown):
            pages_per_text[block.text].add(url)
            blocks.append(block)
    kept = [b for b in blocks if len(pages_per_text[b.text]) < BOILERPLATE_MIN_PAGES]
    return SnapshotView(snapshot.snapshot_id, snapshot.taken_on, Index(kept))
