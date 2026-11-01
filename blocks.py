"""Split a crawled markdown page into searchable blocks.

A block is the smallest unit we retrieve: one table row (carrying its caption, column
headers and the effective date printed after the table) or one short passage of prose.
"""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Iterator

from textnorm import MONEY_MARK, clean, strip_links

MAX_PROSE_CHARS = 400
MIN_BLOCK_CHARS = 40  # shorter digit-free lines are headings, attached to the prose after them
MAX_LIST_LINE_CHARS = 160  # a paragraph of >= 3 lines this short is a list: one block per line
MIN_LIST_LINES = 3
MAX_CAPTION_CHARS = 80
EFFECTIVE_LOOKAHEAD_LINES = 4

# "Rates are effective as of 09/25/2026" (or "08/13/25") -> month, day, year groups.
_EFFECTIVE = re.compile(r"effective\s+(?:as of\s+)?(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}")
_EMPHASIS = re.compile(r"\*\*|__")
_HAS_DIGIT = re.compile(r"\d")


@dataclass(frozen=True)
class Block:
    url: str
    text: str
    effective: str = ""  # ISO date printed next to a rate table, "" when none
    links: tuple[str, ...] = field(default_factory=tuple)
    is_row: bool = False  # a table row (carries a value in its tier/column) rather than prose


def _iso(match: re.Match) -> str:
    month, day, year = match.groups()
    full_year = int(year) + 2000 if len(year) == 2 else int(year)
    return f"{full_year}-{int(month):02d}-{int(day):02d}"


def _cells(line: str) -> list[str]:
    inner = line.strip().removeprefix("|").removesuffix("|")  # keep empty cells next to the edges
    return [c.strip() for c in inner.split("|")]


def _is_caption(line: str) -> bool:
    return (0 < len(line) <= MAX_CAPTION_CHARS and not line.startswith(("|", "[", "!", "-"))
            and not _EFFECTIVE.search(line))


def _value_kind(cell: str) -> str:
    """'%' for a rate cell, '$' for a money cell, '' otherwise."""
    return "%" if "%" in cell else "$" if MONEY_MARK.search(cell) else ""


def _first_row_is_data(rows: list[str]) -> bool:
    """Some PDF fee tables have no header: the first row holds a value like the rows below it."""
    first, below = _cells(rows[0])[1:], _cells(rows[2])[1:] if len(rows) > 2 else []
    kinds = {_value_kind(c) for c in first if c}
    return bool(kinds) and "" not in kinds and kinds == {_value_kind(c) for c in below if c}


def _table_rows(caption: str, rows: list[str]) -> list[tuple[str, tuple[str, ...]]]:
    """Render each data row as 'caption | section | header cell | header cell'.

    A row with a single filled cell (e.g. '|Deposit Services Business||||') names the
    section of the rows after it."""
    header, body = _cells(rows[0]), rows[2:]  # rows[1] is the |---| separator
    if _first_row_is_data(rows):
        header, body = [], [rows[0], *body]
    section, rendered = "", []
    for line in body:
        cells = _cells(line)
        filled = [c for c in cells if c]
        if len(filled) == 1 and len(cells) > 1:
            section = filled[0]
            continue
        pairs = [f"{h} {c}".strip() for h, c in zip(header, cells) if c] if header else filled
        text, links = strip_links(" | ".join(p for p in (caption, section, *pairs) if p))
        rendered.append((text, tuple(links)))
    return rendered


def _prose_pieces(paragraph: str) -> Iterator[str]:
    """Yield the paragraph whole if short, else sentence-sized pieces (never over the cap)."""
    if len(paragraph) <= MAX_PROSE_CHARS:
        yield paragraph
        return
    piece = ""
    for sentence in _SENTENCE_END.split(paragraph):
        if piece and len(piece) + len(sentence) > MAX_PROSE_CHARS:
            yield from textwrap.wrap(piece, MAX_PROSE_CHARS)
            piece = ""
        piece = f"{piece} {sentence}".strip()
    if piece:
        yield from textwrap.wrap(piece, MAX_PROSE_CHARS)


def _effective_after(lines: list[str], start: int) -> str:
    """Effective date from the first few non-empty lines after a table, else ''."""
    seen = 0
    for line in lines[start:]:
        if not line.strip():
            continue
        match = _EFFECTIVE.search(line)
        if match:
            return _iso(match)
        seen += 1
        if seen >= EFFECTIVE_LOOKAHEAD_LINES or line.lstrip().startswith("|"):
            break
    return ""


class _PageParser:
    """Walks a page once, emitting table-row and prose blocks."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.blocks: list[Block] = []
        self.caption = ""  # latest short plain line; becomes the label of the next table
        self.heading = ""  # latest short digit-free line; prefixed to the prose after it
        self.paragraph: list[str] = []

    def flush_paragraph(self) -> None:
        if not self.paragraph:
            return
        lines, self.paragraph = self.paragraph, []
        is_list = len(lines) >= MIN_LIST_LINES and all(len(l) <= MAX_LIST_LINE_CHARS for l in lines)
        for line in (lines if is_list else [" ".join(lines)]):
            self._add_prose(line)

    def _add_prose(self, raw: str) -> None:
        text, links = strip_links(_EMPHASIS.sub("", raw).lstrip("#* ").strip())
        if _is_caption(text):
            self.caption = text
        if len(text) < MIN_BLOCK_CHARS and not _HAS_DIGIT.search(text):
            self.heading = text  # too short to answer anything alone
            return
        for piece in _prose_pieces(text):
            self.blocks.append(Block(self.url, f"{self.heading}: {piece}" if self.heading else piece,
                                     links=tuple(links)))

    def add_table(self, rows: list[str], effective: str) -> None:
        for text, links in _table_rows(self.caption, rows):
            self.blocks.append(Block(self.url, text, effective, links, is_row=True))


def parse_page(url: str, markdown: str) -> list[Block]:
    lines = clean(markdown).splitlines()
    parser = _PageParser(url)
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|"):
            parser.flush_paragraph()
            start = i
            while i < len(lines) and lines[i].strip().startswith("|"):
                i += 1
            rows = lines[start:i]
            if len(rows) > 2 and _TABLE_SEPARATOR.match(rows[1].strip()):
                parser.add_table(rows, _effective_after(lines, i))
            continue
        if line:
            parser.paragraph.append(line)
        else:
            parser.flush_paragraph()
        i += 1
    parser.flush_paragraph()
    return parser.blocks
