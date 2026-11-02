"""Split a crawled markdown page into searchable blocks.

A block is the smallest unit we retrieve: one table row (carrying its caption, column
headers and the effective date printed after the table) or one short passage of prose.
"""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Iterator, NamedTuple, Optional

from pagedate import page_date
from textnorm import MONEY_MARK, clean, strip_links

MAX_PROSE_CHARS = 400
MIN_BLOCK_CHARS = 40  # shorter digit-free lines are headings, attached to the prose after them
MAX_LIST_LINE_CHARS = 160  # a paragraph of >= 3 lines this short is a list: one block per line
MIN_LIST_LINES = 3
MAX_CAPTION_CHARS = 80
EFFECTIVE_LOOKAHEAD_LINES = 4

# "Rates are effective as of 01/15/2030" (or "01/15/30") -> month, day, year groups.
_EFFECTIVE = re.compile(r"effective\s+(?:as of\s+)?(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}")
# Column header words -> kind of figure, first match wins ("Annual Percentage Rate" is apr, not rate).
COLUMN_KINDS = (
    ("payment", ("payment",)),
    ("points", ("point",)),
    ("apy", ("yield", "apy")),
    ("apr", ("percentage rate", "apr")),
    ("rate", ("rate",)),
    ("minimum", ("min.", "minimum")),  # "Min. to Open", "Min. to Earn": not part of a row's identity
    ("fee", ("fee",)),
)
# "Dividend Rate for 1st 6 Months" is a teaser rate: it must not be compared with standard rates.
_INTRO_COLUMN = re.compile(r"1st \d+ months?|introductory", re.I)
FIGURE_KINDS = frozenset({"apy", "apr", "rate"})  # the columns whose values can contradict each other
_EMPHASIS = re.compile(r"\*\*|__")
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)")
_HAS_DIGIT = re.compile(r"\d")


@dataclass(frozen=True)
class Block:
    url: str
    text: str
    effective: str = ""  # ISO date printed next to a rate table, "" when none
    links: tuple[str, ...] = field(default_factory=tuple)
    is_row: bool = False  # a table row (carries a value in its tier/column) rather than prose
    label: str = ""  # rows only: which row of its table (section, tier cells)
    figures: tuple[tuple[str, str], ...] = ()  # rows only: (column kind, cell) per rate/APY/APR cell
    caption: str = ""  # rows only: the table's title, which varies between pages
    context: str = ""  # page title, URL path words and nearest H1 > H2: searchable, never shown
    page_date: str = ""  # the date the page states for itself ("" when it states none)
    expired: bool = False  # the offer this block describes has ended (set when the snapshot date is known)

    @property
    def freshness(self) -> str:
        """How current this block's content is: its table's effective date, else its page's date."""
        return self.effective or self.page_date


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


def column_kind(header: str) -> str:
    """What a figure under this column header is: apy, apr, rate, points, payment, minimum, fee,
    promo (teaser rate) or other."""
    h = header.lower()
    if _INTRO_COLUMN.search(h):
        return "promo"
    for kind, words in COLUMN_KINDS:
        if any(word in h for word in words):
            return kind
    return "other"


class _Row(NamedTuple):
    text: str
    links: tuple[str, ...]
    label: str  # section and the tier cells (term, balance, model year): which row of the table
    figures: tuple[tuple[str, str], ...]  # (column kind, cell) for each rate/money cell


def _table_rows(caption: str, rows: list[str]) -> list[_Row]:
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
        kinds = [column_kind(h) for h in header] if header else ["other"] * len(cells)
        figures = tuple((k, strip_links(c)[0]) for k, c in zip(kinds, cells) if k in FIGURE_KINDS and _value_kind(c))
        names = [strip_links(c)[0] for k, c in zip(kinds, cells) if c and k == "other"]
        rendered.append(_Row(text, tuple(links), " | ".join(p for p in (section, *names) if p), figures))
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


def _path_segments(url: str) -> list[str]:
    """A URL's path segments as words, query and fragment dropped: '/auto-loans/?x' -> ['auto loans']."""
    path = url.split("?")[0].split("#")[0].split("//", 1)[-1].partition("/")[2]
    return [re.sub(r"[-_]+", " ", s) for s in path.split("/") if s]


def _all_path_words(url: str) -> str:
    """Every path segment of a page URL as words: '/personal/loans/auto-loans' -> 'personal loans auto loans'."""
    return " ".join(_path_segments(url))


def _looks_like_code(text: str) -> bool:
    """A fragment of embedded JSON/script: braces or '":' pairs with several quoted keys."""
    return text.count('"') >= 4 and (("{" in text or "}" in text) or text.count('":') >= 2 or '" :' in text)


def _path_words(url: str) -> str:
    """The last two path segments of a link as words: '.../personal/savings/certificates/?x' -> 'savings certificates'."""
    return " ".join(_path_segments(url)[-2:])


class _PageParser:
    """Walks a page once, emitting table-row and prose blocks."""

    def __init__(self, url: str, title: str = "", published: str = "") -> None:
        self.url = url
        self.published = published
        self.page_words = f"{title} {_all_path_words(url)}".strip()  # what the page is, for every block on it
        self.context = self.page_words  # plus the nearest H1 > H2 headings once seen
        self.h1 = self.h2 = ""
        self.blocks: list[Block] = []
        self.caption = ""  # latest short plain line; becomes the label of the next table
        self.heading = ""  # latest short digit-free line; prefixed to the prose after it
        self.link_context = ""  # path words of the latest web link, e.g. "savings certificates"
        self.paragraph: list[str] = []

    def flush_paragraph(self) -> None:
        if not self.paragraph:
            return
        lines, self.paragraph = self.paragraph, []
        is_list = len(lines) >= MIN_LIST_LINES and all(len(l) <= MAX_LIST_LINE_CHARS for l in lines)
        for line in (lines if is_list else [" ".join(lines)]):
            self._add_prose(line)

    def _add_prose(self, raw: str) -> None:
        if _looks_like_code(raw):
            return  # JSON-LD / script text embedded in the crawl is not content
        text, links = strip_links(_EMPHASIS.sub("", raw).lstrip("#* ").strip())
        if _is_caption(text):
            self.caption = text
        web_links = [link for link in links if link.startswith("http")]
        if web_links:
            self.link_context = _path_words(web_links[-1])
        if len(text) < MIN_BLOCK_CHARS and not _HAS_DIGIT.search(text):
            self.heading = text  # too short to answer anything alone
            return
        for piece in _prose_pieces(text):
            self.blocks.append(Block(self.url, f"{self.heading}: {piece}" if self.heading else piece,
                                     links=tuple(links), context=self.context, page_date=self.published))

    def note_heading(self, level: int, text: str) -> None:
        """Track the nearest H1 and H2 (deeper headings count as H2) for the blocks that follow."""
        text = _EMPHASIS.sub("", text).strip()
        if level == 1:
            self.h1, self.h2 = text, ""
        else:
            self.h2 = text
        self.context = " ".join(p for p in (self.page_words, self.h1, self.h2) if p)

    def _table_caption(self) -> str:
        """The caption; a one-word one ("Regular") borrows the nearest earlier link's path words
        ("savings certificates") so the rows still say what they are rates of."""
        if len(self.caption.split()) >= 2 or not self.link_context:
            return self.caption
        return f"{self.caption} {self.link_context}".strip()

    def add_table(self, rows: list[str], effective: str) -> None:
        caption = self._table_caption()
        for row in _table_rows(caption, rows):
            self.blocks.append(Block(self.url, row.text, effective, row.links, True, row.label, row.figures,
                                     caption, self.context, self.published))


def parse_page(url: str, markdown: str, title: str = "", cleaned: Optional[str] = None) -> list[Block]:
    """`cleaned` is clean(markdown), when the caller already has it."""
    lines = (clean(markdown) if cleaned is None else cleaned).splitlines()
    parser = _PageParser(url, title, page_date(markdown))
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
            heading = _HEADING.match(line)
            if heading:
                parser.note_heading(len(heading.group(1)), heading.group(2))  # boundaries stay as they were
            parser.paragraph.append(line)
        else:
            parser.flush_paragraph()
        i += 1
    parser.flush_paragraph()
    return parser.blocks
