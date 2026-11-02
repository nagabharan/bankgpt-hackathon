"""Entry point for the grader: build() returns the knowledge system.

`contract` is importable because the scorer puts harness/ on sys.path before loading
this folder; always return the harness's own Answer type (the repeat-consistency check
uses isinstance on it).

Keep @dataclass definitions out of this file: the scorer loads it without registering it
in sys.modules, which breaks dataclasses under `from __future__ import annotations`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from askdate import split_date
from changes import change_answer, is_change_question
from conflicts import find_conflicts
from contract import Answer, Conflict, Snapshot, Source
from matching import MONEY_WORDS, asks_for_link, asks_where_online, best_link, duration_notes, focused_lines, links_of, select_blocks, trim_blocks, with_body, with_steps
from pressfacts import answer_from_facts, sentence_for
from qualifiers import parse_question
from textnorm import clean, content_tokens
from views import SnapshotView, build_view

# Refusal text must contain no digits: a refusal fails if it adds any number the
# question did not contain.
NOT_LISTED = "That isn't listed on our website; please contact us."
OFFER_ENDED = "That offer is no longer available."

MAX_SOURCES = 5


def cited_sources(view: SnapshotView, blocks: list) -> list[Source]:
    """The pages behind the answer: the first block's page, then every page that carries exactly
    the same text (rate sheets repeat on product pages), never a page with a different value."""
    urls = []
    for block in blocks:
        urls += [block.url, *view.twin_urls.get(block.text, ())]
    distinct = list(dict.fromkeys(urls))[:MAX_SOURCES]
    return [Source(url=u, snapshot_id=view.snapshot_id, quote=blocks[0].text[:200] if u == distinct[0] else None)
            for u in distinct]


def iso_date(as_of: Optional[str]) -> Optional[str]:
    """`as_of` as 'YYYY-MM-DD', accepting ISO dates/datetimes and 'M/D/YYYY'; None when absent or unreadable
    (an unreadable date means 'now', not 'before every snapshot')."""
    text = (as_of or "").strip()
    for candidate, pattern in ((text[:10], "%Y-%m-%d"), (text, "%m/%d/%Y")):
        try:
            return datetime.strptime(candidate, pattern).date().isoformat()
        except ValueError:
            continue
    return None


class KnowledgeBase:
    def __init__(self) -> None:
        self.views: list[SnapshotView] = []  # oldest first

    def ingest(self, snapshot: Snapshot) -> None:
        self.views.append(build_view(snapshot))

    def _view_for(self, day: Optional[str]) -> Optional[SnapshotView]:
        """Newest snapshot taken on or before `day` (newest overall when None)."""
        eligible = [v for v in self.views if day is None or v.taken_on <= day]
        return eligible[-1] if eligible else None

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        day = iso_date(as_of)
        known = [v for v in self.views if day is None or v.taken_on <= day]  # nothing after as_of may leak in
        if len(known) >= 2 and is_change_question(question):
            changed = change_answer(known, question)
            if changed is not None:
                return Answer(text=changed[0], sources=[changed[1]])
        newest = self.views[-1].taken_on if self.views else datetime.now().date().isoformat()
        stripped, written_date = split_date(question, int(newest[:4]))
        if written_date and written_date <= newest:  # "in May 2026" asks about May; "today (October 2026)" does not
            question = stripped
        else:
            written_date = None
        view = self._view_for(day or written_date)
        calendar_years = parse_question(clean(question)).calendar_years
        cutoff = f"{max(calendar_years)}-12-31" if calendar_years else ""
        if (calendar_years and self.views and max(calendar_years) < int(self.views[0].taken_on[:4])
                and MONEY_WORDS.intersection(content_tokens(clean(question)))):
            return Answer(text=NOT_LISTED, confident=False)  # the site's rates start in the first snapshot
        fact = answer_from_facts(question, view.dated_facts, cutoff) if view else None
        if fact is not None:
            return Answer(text=sentence_for(fact), sources=[Source(fact.url, view.snapshot_id, fact.value)])
        query = content_tokens(clean(question))
        if view and asks_where_online(query):
            link = best_link(view.index, query)
            if link:
                return Answer(text=f"You can do that online here: {link}",
                              sources=[Source(link, view.snapshot_id, link)])
        blocks = select_blocks(view.index, question) if view else []
        if not blocks:
            return Answer(text=NOT_LISTED, confident=False)
        future = [y for y in calendar_years if y > int(newest[:4])]
        if future and not any(str(y) in block.text for y in future for block in blocks):
            return Answer(text=NOT_LISTED, confident=False)  # a year after the newest crawl cannot be on the site
        if blocks[0].expired:  # the only match is an ended offer: do not present its terms as a live deal
            return Answer(text=OFFER_ENDED, confident=False)
        blocks = with_steps(view.index, with_body(view.index, trim_blocks(view.index, blocks, question), question), question)
        lines = focused_lines(view.index, blocks, question) + duration_notes(view.index, blocks, question)
        if asks_for_link(question):
            lines += links_of(blocks)
        return Answer(text="\n".join(lines), sources=cited_sources(view, blocks))

    def conflicts(self) -> list[Conflict]:
        return find_conflicts(self.views[-1]) if self.views else []


def build() -> KnowledgeBase:
    return KnowledgeBase()
