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
from matching import asks_for_link, links_of, select_blocks
from pressfacts import answer_from_facts, sentence_for
from views import SnapshotView, build_view

# Refusal text must contain no digits: a refusal fails if it adds any number the
# question did not contain.
NOT_LISTED = "That isn't listed on our website; please contact us."


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
        if self.views and is_change_question(question):
            changed = change_answer(self.views, question)
            if changed is not None:
                return Answer(text=changed[0], sources=[changed[1]])
        newest = self.views[-1].taken_on if self.views else datetime.now().date().isoformat()
        stripped, written_date = split_date(question, int(newest[:4]))
        if written_date and written_date <= newest:  # "in May 2026" asks about May; "today (October 2026)" does not
            question = stripped
        else:
            written_date = None
        view = self._view_for(iso_date(as_of) or written_date)
        fact = answer_from_facts(question, view.dated_facts) if view else None
        if fact is not None:
            return Answer(text=sentence_for(fact), sources=[Source(fact.url, view.snapshot_id, fact.value)])
        blocks = select_blocks(view.index, question) if view else []
        if not blocks:
            return Answer(text=NOT_LISTED, confident=False)
        lines = [block.text for block in blocks]
        if asks_for_link(question):
            lines += links_of(blocks)
        return Answer(text="\n".join(lines), sources=cited_sources(view, blocks))

    def conflicts(self) -> list[Conflict]:
        return find_conflicts(self.views[-1]) if self.views else []


def build() -> KnowledgeBase:
    return KnowledgeBase()
