"""Entry point for the grader: build() returns the knowledge system.

`contract` is importable because the scorer puts harness/ on sys.path before loading
this folder; always return the harness's own Answer type (the repeat-consistency check
uses isinstance on it).

Keep @dataclass definitions out of this file: the scorer loads it without registering it
in sys.modules, which breaks dataclasses under `from __future__ import annotations`.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from askdate import split_date
from changes import change_answer, is_change_question
from conflicts import find_conflicts
from contract import Answer, Conflict, Snapshot, Source
from evidence import kind_for
from matching import MONEY_WORDS, asks_for_link, asks_where_online, best_link, complete_blocks, duration_notes, focused_lines, links_of, names_unknown_place, select_blocks
from pressfacts import answer_from_facts, sentence_for
from sentences import asked_shapes, better_than, select_sentences
from qualifiers import parse_question
from schedules import asks_schedules
from textnorm import clean, content_tokens, normalize_question
from views import SnapshotView, build_view

# Refusal text must contain no digits: a refusal fails if it adds any number the
# question did not contain.
_STATED_FEE = re.compile(r"\$(\d[\d,.]*)\s+(?:\w+\s+)?(?:fee|charge)", re.I)  # a premise: "a $12 monthly fee"
_FEE_WORD = re.compile(r"\b(?:fees?|charges?)\b", re.I)
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


_WHEN = re.compile(r"^\W*when\b|\bwhen (?:did|do|does|is|was|will|can|should)\b|\bwhat (?:date|day)\b", re.I)  # a date is looked for further down the page


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
        question = normalize_question(question)
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
        if view and names_unknown_place(view.index, question):
            return Answer(text=NOT_LISTED, confident=False)  # "the Omaha branch": the site has no such place
        ended = bool(blocks) and blocks[0].expired  # a named ended offer is the answer; no sentence replaces it
        if view and view.sentences is not None and not ended and not kind_for(question) and not asks_schedules(question):
            found = select_sentences(view.sentences, question)
            if better_than(view.index, blocks, found, question):
                blocks = found
        if not blocks:
            return Answer(text=NOT_LISTED, confident=False)
        future = [y for y in calendar_years if y > int(newest[:4])]
        if future and not any(str(y) in block.text for y in future for block in blocks):
            return Answer(text=NOT_LISTED, confident=False)  # a year after the newest crawl cannot be on the site
        stated_year = [y for y in calendar_years if 2010 <= y <= int(newest[:4])] if not re.search(r"\bborn\b", question, re.I) else []
        if stated_year and not parse_question(clean(question)).amounts and not MONEY_WORDS.intersection(content_tokens(clean(question))):
            shown = " ".join(f"{b.text} {b.context} {b.page_date} {b.effective}" for b in blocks)
            shown_years = [int(y) for y in re.findall(r"\b(?:19|20)\d\d\b", shown)]
            if shown_years and any(y < min(shown_years) - 1 for y in stated_year):
                return Answer(text=NOT_LISTED, confident=False)  # "in 2022": the passages are all years later (a one-year gap may be a recap)
        fee_claim = _STATED_FEE.search(question)
        if fee_claim and not any(fee_claim.group(1) in b.text for b in blocks) and not any(_FEE_WORD.search(b.text) for b in blocks):
            return Answer(text=NOT_LISTED, confident=False)  # "a $12 monthly fee": nothing says there is a fee at all
        if blocks[0].expired:  # the only match is an ended offer: do not present its terms as a live deal
            return Answer(text=OFFER_ENDED, confident=False)
        blocks = complete_blocks(view.index, blocks, question, asked_shapes(question) if _WHEN.search(question) else [])
        lines = focused_lines(view.index, blocks, question) + duration_notes(view.index, blocks, question)
        if asks_for_link(question):
            lines += links_of(blocks)
        return Answer(text="\n".join(lines), sources=cited_sources(view, blocks))

    def conflicts(self) -> list[Conflict]:
        return find_conflicts(self.views[-1]) if self.views else []


def build() -> KnowledgeBase:
    return KnowledgeBase()
