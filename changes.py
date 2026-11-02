"""Questions about what changed between snapshots ("did the rate change since May?").

They are answered from two snapshots: the newest and an earlier reference one, and both values
are printed with their dates, because the question is about the difference.
"""
from __future__ import annotations

import calendar
import re
from typing import Optional

from askdate import long_date
from blocks import Block
from conflicts import _tier_ranks
from contract import Source
from matching import MONEY_WORDS, select_blocks
from pressfacts import answer_from_facts, sentence_for
from qualifiers import YEAR_SPAN, caption_words, parse_question, year_spans_in
from textnorm import MONTHS, clean, content_tokens
from views import SnapshotView

_CHANGE_WORDS = re.compile(
    r"\b(?:chang(?:e|ed|es|ing)|since|versus|vs\.?|compared? (?:to|with)|increase[ds]?|decrease[ds]?|"
    r"(?:went|go|goes|gone|going) (?:up|down)|rise[sn]?|rose|dropp?(?:ed)?|fell|used to|previously|happen(?:ed|s)? to|differen(?:t|ce))\b", re.I)
# "since May", "from March 2026", "in May", "vs May": the month the question compares with.
_REFERENCE = re.compile(rf"\b(?:since|after|from|in|as of|during|versus|vs\.?|compared? (?:to|with)|than)\s+(?:the\s+)?"
                        rf"(?P<month>{'|'.join(MONTHS)})(?:\s+(?P<year>\d{{4}}))?\b")
_MONTH_WORD = re.compile(rf"\b(?:{'|'.join(MONTHS)})(?:\s+\d{{4}})?\b")


_STRONG_CHANGE = re.compile(r"\b(?:chang(?:ed|es)|since|between|versus|vs\.?|compared? (?:to|with)|happen(?:ed|s)? to|(?:went|go|goes|gone|going) (?:up|down))\b", re.I)
_PRODUCT_WORDS = frozenset("certificate special savings checking loan mortgage ira card".split())
_HOW_MANY = re.compile(r"\b(?:how many|number of|count of)\b", re.I)
_SEASONS = {"spring": 3, "summer": 6, "fall": 9, "autumn": 9, "winter": 12}
_SEASON_REFERENCE = re.compile(r"\b(?:since|from|in|during|vs\.?|versus)\s+(?:(?:the|last|this|past)\s+)?(?P<season>spring|summer|fall|autumn|winter)\b", re.I)


_ABBREVIATIONS = {m[:3].lower(): i + 1 for i, m in enumerate(MONTHS)} | {"sept": 9}
_MONTH_WORD_FORMS = sorted([*MONTHS, *(m[:3] for m in MONTHS), "Sept"], key=len, reverse=True)
_NAMED_DAY = re.compile(r"\b(?P<month>" + "|".join(_MONTH_WORD_FORMS) + r")\b\.?(?:\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\b)?(?:,?\s+(?P<year>\d{4})\b)?")


def _named_days(question: str, default_year: int) -> list[tuple[str, bool]]:
    """The dates a question names, in order: (ISO date, whether a day was given). A month with no year
    takes the next year named later in the question, else `default_year`."""
    found = []
    for m in _NAMED_DAY.finditer(question):
        word = m.group("month")[:4].lower() if m.group("month").lower() == "sept" else m.group("month")[:3].lower()
        if word == "may" and not (m.group("day") or m.group("year")) and not re.search(r"\b(?:in|since|from|to|and|between|of)\s+$", question[:m.start()], re.I):
            continue  # "May I apply" is a verb
        found.append([_ABBREVIATIONS[word], m.group("day"), m.group("year")])
    years = [y for _, _, y in found if y]
    result = []
    for month, day, year in found:
        year = int(year or (years[0] if years else default_year))
        last = calendar.monthrange(year, month)[1]
        result.append((f"{year}-{month:02d}-{int(day) if day and 1 <= int(day) <= last else last if day else 1:02d}", bool(day)))
    return result


def _range_views(views: list[SnapshotView], question: str):
    """(earlier, later) for a question naming two dates ("between May 12 and June 9", "from May to
    September"): each date picks the snapshot taken on or just before it; a month alone picks the
    first snapshot in it for the start and the last one in it for the end."""
    days = _named_days(question, int(views[-1].taken_on[:4]))
    if len(days) < 2:
        return None
    (start, start_exact), (end, end_exact) = sorted((days[0], days[-1]))  # "June compared to May" reads either way
    if not end_exact:  # a month alone: its last day
        end = f"{end[:8]}{calendar.monthrange(int(end[:4]), int(end[5:7]))[1]:02d}"
    first = next((v for v in views if v.taken_on >= start), None) if not start_exact else \
        next((v for v in reversed(views) if v.taken_on <= start), None)
    last = next((v for v in reversed(views) if v.taken_on <= end), None)
    return (first, last) if first and last and first is not last else None


def is_change_question(question: str) -> bool:
    """A change word together with a rate/fee word or a count ("how do I change my PIN" is not one)."""
    words = content_tokens(clean(question))
    topic_ok = MONEY_WORDS.intersection(words) or _HOW_MANY.search(question)
    if not topic_ok and _STRONG_CHANGE.search(question):  # "what changed on the 17-month special" names a product, not "rate"
        topic_ok = _PRODUCT_WORDS.intersection(words)
    return bool(_CHANGE_WORDS.search(question)) and bool(topic_ok)


def _reference_view(views: list[SnapshotView], question: str) -> Optional[SnapshotView]:
    """The first snapshot taken in or after the month the question names, else the oldest one;
    never the newest itself."""
    older = views[:-1]
    match = _REFERENCE.search(question)
    season = _SEASON_REFERENCE.search(question)
    if (match or season) and older:
        year = int((match.group("year") if match else None) or views[-1].taken_on[:4])
        month = MONTHS.index(match.group("month")) + 1 if match else _SEASONS[season.group("season").lower()]
        start = f"{year}-{month:02d}-01"
        later = [v for v in older if v.taken_on >= start]
        return later[0] if later else older[-1]
    return older[0] if older else None


def _topic(question: str) -> str:
    """The question without the change wording and the month it compares with."""
    question = _NAMED_DAY.sub(" ", question)
    question = _SEASON_REFERENCE.sub(" ", _REFERENCE.sub(" ", question))
    question = _MONTH_WORD.sub(" ", question)
    question = re.sub(r"\bfrom what to what\b|\b(?:between|snapshots?|crawls?|updates?|early|mid|late|the\s+one|recently|lately|right|whatever)\b", " ", question, flags=re.I)
    return _CHANGE_WORDS.sub(" ", question)


def change_answer(views: list[SnapshotView], question: str) -> Optional[tuple[str, Source]]:
    """('Now: ... / As of <date>: ...', source) or None when either snapshot has no answer."""
    span = _range_views(views, question)
    reference, latest = span if span else (_reference_view(views, question), views[-1])
    if reference is None:
        return None
    topic = _topic(question)
    then_fact, now_fact = (answer_from_facts(topic, v.dated_facts) for v in (reference, latest))
    if now_fact is not None and then_fact is not None:  # a count the press releases state: compare the two views
        source = Source(now_fact.url, latest.snapshot_id, now_fact.value)
        if now_fact.value == then_fact.value:
            return f"No change since {reference.taken_on}: {sentence_for(now_fact)}", source
        return f"Now: {sentence_for(now_fact)}\nAs of {reference.taken_on}: {sentence_for(then_fact)}", source
    now = select_blocks(latest.index, topic)
    then = select_blocks(reference.index, topic)
    if RETIRED_ROWS and then and then[0].is_row:
        gone = _retired(latest, reference, then[0], now[0] if now and now[0].is_row else None)
        if gone is not None:
            return gone
    if not now or not then:
        return None
    source = Source(now[0].url, latest.snapshot_id, now[0].text[:200])
    if CHRONOLOGY and now[0].is_row:
        between = [v for v in views if reference.taken_on <= v.taken_on <= latest.taken_on]
        lines = _chronology(between, now[0], topic, bool(_NEWEST_TIER.search(question)))
        if lines:
            return "\n".join(lines[:MAX_CHANGED_ROWS]), source
    changed = _changed_rows(latest, reference, now[0], then[0], topic)
    if not changed:
        return f"No change since {reference.taken_on}: " + "\n".join(b.text for b in now), source
    return "\n".join(f"Now: {n}\nAs of {reference.taken_on}: {t}" for n, t in changed), source


MAX_CHANGED_ROWS = 8
MAX_STEPS = 3
CHRONOLOGY = True  # a changed tier reads "X (as of May), then Y (from June). Current: Y." across every snapshot compared
RETIRED_ROWS = True  # a tier the old table listed and the new one does not: "was listed ... is not listed ..."
_NEWEST_TIER = re.compile(r"\b(?:newest|latest|new)[\s-]+(?:model|car|vehicle|auto)", re.I)
_FIRST_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _values(row: Block) -> tuple[tuple[str, str], ...]:
    """The row's figures as (kind, first number): "2.90% APR for 6 mos." and "2.90%" are one value."""
    return tuple((kind, m.group(0) if (m := _FIRST_NUMBER.search(cell)) else cell) for kind, cell in row.figures)


def _tier_key(row: Block, ranks: dict[tuple[str, str], list[int]]) -> str:
    """The row's tier (balance band, term, model-year rank) without the product name some pages
    repeat in the label, so one tier has the same key on the old and the new page."""
    product = set(content_tokens(row.caption))
    tier = [seg for seg in " ".join(row.label.split()).split(" | ") if not set(content_tokens(seg)) <= product]
    key = YEAR_SPAN.sub("", " | ".join(tier).lower())
    spans = year_spans_in(row.label)
    return f"{key} #tier{ranks[row.url, row.caption].index(spans[0][1])}" if spans else key


def _table_rows(view: SnapshotView, url: str, words: frozenset[str]) -> list[Block]:
    return [b for b in view.index.blocks if b.is_row and b.figures and b.url == url and caption_words(b) == words]


def _fitting(view: SnapshotView, rows: list[Block], best: Block, topic: str) -> dict[str, Block]:
    """The rows that fit the question as well as the best block does, keyed by tier."""
    info = parse_question(clean(topic))
    floor = round(info.fit(best, True), 1)
    ranks = _tier_ranks(rows)
    return {_tier_key(r, ranks): r for r in rows if round(info.fit(r, True), 1) >= floor}


def _changed_rows(latest: SnapshotView, reference: SnapshotView, now: Block, then: Block, topic: str) -> list[tuple[str, str]]:
    """(now, then) text for each tier of the topic's table whose figures differ between the two views.
    Each view contributes its best-ranked table (ranking already prefers the current page over a stale
    copy); tiers are matched by key. Single blocks that are not table rows are compared directly."""
    if not now.is_row:
        return [(now.text, then.text)] if now.text != then.text else []
    now_by_tier = _fitting(latest, _table_rows(latest, now.url, caption_words(now)), now, topic)
    # the old best block may sit in another table when the question is vague ("mortgage rates"):
    # then the product named by the new table finds the old one
    for old in (then, *select_blocks(reference.index, f"{topic} {now.caption}")[:1]):
        then_rows = _table_rows(reference, old.url, caption_words(old)) if old.is_row else []
        then_by_tier = _fitting(reference, then_rows, old, topic) if then_rows else {}
        if set(now_by_tier) & set(then_by_tier):
            pairs = [(now_by_tier[k], then_by_tier[k]) for k in now_by_tier if k in then_by_tier and _values(now_by_tier[k]) != _values(then_by_tier[k])]
            return [(n.text, t.text) for n, t in pairs[:MAX_CHANGED_ROWS]]
    return [(now.text, then.text)] if now.text != then.text else []


_KIND_NAMES = {"apy": "APY", "apr": "APR", "rate": "rate"}


def _figures(row: Block) -> str:
    """The row's rate figures as "3.81% rate, 3.85% APY" (footnote marks dropped)."""
    parts = []
    for kind, cell in row.figures:
        m = re.search(r"\d+(?:\.\d+)?\s*%?", cell)
        parts.append(f"{m.group(0).replace(' ', '') if m else cell} {_KIND_NAMES.get(kind, kind)}")
    return ", ".join(parts)


def _name(row: Block) -> str:
    label = " ".join(row.label.split())
    return label if row.caption.lower() in label.lower() else f"{row.caption}: {label}"


def _chronology(views: list[SnapshotView], now: Block, topic: str, newest_only: bool) -> list[str]:
    """One line per tier of the topic's table whose figures moved between the first and last of `views`:
    "<tier>: 3.85% APY (as of June 9, 2026), then 4.10% APY (from September 28, 2026). Current: 4.10% APY."
    Each view contributes its best-ranked table; tiers are matched by key. "The newest model years" keeps
    the newest model-year tier only."""
    latest = views[-1]
    now_by_tier = _fitting(latest, _table_rows(latest, now.url, caption_words(now)), now, topic)
    if newest_only:
        now_by_tier = {k: r for k, r in now_by_tier.items() if k.endswith("#tier0")}
    series: dict[str, list[tuple[str, Block]]] = {k: [(latest.taken_on, r)] for k, r in now_by_tier.items()}
    for view in views[:-1]:
        candidates = [*select_blocks(view.index, topic)[:1], *select_blocks(view.index, f"{topic} {now.caption}")[:1]]
        for old in candidates:
            rows = _table_rows(view, old.url, caption_words(old)) if old.is_row else []
            by_tier = _fitting(view, rows, old, topic) if rows else {}
            if set(now_by_tier) & set(by_tier):
                for key in now_by_tier:
                    if key in by_tier:
                        series[key].insert(len(series[key]) - 1, (view.taken_on, by_tier[key]))
                break
    lines = []
    for key, versions in series.items():
        steps = [versions[0]]
        for day, row in versions[1:]:
            if _values(row) != _values(steps[-1][1]):
                steps.append((day, row))
        if len(steps) < 2:
            continue
        steps = steps if len(steps) <= MAX_STEPS else [steps[0], steps[-1]]  # a long history: where it started and where it is
        parts = [f"{_figures(row)} ({'as of' if i == 0 else 'from'} {long_date(day)})" for i, (day, row) in enumerate(steps)]
        lines.append(f"{_name(steps[-1][1])}: " + ", then ".join(parts) + f". Current: {_figures(steps[-1][1])}.")
    return lines


def _retired(latest: SnapshotView, reference: SnapshotView, then: Block, now: Optional[Block]) -> Optional[tuple[str, Source]]:
    """A tier the old table listed that the newest version of that table no longer lists ("whatever happened
    to the 25-month special"): when it was listed, that it is gone, and the tiers the table lists now.
    The newest table is the one the question's best match sits in, else the old table's twin by caption."""
    words = caption_words(then)
    if now is not None and caption_words(now) != words:
        return None  # a vague question: the old best match sits in another table than the new one
    old_rows = _table_rows(reference, then.url, words)
    new_rows = _table_rows(latest, now.url, caption_words(now)) if now else _table_rows(latest, then.url, words)
    if not new_rows or not old_rows:
        return None
    old_ranks, new_ranks = _tier_ranks(old_rows), _tier_ranks(new_rows)
    if _tier_key(then, old_ranks) in {_tier_key(r, new_ranks) for r in new_rows}:
        return None
    old_keys = {_tier_key(r, old_ranks) for r in old_rows}
    added = [r for r in new_rows if _tier_key(r, new_ranks) not in old_keys]
    text = f"{_name(then)} ({_figures(then)}) was listed as of {long_date(reference.taken_on)} and is not listed as of {long_date(latest.taken_on)}."
    if added:
        text += " The table now lists: " + "; ".join(dict.fromkeys(" ".join(r.label.split()) for r in added)) + "."
    return text, Source(new_rows[0].url, latest.snapshot_id, new_rows[0].text[:200])
