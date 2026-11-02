"""Pick the blocks that answer a question: retrieve, check, apply qualifiers, prefer fresh,
then gather the sibling blocks that belong in the same answer."""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import NamedTuple

from bm25 import Index
from blocks import Block
from evidence import Kind, kind_for, phone_in
from qualifiers import QuestionInfo, fact_key, parse_question, year_spans_in
from schedules import OPENING, asks_schedules, distinct_schedules
from segments import authority, intent_of, is_article, page_type, segment_factor, url_match_factor
from textnorm import MONEY_MARK, clean, content_tokens, plural_nouns

CANDIDATES = 100  # wide enough that table rows survive the prose that mentions the same words
# Share of the question's idf mass the block must cover, else it does not answer the question.
MIN_COVERAGE = 0.6
NEAR_TIE = 0.9  # candidates within this share of the best score compete on freshness
# Share of the best score a sibling block needs to join the answer, by question kind.
HOURS_TIE = 0.6
LIST_TIE = 0.5
MAX_BLOCKS = 6
EXPIRED_FACTOR = 0.05  # an ended offer loses to any live one
MAX_LIST_BLOCKS = 8
MAX_LINKS = 3
NO_AMOUNT_FACTOR = 0.4  # a block without an amount, when the question asks how much

# Other words the site uses for the same thing ("Home Loans - Purchase" rows never say mortgage).
ALIASES = {
    "mortgage": ("home",), "cd": ("certificate",), "car": ("auto",), "vehicle": ("auto",),
    "eligible": ("member", "membership"), "eligibility": ("member", "membership"), "join": ("member", "membership"),
    "student": ("college", "university"), "college": ("student", "university"), "university": ("student", "college"),
    "cost": ("fee", "price"), "overdraw": ("overdraft",), "overdrawn": ("overdraft",), "charge": ("fee",),
    "mailing": ("box",), "become": ("join",), "qualify": ("qualifying", "qualified"), "open": ("opening", "opened"),
    "stop": ("close", "closing", "closure", "last"), "serving": ("service",),
    "oversea": ("international", "foreign"), "expedited": ("rush",), "expedite": ("rush",),
    "bonus": ("promo", "offer", "special"), "reward": ("promo", "offer"), "earn": ("promo", "offer", "get"), "wallet": ("payment", "pay"),
    "office": ("branch",), "promotion": ("special",), "thru": ("up",), "refi": ("refinance",),
    "transit": ("routing",), "share": ("certificate",),  # share: coverage only (ALIAS_NO_SEARCH)
}
ALIAS_NO_SEARCH = frozenset({"share", "become", "qualify", "open", "serving"})  # satisfies a coverage check but must not pull in other products
# Words that say which value to pick, not what the topic is; no block contains "highest".
SUPERLATIVES = frozenset("best highest higher largest lowest lower maximum minimum most least top".split())
HIGHEST = frozenset("best highest higher largest maximum top".split())
LOWEST = frozenset("lowest lower smallest least cheapest".split())
# Words that ask for a link or for every item, not topics.
LINK_WORDS = frozenset({"link", "url", "website", "site"})
LIST_WORDS = frozenset({"all", "every", "each", "list", "offer", "option", "type", "kind"})
# Plural nouns that do not mean "several items" ("savings" is a product name).
PLURAL_EXEMPT = frozenset({"hour", "month", "year", "day", "week", "saving", "apply", "reward"})
_MONEY_OR_PERCENT = re.compile(f"%|{MONEY_MARK.pattern}")
_EXISTENCE = re.compile(r"^\s*(?:does|do)\s+(?:dupaco|you|the credit union)\s+(?:have|offer|provide)\b", re.I)
_HOW_MUCH = re.compile(r"\bhow much\b|\bminimum\s+(?:\w+\s+)?(?:deposit|balance|amount|to)\b|\bneed to (?:deposit|pay|put|spend)\b", re.I)
_CLOCK_TIME = re.compile(r"\d\s*(?::\d\d)?\s*[ap]\.?m", re.I)
RATE_WORDS = frozenset(("rate", "apr", "apy", "interest"))
MONEY_WORDS = RATE_WORDS | frozenset(("fee", "cost", "charge", "price"))
# A percent within a few words of a rate word, either order: "APY: 3.50%", "4.25% introductory annual percentage rate".
_PERCENT = r"\d+(?:\.\d+)?%"
_RATE_TERM = r"(?:rate|apr|apy|interest|yield|dividend)"
_RATE_NEAR_PERCENT = re.compile(rf"{_RATE_TERM}\W+(?:\w+\W+){{0,6}}?{_PERCENT}|{_PERCENT}\W+(?:\w+\W+){{0,6}}?{_RATE_TERM}", re.I)
# A question word asks for a value of this shape; a block without one cannot answer it, and
# one with it needn't repeat the word itself (hours are written "8:30 a.m.", not "hours").
# A price sits next to a price word; an address has a number or an @ in it.
_PRICE_SHAPE = re.compile(rf"\b(?:price|cost|fee|charge)s?\b\W+(?:\w+\W+){{0,8}}?(?:{MONEY_MARK.pattern})|(?:{MONEY_MARK.pattern})\W*(?:\w+\W+){{0,8}}?\b(?:price|cost|fee|charge)", re.I)
_ADDRESS_SHAPE = re.compile(r"\b\d{5}\b|@|\bP\.?\s?O\.? Box\b|\b(?:St|Ave|Rd|Blvd|Hwy|Pkwy)\b|\b(?:Street|Avenue|Road|Boulevard|Highway|Drive)\b", re.I)
VALUE_SHAPES = {**dict.fromkeys(MONEY_WORDS, _MONEY_OR_PERCENT), "price": _PRICE_SHAPE, "hour": _CLOCK_TIME, "address": _ADDRESS_SHAPE}


@lru_cache(maxsize=1 << 15)  # the same candidate blocks and page contexts recur across questions
def _word_set(text: str) -> frozenset[str]:
    return frozenset(content_tokens(text))


_SUFFIXES = ("ing", "ed", "ral", "al", "er", "ion", "ment", "ly", "e")


@lru_cache(maxsize=1 << 16)
def root(word: str) -> str:
    """A word cut to a rough root so derived forms meet ("referring", "referral" and "refer";
    "expedited" and "expedite"). Only used to decide whether a question word is present in a block."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 5:
            word = word[:-len(suffix)]
            break
    if len(word) > 5 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]  # "referr" -> "refer"
    return word


@lru_cache(maxsize=1 << 15)
def _root_set(text: str) -> frozenset[str]:
    return frozenset(root(t) for t in content_tokens(text) if not t[0].isdigit())


def block_roots(block: Block) -> frozenset[str]:
    return _root_set(block.text) | _root_set(block.context)


def is_present(index: Index, token: str, block: Block) -> bool:
    """The question word is in the block: as written, through an alias, or as a derived form."""
    present = block_words(block)
    return (token in present or bool(present.intersection(aliases_of(index, token)))
            or (not token[0].isdigit() and root(token) in block_roots(block)))


def block_words(block: Block) -> frozenset[str]:
    """What the block is about: its own words plus its page's title, URL and headings."""
    return _word_set(block.text) | _word_set(block.context)


def coverage(index: Index, required: list[str], block: Block) -> float:
    """Fraction of the required tokens' idf mass that appears in the block (aliases count)."""
    wanted = {t: index.idf(t) for t in sorted(set(required))}  # sorted: float sums stay hash-seed independent
    total = sum(wanted.values())
    covered = sum(w for t, w in wanted.items() if is_present(index, t, block))
    return covered / total if total else 0.0


def has_anchor(index: Index, required: list[str], block: Block) -> bool:
    """The question's rarest topic word (the thing it names) must itself appear in the block;
    otherwise a block about something else can pass on the common words alone."""
    words = [t for t in required if not any(c.isdigit() for c in t)]  # amounts/terms are matched by qualifiers
    if not words:
        return True
    anchor = max(sorted(set(words)), key=index.idf)
    return is_present(index, anchor, block)


def aliases_of(index: Index, token: str) -> tuple[str, ...]:
    """Fixed aliases, else the words an acronym the site defines stands for (HSA, HELOC)."""
    return ALIASES.get(token) or index.acronyms.get(token, ())


def with_aliases(index: Index, query: list[str]) -> list[str]:
    return query + [alias for t in query if t not in ALIAS_NO_SEARCH for alias in aliases_of(index, t)]


def has_asked_value(query: list[str], block: Block) -> bool:
    for word, shape in VALUE_SHAPES.items():
        if word not in query:
            continue
        if not shape.search(block.text):
            return False
        if word in RATE_WORDS and "%" not in block.text:
            return False  # a price is not a rate
        if word in RATE_WORDS and not block.is_row and not _RATE_NEAR_PERCENT.search(block.text):
            return False  # a sentence must tie the percent to a rate ("80% of costs" is not a rate)
    return True


def _host(url: str) -> str:
    return url.split("//", 1)[-1].split("/", 1)[0]


def links_of(blocks: list[Block]) -> list[str]:
    """The blocks' link targets, keeping only same-host ones when the page has any."""
    found = list(dict.fromkeys(link for block in blocks for link in block.links))
    own = [link for link in found if _host(link) == _host(blocks[0].url)]
    return (own or found)[:MAX_LINKS]


def asks_for_link(question: str) -> bool:
    return bool(LINK_WORDS.intersection(content_tokens(clean(question))))


def wants_list(text: str, query: list[str]) -> bool:
    """A list word (all, each, offer...) or a plural topic noun (cards, rates): several rows are wanted.
    "Hours" is plural by nature and has its own band."""
    return bool(LIST_WORDS.intersection(query)) or bool(plural_nouns(text) - PLURAL_EXEMPT)


class Question(NamedTuple):
    """What is worked out about a question once, for every step that ranks blocks."""
    raw: str  # as asked: the yes/no pattern reads the original wording
    text: str  # cleaned
    query: list[str]  # content tokens plus implied words
    info: QuestionInfo
    intent: str
    asks_money: bool  # names a rate, fee or price
    several: bool  # wants a list of rows


def _question(question: str) -> Question:
    text = clean(question)
    query = with_implied_words(text, content_tokens(text))
    asks_money = bool(MONEY_WORDS.intersection(query))
    wants_extreme = bool(RATE_WORDS.intersection(query)) and bool((HIGHEST | LOWEST).intersection(query))
    return Question(question, text, query, parse_question(text), intent_of(question, asks_money), asks_money,
                    wants_list(text, query) or wants_extreme)  # the extreme is picked from the rates of several products


FRAMING_PROBE = 20  # best hits that must show a question word for it to count as part of the topic
# Verbs and adverbs that only frame an ask ("what does it carry", "ways to avoid", "back then").
FRAMING_WORDS = frozenset("""avoid carry cover load reply require describe recommend offered offer bring keep make come
    brand then back past far late outside long directly people article limit""".split())
_FRAMING_ENDINGS = ("ing", "ed", "ly")


def _is_framing(word: str) -> bool:
    return word in FRAMING_WORDS or (len(word) > 4 and word.endswith(_FRAMING_ENDINGS))


def _without_framing_words(index: Index, required: list[str], hits: list[tuple[float, int]]) -> list[str]:
    """Question words that only frame the ask ("looking", "load", "carry", "avoid") are used elsewhere on
    the site but by none of the best hits; they should not veto a block that has every other topic word.
    Names (nouns) are never dropped, nor is a word the site never uses: the question names something that is
    not there. At least half of the words, never fewer than two, are kept."""
    if len(required) < 3:
        return required
    best = [index.blocks[doc_id] for _, doc_id in hits[:FRAMING_PROBE]]
    framing = [t for t in sorted(set(required))
               if _is_framing(t) and index.document_frequency(t) > 0 and not any(is_present(index, t, b) for b in best)]
    if not framing or len(framing) > len(required) // 2:
        return required
    return [t for t in required if t not in framing]


def _qualifying(index: Index, q: Question, rows_only: bool) -> list[tuple[float, int]]:
    """(score, doc_id) for the searched blocks that answer the question, best first."""
    wants_link = bool(LINK_WORDS.intersection(q.query))
    skipped = q.info.consumed.union(VALUE_SHAPES, SUPERLATIVES, LINK_WORDS, LIST_WORDS)
    wants_amount = bool(_HOW_MUCH.search(q.raw))
    required = [t for t in q.query if t not in skipped and not (wants_amount and t == "amount")]
    topic_words = {t for t in required if not any(c.isdigit() for c in t)}

    hits = index.search(with_aliases(index, q.query), CANDIDATES, rows_only)
    required = _without_framing_words(index, required, hits)
    existence = bool(_EXISTENCE.search(q.raw))
    topic_required = [t for t in required if t != "dupaco"]
    scored = []
    for score, doc_id in hits:
        block = index.blocks[doc_id]
        factor = q.info.fit(block, ignore_tiers=q.several)
        if q.asks_money and page_type(block.url) == "calculator":
            continue  # a calculator's example rows are not the credit union's rates
        if not factor or coverage(index, required, block) < MIN_COVERAGE or not has_anchor(index, required, block):
            continue
        if not (block.links if wants_link else has_asked_value(q.query, block)):
            continue
        if existence and (coverage(index, topic_required, block) < 1.0):
            continue  # "does Dupaco have X": a block that lacks part of X does not show Dupaco has it
        if wants_amount and not _MONEY_OR_PERCENT.search(block.text):
            if q.asks_money:
                continue  # "how much is the fee" asks for an amount
            factor *= NO_AMOUNT_FACTOR  # "how much do I need to join": the passage with an amount is likelier
        scored.append((score * factor * segment_factor(q.intent, block.url) * index.page_prior.get(block.url, 1.0)
                       * affirmative_factor(q.raw, block) * url_match_factor(topic_words, block.url)
                       * (EXPIRED_FACTOR if block.expired else 1.0), doc_id))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))  # ties keep index order: deterministic
    return scored


def _score_candidates(index: Index, q: Question) -> list[tuple[float, int]]:
    """A question about a rate, fee or price is answered by a table row when one qualifies;
    otherwise (or for hours, policies) by any block, which may state the rate in prose."""
    if not q.asks_money:
        return _drop_superseded(index, _qualifying(index, q, rows_only=False))
    rows = _qualifying(index, q, rows_only=True)
    if rows:
        return _drop_superseded(index, rows)
    prose = _qualifying(index, q, rows_only=False)  # a rate stated in a sentence
    # Blog articles and press releases discuss rates in general or years ago; they do not state this
    # credit union's current rate.
    prose = [(s, d) for s, d in prose if not is_article(index.blocks[d].url)]
    if q.info.years:  # a rate for a model year must come from text that names a range covering it
        prose = [(s, d) for s, d in prose if any(lo <= y <= hi for y in q.info.years
                                                 for lo, hi in year_spans_in(index.blocks[d].text))]
    return prose


def _drop_superseded(index: Index, scored: list[tuple[float, int]]) -> list[tuple[float, int]]:
    """Where several pages carry the same table row, keep only the newest-dated version, however
    well an older copy matches the words: a stale page repeats the old tier labels and old rates."""
    newest: dict[tuple[str, str], str] = {}
    for _, doc_id in scored:
        block = index.blocks[doc_id]
        if block.is_row:
            key = fact_key(block)
            newest[key] = max(newest.get(key, ""), block.freshness)
    return [(s, d) for s, d in scored
            if not index.blocks[d].is_row or not index.blocks[d].freshness
            or index.blocks[d].freshness == newest[fact_key(index.blocks[d])]]


EVIDENCE_CANDIDATES = 1000
EVIDENCE_POOL = 40  # passing candidates compared when picking the best


def _by_evidence(index: Index, kind: Kind, q: Question) -> list[Block]:
    """The best block that shows the evidence this kind of question needs, else none."""
    skipped = q.info.consumed.union(VALUE_SHAPES, SUPERLATIVES, LINK_WORDS, LIST_WORDS, kind.filler)
    required = [t for t in q.query if t not in skipped]
    found = []
    for score, doc_id in index.search(with_aliases(index, q.query), EVIDENCE_CANDIDATES, only=index.showing(kind.name, kind.shows)):
        block = index.blocks[doc_id]
        if (kind.shows(block.text) and q.info.fit(block) and (not required or coverage(index, required, block) >= kind.min_coverage)
                and has_anchor(index, required, block)
                and not (kind.skip_articles and is_article(block.url))):
            found.append((score * segment_factor(q.intent, block.url) * index.page_prior.get(block.url, 1.0)
                          * (coverage(index, required, block) if required else 1.0)
                          * (EXPIRED_FACTOR if block.expired else 1.0), block))  # the fuller topic match wins
            if len(found) == EVIDENCE_POOL:
                break
    if not found:
        return []
    if kind.name == "phone":  # among the matches, the number that is everywhere is the main line
        best = max(found, key=lambda sb: (index.phone_counts[phone_in(sb[1].text)], sb[0]))[1]
        if not [t for t in required if t != "dupaco"]:  # nothing but "contact/phone/number": one plain sentence
            digits = phone_in(best.text)[-10:]
            return [Block(best.url, f"You can reach Dupaco at {digits[:3]}-{digits[3:6]}-{digits[6:]}.",
                          context=best.context, page_date=best.page_date)]
        return [best]
    return [max(found, key=lambda sb: sb[0])[1]]


# Wordings that ask for a rate or for opening hours without saying so.
_PAYS = re.compile(r"\b(?:how much|what)\b[^?]*\b(?:pay|pays|earn|earns|yield|dividends?)\b|\bwhat do(?:es)? [^?]*\b(?:pay|earn)\b", re.I)
_STATED_PERCENT = re.compile(r"\d(?:\.\d+)?\s*%")  # "only 6%?", "paying 4.5%": the question is about a rate


_BONUS_ROOTS = frozenset({"refer", "bonus", "reward", "promo", "promotion", "gift", "cashback"})


def with_implied_words(text: str, query: list[str]) -> list[str]:
    """Add the word a wording implies: "what does it pay?" or a stated percent asks for a rate,
    "when is it open?" for hours."""
    bonus = any(root(t) in _BONUS_ROOTS for t in query)  # "how much can I earn by referring a friend" asks for an amount
    extra = (["rate"] if (_PAYS.search(text) or _STATED_PERCENT.search(text)) and not MONEY_WORDS.intersection(query) and not bonus else [])
    extra += (["hour"] if OPENING.search(text) and "hour" not in query else [])
    return query + extra


def select_blocks(index: Index, question: str) -> list[Block]:
    """The best block plus its siblings (same page, date and kind) that belong in the answer."""
    if asks_schedules(question):
        schedules = distinct_schedules(index, question)
        if schedules:
            return list({block.text: block for block in schedules}.values())
    q = _question(question)
    kind = kind_for(question)
    if kind is not None and not q.asks_money:
        found = _by_evidence(index, kind, q)
        if found:
            return found
    scored = _score_candidates(index, q)
    if not scored:
        return []
    top = scored[0][0]
    blocks = index.blocks
    contenders = [doc_id for score, doc_id in scored if score >= top * NEAR_TIE]
    in_url = lambda d: sum(word in blocks[d].url.lower() for word in set(q.query))  # the page named for the topic
    fit = lambda d: round(q.info.fit(blocks[d], q.several), 1)  # the tier that was asked for before the newest one
    best = blocks[max(contenders, key=lambda d: (fit(d), authority(blocks[d].url), blocks[d].freshness, in_url(d)))]  # equals: best scored first

    tie = LIST_TIE if q.several else HOURS_TIE if "hour" in q.query else NEAR_TIE
    same_kind = [doc_id for score, doc_id in scored if score >= top * tie and blocks[doc_id].is_row == best.is_row]
    if q.several and best.is_row:  # a list may span pages and dates (personal and business cards)
        group = [d for d in same_kind if blocks[d].figures or not (RATE_WORDS | {"price"}).intersection(q.query)]  # a rate list holds rows with a rate
    else:
        group = [d for d in same_kind if blocks[d].url == best.url and blocks[d].freshness == best.freshness]
    if q.info.amounts:  # a named balance or loan size picks one amount tier; the wider tiers are other facts
        group = [d for d in group if fit(d) >= round(q.info.fit(best, q.several), 1)]
    kept = group[:MAX_LIST_BLOCKS if q.several else MAX_BLOCKS]  # the highest scored...
    unique = list(dict.fromkeys(blocks[d] for d in sorted(kept)))  # ...shown in page order
    unique = list({block.text: block for block in unique}.values())  # identical rows on two pages count once
    return _extreme_rows(unique, q.query)


_HOW_LONG = re.compile(r"\bhow long\b|\bhow many (?:days|months|years)\b", re.I)
_DURATION = re.compile(r"\b\d+ (?:day|month|year)s?\b")
_ENDS_AFTER = re.compile(r"\b(?:expires?|lasts?|ends?)\b", re.I)
_PLURAL_UNIT = re.compile(r"\b(?:[2-9]|\d{2,}) (?:day|month|year)\b(?!s)")  # clean() folds "six months" to "6 month"
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def duration_notes(index: Index, blocks: list[Block], question: str) -> list[str]:
    """"How long does the 2.90% intro rate last?": a rate row cannot say; the sentence of its page that
    repeats the rate together with a length of time does."""
    if not _HOW_LONG.search(question) or not blocks or not blocks[0].is_row or not blocks[0].figures:
        return []
    rate = _FIGURE_VALUE.search(blocks[0].figures[0][1])
    if rate is None:
        return []
    shown = rate.group(0) + "%"
    prose = index.cached("prose_by_url", lambda: _group_by_url(b for b in index.blocks if not b.is_row))
    found = [sentence for block in prose.get(blocks[0].url, ()) for sentence in _SENTENCE_END.split(clean(block.text))
             if shown in sentence and _DURATION.search(sentence)]
    if not found:
        return []
    sentence = next((s for s in found if _ENDS_AFTER.search(s)), found[0])  # "expires six months from..."
    end = _DURATION.search(sentence).end()
    sentence = sentence[:end + sentence[end:].find(",")] if "," in sentence[end:] else sentence  # the clause after is another fact
    return [_PLURAL_UNIT.sub(lambda m: m.group(0) + "s", sentence)]


def _group_by_url(blocks) -> dict[str, list[Block]]:
    grouped: dict[str, list[Block]] = {}
    for block in blocks:
        grouped.setdefault(block.url, []).append(block)
    return grouped


_FIGURE_VALUE = re.compile(r"\d+(?:\.\d+)?")


def _row_value(block: Block) -> float:
    """The row's highest rate/APY/APR figure."""
    return max((float(m.group(0)) for _, cell in block.figures for m in [_FIGURE_VALUE.search(cell)] if m), default=0.0)


def _extreme_rows(blocks: list[Block], query: list[str]) -> list[Block]:
    """"Highest/lowest rate": only the rows that hold the extreme value among those found."""
    wants_high, wants_low = HIGHEST.intersection(query), LOWEST.intersection(query)
    rows = [b for b in blocks if b.is_row and b.figures]
    if not (wants_high or wants_low) or not RATE_WORDS.intersection(query) or len(rows) < 2:
        return blocks
    values = [_row_value(b) for b in rows]
    target = max(values) if wants_high else min(v for v in values if v) if any(values) else 0.0
    return [b for b, v in zip(rows, values) if v == target]


_YES_NO_QUESTION = re.compile(r"^\s*(?:can|could|do|does|is|are|will)\s+(?:i|you|we|dupaco|my|your)\b", re.I)
_AFFIRMATIVE = re.compile(r"\b(?:yes|definitely|absolutely|you can|we offer|you may|available)\b", re.I)
AFFIRMATIVE_FACTOR = 1.3


def affirmative_factor(question: str, block: Block) -> float:
    """A yes/no question is best answered by a block that says yes (or 'you can ...')."""
    return AFFIRMATIVE_FACTOR if _YES_NO_QUESTION.search(question) and _AFFIRMATIVE.search(block.text) else 1.0


ACTION_WORDS = frozenset({"join", "apply", "open", "enroll", "register", "signup", "application"})
HOST_WORD_WEIGHT = 3


def asks_where_online(query: list[str]) -> bool:
    """"How do I join/apply/open ... online?": the answer is a link."""
    return "online" in query and bool(ACTION_WORDS.intersection(query))


def best_link(index: Index, query: list[str]) -> str:
    """The site link whose address best matches the question's words (host words count triple)."""
    uses: Counter = index.cached("site_links", lambda: Counter(
        link for block in index.blocks for link in block.links
        if link.startswith("http") and page_type(link) not in ("blog", "news", "calculator", "document")))
    words = set(query) - {"online"}
    def score(link: str) -> tuple[int, int, str]:
        host, _, path = link.split("//", 1)[-1].partition("/")
        host_hits = sum(w in set(re.split(r"\W+", host.lower())) for w in words)
        path_hits = sum(w in set(re.split(r"\W+", path.lower())) for w in words)
        return (HOST_WORD_WEIGHT * host_hits + path_hits, uses[link], link)
    best = max(uses, key=score, default="")
    return best if best and score(best)[0] > 0 else ""


BODY_MAX_BLOCKS = 2  # a bare heading is followed by its paragraph and, when short, the line after it
BODY_SHORT_LINE = 100


def _label(text: str) -> str:
    return text.split(":")[0] if ":" in text else ""


def _body_after(blocks: list[Block], i: int, continues: bool = True) -> list[Block]:
    """The paragraph right after block `i` on its page, and a short line after that (a date, a label).
    A heading that already holds a figure (an address line) only takes continuation lines under the
    same label."""
    body = []
    heading = blocks[i]
    has_figure = any(c.isdigit() for c in heading.text)
    if has_figure and not continues:
        return body
    for follower in blocks[i + 1:i + 1 + BODY_MAX_BLOCKS]:
        if follower.url != heading.url or follower.is_row or ((body or has_figure) and len(follower.text) > BODY_SHORT_LINE):
            break
        if has_figure and (not _label(heading.text) or _label(follower.text) != _label(heading.text)):
            break
        body.append(follower)
    return body


def _short_blocks(index: Index) -> dict[str, list[int]]:
    """Positions of each short, non-row block text (headings and headlines)."""
    found: dict[str, list[int]] = {}
    for i, b in enumerate(index.blocks):
        if not b.is_row and len(b.text) <= 100:
            found.setdefault(b.text, []).append(i)
    return found


_ADDRESS_WORDS = frozenset({"address", "mailing", "location", "located", "where"})
_PHONE_WORDS = frozenset({"phone", "call", "number", "contact", "text", "fax"})


def with_body(index: Index, blocks: list[Block], question: str = "") -> list[Block]:
    """A block that is only a heading ("Application period: ..." / a headline with no sentence) is
    followed by the paragraphs under it. A headline repeated on other pages (as a teaser) is read
    where its body is longest, which is the article itself."""
    copies = index.cached("heading_copies", lambda: _short_blocks(index))
    asked = _question(question).query if question else []
    continues = not question or bool(_ADDRESS_WORDS.intersection(asked))  # only an address spans several labelled lines
    result = []
    for block in blocks:
        if (block.is_row or len(block.text) > 100 or block.text.rstrip().endswith((".", "!", "?"))
                or any(other is not block and other.url == block.url for other in blocks)):  # the page's other passages are chosen already
            result.append(block)
            continue
        best = max(copies[block.text], key=lambda i: (sum(len(b.text) for b in _body_after(index.blocks, i)), -i))
        result.append(index.blocks[best])
        asks_phone = not question or bool(_PHONE_WORDS.intersection(asked))
        result.extend(b for b in _body_after(index.blocks, best, continues) if b not in blocks and (asks_phone or not phone_in(b.text)))
    return result


FOCUS_KEEP = 0.8  # a sentence, row or one-line fact stays if it covers this much of what the best one covers


def _is_heading(block: Block) -> bool:
    """A headline or label with no sentence of its own (with_body adds what is under it)."""
    return not block.is_row and len(block.text) <= 100 and not block.text.rstrip().endswith((".", "!", "?"))


def trim_blocks(index: Index, blocks: list[Block], question: str) -> list[Block]:
    """Of several rows found together, keep those that cover the question's words nearly as well as
    the best one: a neighbouring row about something else adds numbers, not answers."""
    q = _question(question)
    if len(blocks) < 2 or q.several or kind_for(question) or "hour" in q.query or asks_schedules(question):
        return blocks
    skipped = q.info.consumed.union(VALUE_SHAPES, SUPERLATIVES, LINK_WORDS, LIST_WORDS)
    required = sorted({t for t in q.query if t not in skipped and not any(c.isdigit() for c in t)})
    if not required:
        return blocks
    covers = [coverage(index, required, b) for b in blocks]
    best = max(covers)
    # a row or a one-line fact is compared closely; a longer passage only goes when it barely touches the question
    return [b for b, c in zip(blocks, covers) if _is_heading(b) or c >= best * (FOCUS_KEEP if b.is_row or len(b.text) <= FACT_LINE_CHARS else FOCUS_FAR)]


FACT_LINE_CHARS = 160
FOCUS_FAR = 0.5
FOCUS_MIN_CHARS = 200  # a shorter passage is shown whole
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"'(])|\n+")
_HEADING = re.compile(r"^([^:\n]{1,80}):\s")


def focused_lines(index: Index, blocks: list[Block], question: str) -> list[str]:
    """The answer text of each block. A long passage keeps the sentences that talk about the question
    and drops the ones that only add other numbers (a referral bonus beside a deposit amount);
    sentences without a number, and the sentence holding the asked value, always stay."""
    q = _question(question)
    if q.several or kind_for(question) or "hour" in q.query or asks_schedules(question):
        return [block.text for block in blocks]
    skipped = q.info.consumed.union(VALUE_SHAPES, SUPERLATIVES, LINK_WORDS, LIST_WORDS)
    required = sorted({t for t in q.query if t not in skipped and not any(c.isdigit() for c in t)})
    asked_numbers = set(re.findall(r"\d[\d,]*(?:\.\d+)?", question))
    return [_focus(index, block, required, asked_numbers) if required else block.text for block in blocks]


def _focus(index: Index, block: Block, required: list[str], asked_numbers: set[str]) -> str:
    text = block.text
    if block.is_row or len(text) <= FOCUS_MIN_CHARS:
        return text
    sentences = [s for s in _SENTENCE_BREAK.split(text) if s.strip()]
    if len(sentences) < 2:
        return text
    scores = [_sentence_coverage(index, required, s) for s in sentences]
    numbers = [set(re.findall(r"\d[\d,]*(?:\.\d+)?", s)) for s in sentences]
    best = max((score for score, n in zip(scores, numbers) if n), default=0.0)  # the answer is a sentence with a number
    if best == 0:
        return text
    kept = [s for s, score, n in zip(sentences, scores, numbers) if not n or n & asked_numbers or score >= best * FOCUS_KEEP]
    if len(kept) == len(sentences):
        return text
    heading = _HEADING.match(text)
    if heading and not kept[0].startswith(heading.group(1)):
        kept.insert(0, heading.group(1) + ":")
    return "\n".join(kept) if "\n" in text else " ".join(kept)


def _sentence_coverage(index: Index, required: list[str], sentence: str) -> float:
    words = _word_set(sentence)
    wanted = {t: index.idf(t) for t in required}
    total = sum(wanted.values())
    if not total:
        return 0.0
    return sum(w for t, w in wanted.items() if t in words or words.intersection(aliases_of(index, t))) / total
