"""Pick the blocks that answer a question: retrieve, check, apply qualifiers, prefer fresh,
then gather the sibling blocks that belong in the same answer."""
from __future__ import annotations

import re
from collections import Counter

from bm25 import Index
from blocks import Block
from evidence import Kind, kind_for, phone_in
from qualifiers import QuestionInfo, fact_key, parse_question, year_spans_in
from schedules import compares_hours, distinct_schedules
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
MAX_LIST_BLOCKS = 8
MAX_LINKS = 3

# Other words the site uses for the same thing ("Home Loans - Purchase" rows never say mortgage).
ALIASES = {
    "mortgage": ("home",), "cd": ("certificate",), "car": ("auto",), "vehicle": ("auto",),
    "eligible": ("member", "membership"), "eligibility": ("member", "membership"), "join": ("member", "membership"),
    "student": ("college", "university"), "college": ("student", "university"), "university": ("student", "college"),
    "cost": ("fee", "price"), "wallet": ("payment", "pay"),
    "office": ("branch",), "promotion": ("special",), "thru": ("up",), "refi": ("refinance",),
    "transit": ("routing",), "share": ("certificate",),
}
# Words that say which value to pick, not what the topic is; no block contains "highest".
SUPERLATIVES = frozenset("best highest higher largest lowest lower maximum minimum most least top".split())
# Words that ask for a link or for every item, not topics.
LINK_WORDS = frozenset({"link", "url", "website", "site"})
LIST_WORDS = frozenset({"all", "every", "each", "list", "offer", "option", "type", "kind"})
PLURAL_EXEMPT = frozenset({"hour", "month", "year", "day", "week"})  # plural nouns that do not mean "several items"
_MONEY_OR_PERCENT = re.compile(f"%|{MONEY_MARK.pattern}")
_CLOCK_TIME = re.compile(r"\d\s*(?::\d\d)?\s*[ap]\.?m", re.I)
# A question word asks for a value of this shape; a block without one cannot answer it, and
# one with it needn't repeat the word itself (hours are written "8:30 a.m.", not "hours").
# Blog articles and press releases (segments.is_article) discuss rates in general or years ago; they do not
# state this credit union's current rate, so a rate/fee question is not answered from them.
RATE_WORDS = frozenset(("rate", "apr", "apy", "interest"))
MONEY_WORDS = RATE_WORDS | frozenset(("fee", "cost", "charge", "price"))
# A percent within a few words of a rate word, either order: "APY: 3.50%", "4.25% introductory annual percentage rate".
_PERCENT = r"\d+(?:\.\d+)?%"
_RATE_TERM = r"(?:rate|apr|apy|interest|yield|dividend)"
_RATE_NEAR_PERCENT = re.compile(rf"{_RATE_TERM}\W+(?:\w+\W+){{0,6}}?{_PERCENT}|{_PERCENT}\W+(?:\w+\W+){{0,6}}?{_RATE_TERM}", re.I)
VALUE_SHAPES = {**dict.fromkeys(MONEY_WORDS, _MONEY_OR_PERCENT), "hour": _CLOCK_TIME}


def block_words(block: Block) -> set[str]:
    """What the block is about: its own words plus its page's title, URL and headings."""
    return set(content_tokens(block.text)) | set(content_tokens(block.context))


def coverage(index: Index, required: list[str], block: Block) -> float:
    """Fraction of the required tokens' idf mass that appears in the block (aliases count)."""
    present = block_words(block)
    wanted = {t: index.idf(t) for t in sorted(set(required))}  # sorted: float sums stay hash-seed independent
    total = sum(wanted.values())
    covered = sum(w for t, w in wanted.items() if t in present or present.intersection(ALIASES.get(t, ())))
    return covered / total if total else 0.0


def has_anchor(index: Index, required: list[str], block: Block) -> bool:
    """The question's rarest topic word (the thing it names) must itself appear in the block;
    otherwise a block about something else can pass on the common words alone."""
    words = [t for t in required if not any(c.isdigit() for c in t)]  # amounts/terms are matched by qualifiers
    if not words:
        return True
    anchor = max(sorted(set(words)), key=index.idf)
    present = block_words(block)
    return anchor in present or bool(present.intersection(ALIASES.get(anchor, ())))


def with_aliases(query: list[str]) -> list[str]:
    return query + [alias for t in query for alias in ALIASES.get(t, ())]


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


def _qualifying(index: Index, info: QuestionInfo, query: list[str], several: bool, rows_only: bool,
                intent: str, question: str) -> list[tuple[float, int]]:
    """(score, doc_id) for the searched blocks that answer the question, best first."""
    wants_link = bool(LINK_WORDS.intersection(query))
    skipped = info.consumed.union(VALUE_SHAPES, SUPERLATIVES, LINK_WORDS, LIST_WORDS)
    required = [t for t in query if t not in skipped]
    topic_words = {t for t in required if not any(c.isdigit() for c in t)}

    scored = []
    for score, doc_id in index.search(with_aliases(query), CANDIDATES, rows_only):
        block = index.blocks[doc_id]
        factor = info.fit(block, ignore_tiers=several)
        if MONEY_WORDS.intersection(query) and page_type(block.url) == "calculator":
            continue  # a calculator's example rows are not the credit union's rates
        if not factor or coverage(index, required, block) < MIN_COVERAGE or not has_anchor(index, required, block):
            continue
        if not (block.links if wants_link else has_asked_value(query, block)):
            continue
        scored.append((score * factor * segment_factor(intent, block.url) * index.page_prior.get(block.url, 1.0)
                       * affirmative_factor(question, block) * url_match_factor(topic_words, block.url), doc_id))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))  # ties keep index order: deterministic
    return scored


def _score_candidates(index: Index, info: QuestionInfo, query: list[str], several: bool,
                      intent: str, question: str) -> list[tuple[float, int]]:
    """A question about a rate, fee or price is answered by a table row when one qualifies;
    otherwise (or for hours, policies) by any block, which may state the rate in prose."""
    if not MONEY_WORDS.intersection(query):
        return _drop_superseded(index, _qualifying(index, info, query, several, False, intent, question))
    rows = _qualifying(index, info, query, several, True, intent, question)
    if rows:
        return _drop_superseded(index, rows)
    prose = _qualifying(index, info, query, several, False, intent, question)  # a rate stated in a sentence
    prose = [(s, d) for s, d in prose if not is_article(index.blocks[d].url)]
    if info.years:  # a rate for a model year must come from text that names a range covering it
        prose = [(s, d) for s, d in prose if any(lo <= y <= hi for y in info.years
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


def _by_evidence(index: Index, kind: Kind, info: QuestionInfo, query: list[str], intent: str) -> list[Block]:
    """The best block that shows the evidence this kind of question needs, else none."""
    skipped = info.consumed.union(VALUE_SHAPES, SUPERLATIVES, LINK_WORDS, LIST_WORDS, kind.filler)
    required = [t for t in query if t not in skipped]
    found = []
    for score, doc_id in index.search(with_aliases(query), EVIDENCE_CANDIDATES, only=index.showing(kind.name, kind.shows)):
        block = index.blocks[doc_id]
        if (kind.shows(block.text) and info.fit(block) and (not required or coverage(index, required, block) >= kind.min_coverage)
                and has_anchor(index, required, block)
                and not (kind.skip_articles and is_article(block.url))):
            found.append((score * segment_factor(intent, block.url) * index.page_prior.get(block.url, 1.0)
                          * (coverage(index, required, block) if required else 1.0), block))  # the fuller topic match wins
            if len(found) == EVIDENCE_POOL:
                break
    if not found:
        return []
    if kind.name == "phone":  # among the matches, the number that is everywhere is the main line
        return [max(found, key=lambda sb: (index.phone_counts[phone_in(sb[1].text)], sb[0]))[1]]
    return [max(found, key=lambda sb: sb[0])[1]]


# Wordings that ask for a rate or for opening hours without saying so.
_PAYS = re.compile(r"\b(?:how much|what)\b[^?]*\b(?:pay|pays|earn|earns|yield|dividends?)\b|\bwhat do(?:es)? [^?]*\b(?:pay|earn)\b", re.I)
_OPENING = re.compile(r"\bwhat time\b|\bwhen (?:is|are)\b[^?]*\bopen\b|\bopen and close\b|\bopening\b|\bclosing time\b", re.I)


def with_implied_words(text: str, query: list[str]) -> list[str]:
    """Add the word a wording implies: "what does it pay?" asks for a rate, "when is it open?" for hours."""
    extra = (["rate"] if _PAYS.search(text) and not MONEY_WORDS.intersection(query) else [])
    extra += (["hour"] if _OPENING.search(text) and "hour" not in query else [])
    return query + extra


def select_blocks(index: Index, question: str) -> list[Block]:
    """The best block plus its siblings (same page, date and kind) that belong in the answer."""
    text = clean(question)
    query = with_implied_words(text, content_tokens(text))
    if compares_hours(question):
        schedules = distinct_schedules(index, question)
        if schedules:
            return list({block.text: block for block in schedules}.values())
    kind = kind_for(question)
    intent = intent_of(question, bool(MONEY_WORDS.intersection(query)))
    if kind is not None and not MONEY_WORDS.intersection(query):
        found = _by_evidence(index, kind, parse_question(text), query, intent)
        if found:
            return found
    several = wants_list(text, query)
    scored = _score_candidates(index, parse_question(text), query, several, intent, question)
    if not scored:
        return []
    top = scored[0][0]
    blocks = index.blocks
    contenders = [doc_id for score, doc_id in scored if score >= top * NEAR_TIE]
    in_url = lambda d: sum(word in blocks[d].url.lower() for word in set(query))  # the page named for the topic
    info = parse_question(text)
    fit = lambda d: round(info.fit(blocks[d], several), 1)  # the tier that was asked for before the newest one
    best = blocks[max(contenders, key=lambda d: (fit(d), authority(blocks[d].url), blocks[d].freshness, in_url(d)))]  # equals: best scored first

    tie = LIST_TIE if several else HOURS_TIE if "hour" in query else NEAR_TIE
    same_kind = [doc_id for score, doc_id in scored if score >= top * tie and blocks[doc_id].is_row == best.is_row]
    if several and best.is_row:  # a list may span pages and dates (personal and business cards)
        asks_rate = bool(MONEY_WORDS.intersection(query))
        group = [d for d in same_kind if blocks[d].figures or not asks_rate]  # a rate list holds rows with a rate
    else:
        group = [d for d in same_kind if blocks[d].url == best.url and blocks[d].freshness == best.freshness]
    kept = group[:MAX_LIST_BLOCKS if several else MAX_BLOCKS]  # the highest scored...
    unique = list(dict.fromkeys(blocks[d] for d in sorted(kept)))  # ...shown in page order
    return list({block.text: block for block in unique}.values())  # identical rows on two pages count once


_YES_NO_QUESTION = re.compile(r"^\s*(?:can|could|do|does|is|are|will)\s+(?:i|you|we|dupaco|my|your)\b", re.I)
_AFFIRMATIVE = re.compile(r"\b(?:yes|definitely|absolutely|you can|we offer|you may|available)\b", re.I)
AFFIRMATIVE_FACTOR = 1.3


def affirmative_factor(question: str, block: Block) -> float:
    """A yes/no question is best answered by a block that says yes (or 'you can ...')."""
    return AFFIRMATIVE_FACTOR if _YES_NO_QUESTION.search(question) and _AFFIRMATIVE.search(block.text) else 1.0


ACTION_WORDS = frozenset({"join", "apply", "open", "enroll", "register", "signup"})
HOST_WORD_WEIGHT = 3


def asks_where_online(query: list[str]) -> bool:
    """"How do I join/apply/open ... online?": the answer is a link."""
    return "online" in query and bool(ACTION_WORDS.intersection(query))


def best_link(index: Index, query: list[str]) -> str:
    """The site link whose address best matches the question's words (host words count triple)."""
    uses: Counter = Counter(link for block in index.blocks for link in block.links
                            if link.startswith("http") and page_type(link) not in ("blog", "news", "calculator", "document"))
    words = set(query) - {"online"}
    def score(link: str) -> tuple[int, int, str]:
        host, _, path = link.split("//", 1)[-1].partition("/")
        host_hits = sum(w in set(re.split(r"\W+", host.lower())) for w in words)
        path_hits = sum(w in set(re.split(r"\W+", path.lower())) for w in words)
        return (HOST_WORD_WEIGHT * host_hits + path_hits, uses[link], link)
    best = max(uses, key=score, default="")
    return best if best and score(best)[0] > 0 else ""
