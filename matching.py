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
from qualifiers import TIER_WORDS, QuestionInfo, fact_key, parse_question, tier_words_in, year_spans_in
from schedules import OPENING, asks_schedules, distinct_schedules
from segments import authority, intent_of, is_article, page_type, segment_factor, url_match_factor
from textnorm import MONEY_MARK, MONTHS, STOPWORDS, STREET_ABBREVIATIONS, clean, content_tokens, plural_nouns, stem

CANDIDATES = 100  # wide enough that table rows survive the prose that mentions the same words
# Share of the question's idf mass the block must cover, else it does not answer the question.
MIN_COVERAGE = 0.6
NEAR_TIE = 0.9  # candidates within this share of the best score compete on freshness
# Share of the best score a sibling block needs to join the answer, by question kind.
HOURS_TIE = 0.6
LIST_TIE = 0.5
MAX_BLOCKS = 6
CONFIRMED_DIMENSIONS = 2  # qualifiers a row must satisfy to count as the asked row
CONFIRMED_FACTOR = 1.6  # per satisfied qualifier
CONFIRMED_COVERAGE = 0.2  # topic-word coverage still required of such a row
EXPIRED_FACTOR = 0.05  # an ended offer loses to any live one
MAX_LIST_BLOCKS = 8
MAX_LINKS = 3
INFLECTION_MIN_IDF = 5.0  # "backed" -> "back" is another word: a common base form is not accepted
UNASKED_OFFER_FACTOR = 0.3
FEE_STATED_FACTOR = 1.3
NO_AMOUNT_FACTOR = 0.4  # a block without an amount, when the question asks how much

# Other words the site uses for the same thing ("Home Loans - Purchase" rows never say mortgage).
ALIASES = {
    "mortgage": ("home",), "cd": ("certificate",), "car": ("auto",), "vehicle": ("auto",),
    "eligible": ("member", "membership"), "eligibility": ("member", "membership"), "join": ("member", "membership"),
    "student": ("college", "university"), "college": ("student", "university"), "university": ("student", "college"),
    "cost": ("fee", "price"), "overdraw": ("overdraft",), "overdrawn": ("overdraft",), "charge": ("fee",),
    "mailing": ("box",), "become": ("join",), "qualify": ("qualifying", "qualified"), "open": ("opening", "opened"),
    "stop": ("close", "closing", "closure", "last"), "serving": ("service",),
    "oversea": ("international", "foreign"), "expedited": ("rush",), "expedite": ("rush",), "telephone": ("phone",),
    "send": ("transfer", "outgoing", "domestic"), "sending": ("transfer", "outgoing", "domestic"), "sent": ("transfer", "outgoing", "domestic"), "receive": ("incoming",), "receiving": ("incoming",),
    "lose": ("lost",), "losing": ("lost",), "hit": ("deposit", "paid", "posted"), "cap": ("limit", "below", "maximum"),
    "outgoing": ("domestic",), "intro": ("introductory", "introduction"), "country": ("international", "foreign"), "shut": ("close", "closing", "closure"), "turn": ("activate", "enable"), "outside": ("non", "network"), "location": ("branch",), "first": ("1st",), "1st": ("first",), "yield": ("apy", "dividend", "rate"), "introductory": ("intro",), "promo": ("promotion", "offer", "reward", "earn"),
    "bonus": ("promo", "offer", "special"), "reward": ("promo", "offer"), "earn": ("promo", "offer", "get"), "wallet": ("payment", "pay"),
    "office": ("branch",), "promotion": ("special",), "thru": ("up",), "refi": ("refinance",),
    "transit": ("routing",), "share": ("certificate",),  # share: coverage only (ALIAS_NO_SEARCH)
}
ALIAS_NO_SEARCH = frozenset({"share", "become", "qualify", "open", "serving", "first", "1st"})  # satisfies a coverage check but must not pull in other products
# Words that say which value to pick, not what the topic is; no block contains "highest".
SUPERLATIVES = frozenset("best highest higher largest lowest lower maximum minimum most least top".split())
HIGHEST = frozenset("best highest higher largest maximum top".split())
LOWEST = frozenset("lowest lower smallest least cheapest".split())
# Words that ask for a link or for every item, not topics.
LINK_WORDS = frozenset({"link", "url", "website", "site"})
LIST_WORDS = frozenset({"all", "every", "each", "list", "offer", "option", "type", "kind"})
# Plural nouns that do not mean "several items" ("savings" is a product name).
PLURAL_EXEMPT = frozenset({"hour", "month", "year", "day", "week", "saving", "apply", "reward", "rapid", "fall"})  # rapid, fall: Cedar Rapids, Cedar Falls
_MONEY_OR_PERCENT = re.compile(f"%|{MONEY_MARK.pattern}")
_EXISTENCE = re.compile(r"^\s*(?:does|do)\s+(?:dupaco|you|the credit union)\s+(?:have|offer|provide)\b", re.I)
_CODE = re.compile(r"[a-z]{3,}\d{2,}$")
_COMPARISON = re.compile(r"\b(?:vs\.?|versus|compared? (?:to|with)|difference between)\b", re.I)  # two items asked about side by side
_HOW_MUCH = re.compile(r"\bhow much\b|\bdollars?\b|\bminimum\s+(?:\w+\s+)?(?:deposit|balance|amount|to)\b|\bneed to (?:deposit|pay|put|spend)\b", re.I)
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
_ADDRESS_SHAPE = re.compile(r"\b\d{5}(?:-\d{4})?\b|@|\bP\.?\s?O\.? Box\b|\b\d{1,5}\s+(?:[A-Za-z.]+\s+){1,3}(?:St|Ave|Rd|Blvd|Hwy|Pkwy|Dr|Street|Avenue|Road|Boulevard|Highway|Drive)\b", re.I)
# "What date did it close?" asks for a calendar date, not for a block that says "date".
_DATE_SHAPE = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
PHONE_FACTOR = 1.5  # a phone question prefers the line that holds a number to dial
VALUE_SHAPES = {**dict.fromkeys(MONEY_WORDS, _MONEY_OR_PERCENT), "price": _PRICE_SHAPE, "hour": _CLOCK_TIME, "address": _ADDRESS_SHAPE,
                "date": _DATE_SHAPE}


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


def anchor_word(index: Index, required: list[str]) -> str:
    """The question's rarest topic word (the thing it names), or ''."""
    words = [t for t in required if not any(c.isdigit() for c in t) and t not in FRAMING_WORDS]  # amounts/terms are matched by qualifiers
    return max(sorted(set(words)), key=index.idf) if words else ""


def _unasked_tiers(q: Question, required: list[str], block: Block) -> frozenset[str]:
    """Tier words the row carries that the question does not ask for (an IRA's Roth/Traditional rows are the
    IRA itself when the question names the IRA)."""
    unasked = tier_words_in(block.text) - q.info.tiers
    return unasked - {"roth", "traditional"} if "ira" in required else unasked


def _names_product(required: list[str], block: Block) -> bool:
    """The question uses two or more of the words that name the row's product ("share savings")."""
    named = set(content_tokens(f"{block.caption} {block.label}"))
    return len(named & set(required)) >= 2


def has_anchor(index: Index, required: list[str], block: Block) -> bool:
    """The anchor word must itself appear in the block; otherwise a block about something else
    can pass on the common words alone."""
    anchor = anchor_word(index, required)
    return not anchor or is_present(index, anchor, block)


def aliases_of(index: Index, token: str) -> tuple[str, ...]:
    """Fixed aliases, else the words an acronym the site defines stands for (HSA, HELOC)."""
    return ALIASES.get(token) or index.acronyms.get(token) or tuple(f for f in _inflections(token) if index.idf(f) >= INFLECTION_MIN_IDF) or _parts_of(index, token)


def _parts_of(index: Index, token: str) -> tuple[str, ...]:
    """The rarer half of a run-together name the site spells as two words ("homeready" -> "ready",
    "hyvee" -> "vee" when the site writes "Hy-Vee"). Found once per word: it scans the whole site."""
    found = index.cached("parts_of", dict)
    if token not in found:
        found[token] = _split_name(index, token)
    return found[token]


def _split_name(index: Index, token: str) -> tuple[str, ...]:
    if len(token) < 5 or not token.isalpha() or index.df(token) > 3:
        return ()
    halves = [(token[:i], token[i:]) for i in range(2, len(token) - 1)]
    found = [max(pair, key=index.idf) for pair in halves
             if all(index.df(p) > 0 for p in pair) and (len(token) >= 8 or _written_apart(index, *pair))]
    return tuple(dict.fromkeys(found[:1]))


def _written_apart(index: Index, first: str, second: str) -> bool:
    """The site writes the two words joined by a hyphen or space next to each other ("Hy-Vee")."""
    together = set(index.blocks_with(first)).intersection(index.blocks_with(second))  # only these can hold both words
    pattern = re.compile(rf"\b{first}[- ]{second}\b")
    return any(pattern.search(index.blocks[doc_id].text.lower()) for doc_id in sorted(together))


@lru_cache(maxsize=1 << 12)
def _inflections(token: str) -> tuple[str, ...]:
    """The base forms of a word ending -ing, -ed or -ers ("referring" -> "refer"): a question may
    use another form of the word the site uses."""
    for suffix in ("ing", "ers", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3 and token.isalpha():
            base = token[:-len(suffix)]
            forms = [base, base + "e"]
            if base[-1] == base[-2]:
                forms.append(base[:-1])  # "referring" -> "referr" -> "refer"
            return tuple(dict.fromkeys(forms))
    return ()


def with_aliases(index: Index, query: list[str]) -> list[str]:
    return query + [alias for t in query if t not in ALIAS_NO_SEARCH for alias in aliases_of(index, t)]


_CAPITALISED = re.compile(r"(?<![.?!]\s)(?<!^)\b[A-Z][a-z]{3,}\b")
_NOT_ENTITIES = frozenset("monday tuesday wednesday thursday friday saturday sunday dupaco".split())
ENTITY_MIN_IDF = 6.0  # a capitalised word that is rare on the site names a place or a program


LOWERCASE_ENTITY_MIN_IDF = 5.5  # a place typed in lower case is rarer than a common word but not as rare as a program
NAME_MIN_CAPITALISED = 3
NAME_CAPITALISED_SHARE = 0.85
_LOWER_WORD = re.compile(r"\b[a-z]{4,}\b")
_PLACE_LEAD = frozenset({"in", "at", "near", "from"})
_PLACE_NOUNS = frozenset({"branch", "branches", "lab", "office", "location", "atm", "campus", "store"})
_STREET_WORDS = frozenset(STREET_ABBREVIATIONS.values()) | frozenset(STREET_ABBREVIATIONS)


def _name_tokens(index: Index) -> frozenset[str]:
    """Words the site almost always writes with a capital letter in mid-sentence: names of places and programs."""
    def compute() -> frozenset[str]:
        capitalised: Counter = Counter()
        lower: Counter = Counter()
        for block in index.blocks:
            if block.is_row:
                continue
            capitalised.update(stem(w.lower()) for w in _CAPITALISED.findall(block.text))
            lower.update(stem(w) for w in _LOWER_WORD.findall(block.text))
        return frozenset(t for t, n in capitalised.items()
                         if n >= NAME_MIN_CAPITALISED and n >= NAME_CAPITALISED_SHARE * (n + lower.get(t, 0)))
    return index.cached("name_tokens", compute)


def _entities(index: Index, raw: str) -> frozenset[str]:
    """Rare capitalised words after the first ("the Platteville branch"): they name the place or
    program asked about and must appear in an answer's passage or page title."""
    found = set()
    for word in _CAPITALISED.findall(raw):
        token = stem(word.lower())
        if token not in _NOT_ENTITIES and word not in MONTHS and index.idf(token) >= ENTITY_MIN_IDF and index.df(token) > 0:
            found.add(token)
    return frozenset(found)


def _lowercase_entities(index: Index, raw: str, skip: frozenset[str]) -> frozenset[str]:
    """Words the site always writes as a name, typed in lower case beside another name or a place word
    ("cedar rapids", "the platteville branch"): they must appear somewhere on the answer's page."""
    found = set()
    words = re.findall(r"[a-z]+", raw.lower())
    names = None
    for i, word in enumerate(words):
        token = stem(word)
        if (len(word) < 4 or token in found or token in skip or token in _NOT_ENTITIES or token in _STREET_WORDS
                or index.idf(token) < LOWERCASE_ENTITY_MIN_IDF or index.df(token) == 0):
            continue
        names = _name_tokens(index) if names is None else names
        if token not in names:
            continue
        before = words[i - 1] if i else ""
        after = words[i + 1] if i + 1 < len(words) else ""
        # a name typed in lower case sits beside another name or a place word ("the platteville branch", "in madison")
        if before in _PLACE_LEAD or after in _PLACE_NOUNS or stem(after) in names or stem(before) in names:
            found.add(token)
    return frozenset(found)


def _page_words(index: Index, url: str) -> frozenset[str]:
    """Every word of a page's passages (and their page context)."""
    pages = index.cached("page_words", dict)
    if url not in pages:
        prose = index.cached("prose_by_url", lambda: _group_by_url(b for b in index.blocks if not b.is_row))
        pages[url] = frozenset().union(*(block_words(b) for b in prose.get(url, ())))
    return pages[url]


def _shows_rate(block: Block) -> bool:
    return bool(_MONEY_OR_PERCENT.search(block.text)) and "%" in block.text and (block.is_row or bool(_RATE_NEAR_PERCENT.search(block.text)))


_FEE_TALK = re.compile(r"\b(?:fees?|charges?|charged|costs?|prices?|penalt\w+|waive\w*|free|no charge)\b", re.I)


def has_asked_value(query: list[str], block: Block) -> bool:
    if ({"fee", "cost", "charge"} & set(query)) and not ({"apr", "apy", "interest"} & set(query)) and not _FEE_TALK.search(block.text):
        return False  # a table of balance tiers has dollar amounts, but it does not state a fee
    asked = [word for word in VALUE_SHAPES if word in query]
    other = [word for word in asked if word not in RATE_WORDS]
    if any(not VALUE_SHAPES[word].search(block.text) for word in other):
        return False
    if other and any(word in RATE_WORDS for word in asked):
        return True  # "the monthly fee on 1st Rate Checking": "rate" is part of a name, the fee is the value asked
    return all(_shows_rate(block) for word in asked if word in RATE_WORDS)


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
    names: tuple[str, ...] = ()  # run-together names of the site the question spells as words ("make the most")


_DATE_AS_REFERENCE = re.compile(r"\bas of (?:this |that |today'?s? |the )?date\b|\b(?:today'?s|this|that|current|effective) date\b", re.I)


def _question(question: str) -> Question:
    text = clean(question)
    query = with_implied_words(text, content_tokens(text))
    if _DATE_AS_REFERENCE.search(question):
        query = [t for t in query if t != "date"]  # "as of this date" names the moment, it does not ask for a date
    asks_money = bool(MONEY_WORDS.intersection(query))
    wants_extreme = bool(RATE_WORDS.intersection(query)) and bool((HIGHEST | LOWEST).intersection(query))
    return Question(question, text, query, parse_question(text), intent_of(question, asks_money), asks_money,
                    wants_list(text, query) or wants_extreme or bool(_COMPARISON.search(question)))  # the extreme is picked from the rates of several products


FRAMING_PROBE = 20  # best hits that must show a question word for it to count as part of the topic
# Verbs and adverbs that only frame an ask ("what does it carry", "ways to avoid", "back then").
FRAMING_WORDS = frozenset("""avoid carry cover load reply require describe recommend offered offer bring keep make come
    brand then back past far late outside long directly people article limit
    range dodge mail turn down otherwise exactly basically usually typically""".split())
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

    hits = index.search(with_aliases(index, list(q.names) or q.query), CANDIDATES, rows_only)
    required = list(q.names) if q.names else _without_framing_words(index, required, hits)  # a named page: the name decides
    anchor = anchor_word(index, required)
    entities = _entities(index, q.raw)
    place_names = _lowercase_entities(index, q.raw, entities)
    wants_phone = {"phone", "number"} <= set(q.query) or "telephone" in q.query
    codes = [t for t in required if _CODE.match(t)]  # "CHECK500": a code named in the question must be in the block
    existence = bool(_EXISTENCE.search(q.raw))
    wants_fee = bool({"fee", "cost", "charge"} & set(q.query))
    topic_required = [t for t in required if t != "dupaco"]
    scored = []
    for score, doc_id in hits:
        block = index.blocks[doc_id]
        factor = q.info.fit(block, ignore_tiers=q.several)
        if q.asks_money and page_type(block.url) == "calculator":
            continue  # a calculator's example rows are not the credit union's rates
        confirmed = q.info.confirmations(block)
        if confirmed and (anchor and is_present(index, anchor, block) or q.info.confirmations(block, tiers=False) and _names_product(required, block)):
            confirmed += 1  # the row is also about the product the question names
        if confirmed >= CONFIRMED_DIMENSIONS and _unasked_tiers(q, required, block):
            confirmed = 0  # a row of another tier (an IRA's special for a certificate question) does not stand in for the asked one
        if confirmed >= CONFIRMED_DIMENSIONS:  # the row states the asked term/year/amount/tier: the question's other words may differ from the page's
            factor *= CONFIRMED_FACTOR ** confirmed
            if not factor or coverage(index, required, block) < CONFIRMED_COVERAGE:
                continue
        elif not factor or coverage(index, required, block) < MIN_COVERAGE or not has_anchor(index, required, block):
            continue
        if not (block.links if wants_link else has_asked_value(q.query, block)):
            continue
        if wants_phone and phone_in(block.text):
            factor *= PHONE_FACTOR  # "phone number of ...": the line that holds a number to dial
        if codes and not set(codes) <= block_words(block):
            continue
        if q.names and not set(q.names) <= block_words(block):
            continue
        if q.asks_money and lacks_named_product(index, required, block):
            continue  # "HELOC intro rate": the card's introductory rate is another product's
        if lacks_asked_tier(q, block):
            continue  # "business line of credit": a rate of the personal product is another product's rate
        if entities and not entities <= block_words(block):
            continue  # "Platteville": a passage about another place does not answer
        if place_names and not place_names <= block_words(block) and not place_names <= _page_words(index, block.url):
            continue  # "cedar rapids" typed in lower case: the page must be about that place
        if existence and (coverage(index, topic_required, block) < 1.0):
            continue  # "does Dupaco have X": a block that lacks part of X does not show Dupaco has it
        if wants_amount and not _MONEY_OR_PERCENT.search(block.text):
            if q.asks_money or q.intent == "membership":
                continue  # "how much is the fee", "minimum to join" ask for an amount
            factor *= NO_AMOUNT_FACTOR  # "how much do I need to join": the passage with an amount is likelier
        if wants_amount and _OFFER_WORDS.search(block.text) and not _OFFER_WORDS.search(q.raw):
            factor *= UNASKED_OFFER_FACTOR  # "how much to join": the membership deposit, not the referral bonus
        if wants_fee and _PRICE_SHAPE.search(block.text):
            factor *= FEE_STATED_FACTOR  # the fee and its amount in the same breath
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
    if rows and not _fee_in_prose(index, q, rows):
        return _drop_superseded(index, rows)
    prose = _qualifying(index, q, rows_only=False)  # a rate stated in a sentence
    # Blog articles and press releases discuss rates in general or years ago; they do not state this
    # credit union's current rate.
    prose = [(s, d) for s, d in prose if not is_article(index.blocks[d].url)]
    if q.info.years:  # a rate for a model year must come from text that names a range covering it
        prose = [(s, d) for s, d in prose if any(lo <= y <= hi for y in q.info.years
                                                 for lo, hi in year_spans_in(index.blocks[d].text))]
    return prose


FEE_PROSE_MARGIN = 3.0  # a sentence this much better than the best row states the fee itself


def _fee_in_prose(index: Index, q: Question, rows: list[tuple[float, int]]) -> bool:
    """A fee ("avoid the monthly fee on checking") is often stated in a sentence while the only
    matching row is a rate table; a clearly better-matching sentence on a product page wins."""
    if not ({"fee", "cost", "charge"} & set(q.query)) or {"apr", "apy", "interest"} & set(q.query):
        return False
    best_row = rows[0][0]
    return any(score >= best_row * FEE_PROSE_MARGIN and not is_article(index.blocks[d].url)
               for score, d in _qualifying(index, q, rows_only=False)[:5])


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
    if kind.name == "insurance":  # one fixed fact stated many ways: the most direct statement, whatever the wording
        return [min(found, key=lambda sb: (len(sb[1].text), sb[1].url))[1]]
    return [max(found, key=lambda sb: sb[0])[1]]


# Wordings that ask for a rate or for opening hours without saying so.
_PAYS = re.compile(r"\b(?:how much|what|whats)\b[^?]*\b(?:pay|pays|paying|earn|earns|earning|yield|dividends?)\b|\bwhat do(?:es)? [^?]*\b(?:pay|earn)\b", re.I)
_PAY_VERBS = frozenset({"pay", "paying", "earn", "earning"})
_NOWADAYS = re.compile(r"\b(?:these|those) days\b", re.I)
_I_PAY = re.compile(r"\b(?:i|we) (?:have to |need to |will |would )?pay\b|\bpay(?:ing)? to\b|\b(?:do|will|would) (?:i|we|you) pay(?:ing)? for\b", re.I)  # what the member pays is a cost, not a rate; "what does the IRA pay for $8,000" is a rate
_OFFER_WORDS = re.compile(r"\b(?:cash ?back|promo\w*|bonus|offers?|codes?|rewards?|referral|refer\w*)\b", re.I)  # earning an offer is not a rate
_STATED_PERCENT = re.compile(r"\d(?:\.\d+)?\s*%")  # "only 6%?", "paying 4.5%": the question is about a rate


_BONUS_ROOTS = frozenset({"refer", "bonus", "reward", "promo", "promotion", "gift", "cashback"})


def with_implied_words(text: str, query: list[str]) -> list[str]:
    """Add the word a wording implies: "what does it pay?" or a stated percent asks for a rate,
    "when is it open?" for hours."""
    bonus = any(root(t) in _BONUS_ROOTS for t in query)  # "how much can I earn by referring a friend" asks for an amount
    pays = _PAYS.search(text) and not _OFFER_WORDS.search(text) and not _I_PAY.search(text)
    extra = (["rate"] if (pays or _STATED_PERCENT.search(text)) and not MONEY_WORDS.intersection(query) and not bonus else [])
    extra += (["hour"] if OPENING.search(text) and "hour" not in query else [])
    if "rate" in extra and pays:  # "what is Bright Savings paying": the verb asked for the rate, it is not a topic word
        query = [t for t in query if t not in _PAY_VERBS]
    if _NOWADAYS.search(text):
        query = [t for t in query if t not in ("these", "those", "day")]
    return query + extra


def lacks_asked_tier(q: Question, block: Block) -> bool:
    """A rate question names a product tier (business, jumbo, domestic...) that the block never mentions."""
    return q.asks_money and bool(q.info.tiers) and not q.info.tiers & (tier_words_in(block.text) | tier_words_in(block.context))


PRODUCT_WORD_IDF = 5.0  # as rare as a product name on the site...
PRODUCT_WORD_DF = 20  # ...but used on enough blocks to be a real word, not a typo or a synonym the site avoids


def _product_words(index: Index) -> frozenset[str]:
    """Words of the names of priced rows."""
    return index.cached("product_words", lambda: frozenset(
        t for b in index.blocks if b.is_row and (b.figures or _MONEY_OR_PERCENT.search(b.text)) and not b.url.endswith((".xls", ".xlsx"))
        for t in content_tokens(f"{b.label} {b.caption}")))


def warm(index: Index) -> None:
    """Build the per-index tables every question would otherwise build on its first use, so that
    no answer pays for them."""
    _name_tokens(index)
    _product_words(index)
    index.cached("heading_copies", lambda: _short_blocks(index))
    index.cached("teaser_sources", lambda: _opening_index(index))


def lacks_named_product(index: Index, required: list[str], block: Block) -> bool:
    """A rate question names its product by words that are specific on the site (HELOC, wire, Bright):
    every one of them must be in the block, or the block is another product's rate."""
    names = _product_words(index)
    return any(t in names and t not in TIER_WORDS and index.idf(t) >= PRODUCT_WORD_IDF and index.df(t) >= PRODUCT_WORD_DF and not is_present(index, t, block)
               for t in sorted(set(required)) if not any(c.isdigit() for c in t))


_PLACE_BEFORE = re.compile(r"\b([a-z]+)(?:\s+([a-z]+))?\s+(?:branch|office|location|atm)\b")
_NOT_PLACES = frozenset("nearest closest nearby local main downtown first other biggest largest busiest newest oldest which another walk drive".split())


def names_unknown_place(index: Index, raw: str) -> bool:
    """"The Omaha branch": a place word the site never uses, anywhere, means there is no such branch."""
    for match in _PLACE_BEFORE.finditer(raw.lower()):
        for word in match.groups():
            if (word and len(word) >= 5 and word not in _NOT_PLACES and not word.endswith(("est", "ing", "ly", "er", "al"))
                    and word not in STOPWORDS and not index.knows(stem(word)) and not _parts_of(index, word)):
                return True
    return False


def _compound_names(index: Index, text: str) -> list[str]:
    """Adjacent question words that the site writes as one word ("make the most" on a page named
    makethemost): the joined word is what the page's address or title carries."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    found = []
    for size in (3, 2):
        for i in range(len(words) - size + 1):
            joined = "".join(words[i:i + size])
            if len(joined) >= 8 and index.knows(joined) and index.df(joined) == 0 and joined not in found:
                found.append(joined)
    return found


def select_blocks(index: Index, question: str) -> list[Block]:
    """The best block plus its siblings (same page, date and kind) that belong in the answer."""
    if asks_schedules(question):
        schedules = distinct_schedules(index, question)
        if schedules:
            return list({block.text: block for block in schedules}.values())
    if names_unknown_place(index, question):
        return []
    q = _question(question)
    compounds = _compound_names(index, q.text) if _OFFER_WORDS.search(question) else []  # a named promotion; other names are just words
    if compounds:
        q = q._replace(query=q.query + compounds, names=tuple(compounds))
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


def _body_after(blocks: list[Block], i: int, continues: bool = True, multipart: bool = False) -> list[Block]:
    """The paragraph right after block `i` on its page, and a short line after that (a date, a label).
    A heading that already holds a figure (an address line) only takes continuation lines under the
    same label."""
    body = []
    heading = blocks[i]
    has_figure = any(c.isdigit() for c in heading.text) and not (multipart and not _STREET_NUMBER.search(heading.text))  # a two-part ask needs more than the headline's figure
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


_PROCESS = re.compile(r"\bwhat (?:do|will|would) i (?:get|have to|need)|\bhow (?:do|does|can|to)\b|\bsteps?\b|\brequirements?\b|\bqualify\b|\bwork\b", re.I)
_OFFER_PAGE = re.compile(r"-lp$|offer$", re.I)
STEPS_MAX_BLOCKS = 8
STEPS_MAX_CHARS = 400


def with_steps(index: Index, blocks: list[Block], question: str) -> list[Block]:
    """An offer on a landing page spells out its steps in the passages after the one that names the
    code ("Add Money", "Get Paid"): a "what do I have to do" question gets them too."""
    first = blocks[0]
    if (first.is_row or not _PROCESS.search(question) or not _OFFER_PAGE.search(first.url.split("?")[0].rstrip("/").rsplit("/", 1)[-1])
            or not _OFFER_WORDS.search(question)):
        return blocks
    i = next((n for n, b in enumerate(index.blocks) if b is first), None)
    if i is None:
        return blocks
    steps = []
    for follower in index.blocks[i + 1:i + 1 + STEPS_MAX_BLOCKS]:
        if follower.url != first.url or follower.is_row or len(follower.text) > STEPS_MAX_CHARS:
            break
        steps.append(follower)
    return blocks + [b for b in steps if b not in blocks]



_STREET_NUMBER = re.compile(r"\b\d{1,5}\s+[A-Z][A-Za-z.]+")  # "2050 JFK Rd.": an address line already is the answer
_MULTIPART = re.compile(r"\band\s+(?:what|how|when|which|who|where)\b", re.I)  # "how much and what loan size"
_ADDRESS_WORDS = frozenset({"address", "mailing", "location", "located", "where"})
_PHONE_WORDS = frozenset({"phone", "call", "number", "contact", "text", "fax"})


def with_body(index: Index, blocks: list[Block], question: str = "") -> list[Block]:
    """A block that is only a heading ("Application period: ..." / a headline with no sentence) is
    followed by the paragraphs under it. A headline repeated on other pages (as a teaser) is read
    where its body is longest, which is the article itself."""
    copies = index.cached("heading_copies", lambda: _short_blocks(index))
    asked = _question(question).query if question else []
    continues = not question or bool(_ADDRESS_WORDS.intersection(asked))
    multipart = bool(_MULTIPART.search(question)) and not continues  # an address question already takes its labelled lines  # only an address spans several labelled lines
    result = []
    for block in blocks:
        if (block.is_row or len(block.text) > 100 or block.text.rstrip().endswith((".", "!", "?"))
                or any(other is not block and other.url == block.url for other in blocks)):  # the page's other passages are chosen already
            result.append(block)
            continue
        best = max(copies[block.text], key=lambda i: (sum(len(b.text) for b in _body_after(index.blocks, i)), -i))
        result.append(index.blocks[best])
        asks_phone = not question or bool(_PHONE_WORDS.intersection(asked))
        result.extend(b for b in _body_after(index.blocks, best, continues, multipart) if b not in blocks and (asks_phone or not phone_in(b.text)))
    return result


_FAQ_QUESTION = re.compile(r"^(?:[^:\n]{1,80}:{1,2} )?- [^\n]{3,200}\?$")
ANSWER_MAX_BLOCKS = 6  # a list of requirements runs over several lines
ANSWER_MAX_CHARS = 800


def _positions(index: Index) -> dict[int, int]:
    return index.cached("positions", lambda: {id(b): i for i, b in enumerate(index.blocks)})


def with_answer(index: Index, blocks: list[Block]) -> list[Block]:
    """A frequently-asked question printed as a line of its own ("- How do I ...?") is followed by
    its answer: the lines under it with its label, up to the next question."""
    positions = _positions(index)
    answers: dict[int, list[Block]] = {}
    for block in blocks:
        if block.is_row or not _FAQ_QUESTION.match(block.text) or id(block) not in positions:
            continue
        taken, size = [], 0
        label = _label(block.text) if block.text.find(": - ") > 0 else None  # the answer lines carry the question's label
        for follower in index.blocks[positions[id(block)] + 1:positions[id(block)] + 1 + ANSWER_MAX_BLOCKS]:
            if label is None:
                label = _label(follower.text)
            if (follower.url != block.url or follower.is_row or _FAQ_QUESTION.match(follower.text) or size + len(follower.text) > ANSWER_MAX_CHARS
                    or _label(follower.text) != label):
                break
            taken.append(follower)
            size += len(follower.text)
        answers[id(block)] = taken
    if not answers:
        return blocks
    placed = {id(b) for taken in answers.values() for b in taken}  # shown under their question, not where they were found
    result: list[Block] = []
    for block in blocks:
        if id(block) in placed:
            continue
        result.append(block)
        result.extend(b for b in answers.get(id(block), ()) if b not in result)
    return result


def complete_blocks(index: Index, blocks: list[Block], question: str, shapes: list[re.Pattern] = ()) -> list[Block]:
    """The passages to show: the chosen blocks, trimmed to the question, each with what belongs with it
    (the paragraph under a heading, the answer under a question, the rest of its section, the release
    behind a teaser, the phone line of a release, the steps of an offer)."""
    blocks = with_body(index, trim_blocks(index, blocks, question), question)
    blocks = with_full_text(index, with_section(index, blocks, question))
    blocks = with_value_line(index, blocks, question, shapes)
    blocks = with_steps(index, with_answer(index, with_contact(index, blocks, question)), question)
    return list({block.text: block for block in blocks}.values())  # a line found twice is shown once


VALUE_LINE_REACH = 8  # lines under a headline searched for the one that shows the asked value


def with_value_line(index: Index, blocks: list[Block], question: str, shapes: list[re.Pattern]) -> list[Block]:
    """A headline followed by its opening paragraphs may leave out the line that holds the asked kind
    of value (the date a branch closes comes a few lines down): the first following line that shows it
    and talks about the question joins."""
    if not shapes or not blocks:
        return blocks
    q = _question(question)
    skipped = q.info.consumed.union(VALUE_SHAPES, SUPERLATIVES, LINK_WORDS, LIST_WORDS)
    required = sorted({t for t in q.query if t not in skipped and t not in FRAMING_WORDS and not any(c.isdigit() for c in t)})
    positions = _positions(index)
    result = list(blocks)
    for block in blocks:
        if block.is_row or id(block) not in positions or any(shape.search(b.text) for b in blocks if b.url == block.url for shape in shapes):
            continue
        last = max(positions.get(id(b), -1) for b in blocks if b.url == block.url)
        for follower in index.blocks[last + 1:last + 1 + VALUE_LINE_REACH]:
            if follower.url != block.url or follower.is_row:
                break
            if any(shape.search(follower.text) for shape in shapes) and (not required or any(is_present(index, t, follower) for t in required)):
                result.insert(result.index(max((b for b in blocks if b.url == block.url), key=lambda b: positions.get(id(b), -1))) + 1, follower)
                break
    return result


TEASER_KEY_CHARS = 60
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")


def _opening_key(text: str) -> str:
    """The first letters and digits of a passage, whatever its punctuation or markup."""
    return _NOT_ALNUM.sub("", text.lower())[:TEASER_KEY_CHARS]


def with_full_text(index: Index, blocks: list[Block]) -> list[Block]:
    """A listing page quotes the start of each release and ends it with "...": the release itself
    (same opening, longer) is the passage to answer from."""
    if not any(b.text.endswith("...") for b in blocks):
        return blocks
    full = index.cached("teaser_sources", lambda: _opening_index(index))
    result = []
    for block in blocks:
        if block.text.endswith("...") and not block.is_row:
            longer = [b for b in full.get(_opening_key(block.text), ()) if len(b.text) > len(block.text)]
            if longer:
                block = max(longer, key=lambda b: (len(b.text), b.url))
        if block not in result:
            result.append(block)
    return result


def _opening_index(index: Index) -> dict[str, list[Block]]:
    found: dict[str, list[Block]] = {}
    for b in index.blocks:
        if not b.is_row and len(b.text) > TEASER_KEY_CHARS and is_article(b.url):
            found.setdefault(_opening_key(b.text), []).append(b)
    return found


_CALL_WORDS = frozenset({"phone", "telephone", "call"})


def with_contact(index: Index, blocks: list[Block], question: str) -> list[Block]:
    """A news release names the branch's address in its opening and gives the phone number lines
    later; a "phone number" question answered by the opening gets the line that holds the number,
    from the same release."""
    q = _question(question)
    if not _CALL_WORDS.intersection(q.query) or any(phone_in(b.text) for b in blocks):
        return blocks
    skipped = q.info.consumed.union(VALUE_SHAPES, SUPERLATIVES, LINK_WORDS, LIST_WORDS, _CALL_WORDS, {"number", "contact"})
    required = sorted({t for t in q.query if t not in skipped and not any(c.isdigit() for c in t)})
    positions = _positions(index)
    result = list(blocks)
    for block in blocks:
        if block.is_row or id(block) not in positions or not is_article(block.url):
            continue
        start = positions[id(block)]
        # lines of the same release, nearest first; the one that holds a number and talks about the asked place
        candidates = [b for b in index.blocks[max(0, start - 60):start + 60]
                      if b.url == block.url and not b.is_row and len(b.text) <= 160 and phone_in(b.text)]
        best = max(candidates, key=lambda b: (coverage(index, required, b), -abs(positions[id(b)] - start)), default=None)
        if best is not None and best not in result and (not required or coverage(index, required, best) > 0):
            result.insert(result.index(block) + 1, best)
            break
    return result


SECTION_REACH = 3  # lines on either side of a chosen passage that can still belong to its section
SECTION_MAX_ADDED = 3
SECTION_STEPS = 2
SECTION_NEW_WORDS = 2  # a neighbouring line joins when it covers this many question words the chosen lines miss...
SECTION_RARE_IDF = 6.5  # ...or one word that is rare on the site
_STEPS_QUESTION = re.compile(r"\bwhat (?:do|should|can) i do\b|\bhow (?:do|can|should) i\b|\bsteps?\b", re.I)


def with_section(index: Index, blocks: list[Block], question: str) -> list[Block]:
    """An article prints a section as several lines under one "Label:"; the line that ranked first may
    hold only part of what was asked (the step, but not the phone number to call). Neighbouring lines
    of the same section are added when they cover a question word the chosen lines do not."""
    positions = _positions(index)
    q = _question(question)
    skipped = q.info.consumed.union(VALUE_SHAPES, SUPERLATIVES, LINK_WORDS, LIST_WORDS)
    required = sorted({t for t in q.query if t not in skipped and t not in FRAMING_WORDS and not any(c.isdigit() for c in t)})
    if not required:
        return blocks
    steps = bool(_STEPS_QUESTION.search(question))  # "what do I do": the next lines of the section are the steps
    result: list[Block] = []
    for block in blocks:
        group = [block]
        label = _label(block.text)
        if not block.is_row and label and len(label) <= 80 and id(block) in positions and not _FAQ_QUESTION.match(block.text):
            at = positions[id(block)]
            covered = {t for t in required if is_present(index, t, block)}
            for step in (1, -1):  # outwards through the lines of the same section, no gaps
                for pos in range(at + step, at + step * (SECTION_REACH + 1), step):
                    if not 0 <= pos < len(index.blocks):
                        break
                    other = index.blocks[pos]
                    if other.url != block.url or other.is_row or _label(other.text) != label or _FAQ_QUESTION.match(other.text):
                        break
                    if other in blocks or other in result or len(group) > SECTION_MAX_ADDED:
                        continue
                    new = {t for t in required if is_present(index, t, other)} - covered
                    if len(new) >= SECTION_NEW_WORDS or any(index.idf(t) >= SECTION_RARE_IDF for t in new) or (steps and step == 1 and pos - at <= SECTION_STEPS):
                        covered |= new
                        group.append(other)
        result.extend(sorted(group, key=lambda b: positions.get(id(b), 0)))
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
