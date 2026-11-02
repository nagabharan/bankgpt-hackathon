"""Text normalisation shared by ingest and question handling."""
from __future__ import annotations

import html
import re
from functools import lru_cache

# A leading (?=...) in the patterns below only restates the first character every match starts
# with; it changes no match but lets the engine skip most positions (clean() dominates ingest).
# Footnote markers are digits glued to a value or header: "4.00%1", "5.39% APR1", "(APY)1".
_FOOTNOTE = re.compile(r"(?=\d)(?:(?<=%)|(?<=APR)|(?<=APY)|(?<=\)))\d\b")
# ...or separated by a space at the end of a table cell: "15.50% 1 |", "(As Low As) 1 |".
_SPACED_FOOTNOTE = re.compile(r"(?<=[%)])\s\d(?=\s*\|)")
_LITERAL_NEWLINE = re.compile(r"\\+r?\\*n|\\+r")  # "\r\n" appearing as literal text in the crawl
_DASHES = re.compile(r"[‐-―−]")  # hyphen/en/em/minus variants
# Grouped integers: Western "1,234,567" and Indian "12,34,567" (final group is always 3 digits).
_GROUPED_NUMBER = re.compile(r"(?=\d)(?<![\d,])(?:\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})*,\d{3})\b")

MONTHS = "January February March April May June July August September October November December".split()

CURRENCY_SYMBOLS = "$€£¥₹₩₽₺₫₪₱฿"
CURRENCY_CODES = ("USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "INR", "MXN", "CNY")
# A currency symbol or ISO code: the mark of a money amount (the % sign marks a rate).
MONEY_MARK = re.compile(f"[{re.escape(CURRENCY_SYMBOLS)}]|\\b(?:{'|'.join(CURRENCY_CODES)})\\b")
_K_AMOUNT = re.compile(r"(?=[$\d])\$?\b(\d+(?:\.\d+)?)\s?[kK]\b")
_DOLLARS_WORD = re.compile(r"\b(\d+(?:\.\d+)?)\s+dollars?\b", re.I)
_SYMBOL_SPACE = re.compile(f"([{re.escape(CURRENCY_SYMBOLS)}])\\s+(?=\\d)")  # "€ 591.61" -> "€591.61"
# A footnote digit after a grouped amount: "$25,0002" is "$25,000" plus footnote 2.
_AMOUNT_FOOTNOTE = re.compile(f"([{re.escape(CURRENCY_SYMBOLS)}]\\d{{1,3}}(?:,\\d{{3}})+)\\d\\b")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]*)[^)]*\)")
LINK_SCHEMES = ("http", "tel:", "mailto:")
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

STOPWORDS = frozenset(
    "a an and are as at be by can do does for how i in is it me my of on or s tell "
    "that the there this to what when where which who with you your "
    # question filler that carries no topic
    "about any between correct current currently could did from general get give have has its keep know last like located m re ve ll d t going still new financed finance buy if into over under per bought purchased "
    "many may much need now please provide right same their them they today want was were would "
    "spring summer fall autumn winter park will signup "
    # contraction stubs ("isn't" -> "isn") and premise wording ("why is it only 4.1%?") carry no topic
    "isn aren doesn didn don wasn weren hasn haven why only really true actually".split()
)

# "six month", "sixty-month", "twenty four months" -> "6 month", "60 month", "24 months".
_UNIT_WORDS = {w: i for i, w in enumerate(
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split(), start=1)}
_TENS_WORDS = {w: 10 * i for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)}
_NUMBER_WORD = (rf"(?:(?:{'|'.join(_TENS_WORDS)})(?:[\s-]+(?:{'|'.join(list(_UNIT_WORDS)[:9])}))?"
                rf"|{'|'.join(_UNIT_WORDS)})")
_NUMBER_INITIALS = "".join(sorted({w[0] for w in (*_UNIT_WORDS, *_TENS_WORDS)}))
_WORD_TERM = re.compile(rf"\b(?=[{_NUMBER_INITIALS}])({_NUMBER_WORD})[\s-]*(month|mo|year|yr)s?\b", re.I)


def _word_value(words: str) -> int:
    return sum(_TENS_WORDS.get(w) or _UNIT_WORDS[w] for w in re.split(r"[\s-]+", words.lower()))


def clean(text: str) -> str:
    """Normalise a crawl fragment: drop footnote digits and literal newlines, unify dashes
    and number formats."""
    text = _LITERAL_NEWLINE.sub(" ", text)
    text = _WORD_TERM.sub(lambda m: f"{_word_value(m.group(1))} {m.group(2)}", text)
    text = html.unescape(html.unescape(text))  # the crawl double-escapes: "&amp;#45;"
    text = _DASHES.sub("-", text)
    text = _AMOUNT_FOOTNOTE.sub(r"\1", text)
    text = _FOOTNOTE.sub("", text)
    text = _SPACED_FOOTNOTE.sub("", text)
    text = _GROUPED_NUMBER.sub(lambda m: m.group().replace(",", ""), text)
    text = _K_AMOUNT.sub(lambda m: f"${round(float(m.group(1)) * 1000)}", text)  # "$30k" / "30K"
    text = _DOLLARS_WORD.sub(lambda m: f"${m.group(1)}", text)  # "50000 dollars"
    return _SYMBOL_SPACE.sub(r"\1", text)


def strip_links(text: str) -> tuple[str, list[str]]:
    """Replace markdown links by their label; return the text and the link targets
    (web, phone and mail). Images are dropped, including ones nested inside links."""
    text = _IMAGE.sub("", text)
    links = [m.group(2) for m in _MARKDOWN_LINK.finditer(text) if m.group(2).startswith(LINK_SCHEMES)]
    return _MARKDOWN_LINK.sub(lambda m: m.group(1), text), links


# Street words the site abbreviates ("3999 Pennsylvania Ave.", "Saratoga Rd."): one form for both spellings.
_STREET_ABBREVIATIONS = {"avenue": "ave", "road": "rd", "boulevard": "blvd", "highway": "hwy", "street": "st", "drive": "dr"}


@lru_cache(maxsize=1 << 16)  # millions of calls at ingest over a vocabulary of tens of thousands
def stem(token: str) -> str:
    """Crude plural folding so 'months' matches 'month' and 'branches' matches 'branch'."""
    if token in _STREET_ABBREVIATIONS:
        return _STREET_ABBREVIATIONS[token]
    if len(token) <= 3 or not token.endswith("s") or token.endswith(("ss", "us", "is")):
        return token
    if token.endswith(("ches", "shes", "xes", "sses")):
        return token[:-2]
    if token.endswith("ies"):
        return token[:-3] + "y"
    return token[:-1]


def tokens(text: str) -> list[str]:
    """Lower-cased, plural-folded search tokens (keeps decimals like '5.39')."""
    return [stem(t) for t in _TOKEN.findall(text.lower())]


def plural_nouns(text: str) -> set[str]:
    """Folded forms of the plural content words in `text` ("cards" -> "card", "rates" -> "rate")."""
    return {stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and stem(t) != t}


def content_tokens(text: str) -> list[str]:
    """Search tokens without stopwords (checked before folding, so 'does' is dropped, not 'doe')."""
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
