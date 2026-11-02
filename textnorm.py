"""Text normalisation shared by ingest and question handling."""
from __future__ import annotations

import calendar
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
    "just got another spring summer fall autumn winter park will signup brand buying offered carry carries regular then back looking interested wondering considering curious trying work works whats thats hows wanna gonna gotta lemme im ive id ill dont cant wont isnt vs versus using another once everything anything actually one not after paid "
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


# Chat filler of a question (dropped from questions only: these words also occur in page text).
_QUESTION_FILLER = re.compile(r"\b(?:but|still|also|put|newest|latest|newer|big|bracket|pls|plz|thx|thanks|hey|hi|hello|yo|okay|btw|lol|asap|anyway|honestly|basically|kinda|sorta|ish|roughly|approx|approximately|around(?!\s+(?:"+"|".join(MONTHS)+r"))|throw|toss|stick|sits|sit|ahead|better|kid|kids)\b[,.!]?\s*", re.I)

# Casual wording of a question, folded to the words the site uses. Applied to questions only.
_CASUAL_WORDS = (
    (re.compile(r"\bcerts?\b", re.I), "certificate"),
    (re.compile(r"\bmos?\b", re.I), "months"),
    (re.compile(r"\byrs?\b", re.I), "years"),
    (re.compile(r"\bu\b", re.I), "you"),
    (re.compile(r"\bcds?\b(?!\s*[/)(,]*\s*(?:or\s+)?certificates?\b)(?<!certificate\s\()(?<!certificate\s)", re.I), "certificate"),  # "CD / certificate" already says it
    (re.compile(r"\bmax\b\.?", re.I), "maximum"),
    (re.compile(r"\b(?:harley(?:[- ]davidson)?|yamaha|kawasaki|ducati|triumph|vespa|scooter|moped|dirt ?bike|sportbike|cruiser|chopper)s?\b", re.I), "motorcycle"),
    (re.compile(r"\b(?:civic|accord|camry|corolla|prius|altima|sentra|mustang|corvette|wrangler|jeep|tesla|f-?150|silverado|tacoma|tundra|ram|rav4|cr-?v|explorer|equinox|malibu|impala|fusion|focus|elantra|sonata|outback|forester|mazda|subaru|toyota|honda|ford|chevy|chevrolet|dodge|nissan|hyundai|kia|bmw|audi|lexus|volkswagen|vw|buick|cadillac|gmc)s?\b", re.I), "car"),
    (re.compile(r"\bacct\b", re.I), "account"),
    (re.compile(r"\bmin\b\.?", re.I), "minimum"),
    (re.compile(r"\b(?:suv|pickup|pick-up|truck|sedan|minivan|van|crossover|coupe|hatchback|wagon)s?\b", re.I), "car"),
    (re.compile(r"\b(\d+)(?=(?:yr|mo)s?\b)", re.I), r"\1 "),  # "30yr" -> "30 yr"
)
# "(up to 832,750)", "for 10000": a bare number after an amount word is dollars, unless it reads as a year
# or a term ("for 60 months").
_BARE_AMOUNT = re.compile(r"\b(up to|under|below|over|above|less than|more than|at least|at most|worth|for|of|with|about|have|has|had|deposit|borrow)\s+"
                          r"(?<![$\d])(\d{1,3}(?:,\d{3})+|\d{4,})\b(?!\s*(?:%|-?\s*(?:month|mo\b|year|yr)))", re.I)


def _dollars(match: re.Match) -> str:
    digits = match.group(2)
    if len(digits) == 4 and 1900 <= int(digits) <= 2100:
        return match.group(0)  # a year
    return f"{match.group(1)} ${digits}"


_GRAND = re.compile(r"\$?\b(\d+(?:\.\d+)?)\s*(?:grand|k)\b", re.I)  # "35 grand", "5k"
# "$5000 to $25000" / "5000-25000 balance": the tier that starts at the lower amount answers it.
_AMOUNT_RANGE = re.compile(r"\$(\d+(?:\.\d+)?)\s*(?:to|-|and|through|thru)\s*\$?\d+(?:\.\d+)?")
_YEAR_VEHICLE = re.compile(r"\b((?:19|20)\d\d)\s+(?:model\s+)?(?=(?:car|auto|vehicle)\b)", re.I)
_IRA_CERTIFICATE = re.compile(r"\b(ira|roth|traditional)\s+(?:cds?|certificates?)\b", re.I)  # IRA rows are named by tier and term only
_YEAR_MODEL = re.compile(r"\b((?:19|20)\d\d)\s+model\b(?!\s+year)", re.I)


_MONTH_ABBREVIATIONS = {m[:3].lower(): m for m in MONTHS if m != "May"} | {"sept": "September", "may": "May"}
_MONTH_LOOSE = re.compile(r"\b(" + "|".join(sorted(_MONTH_ABBREVIATIONS, key=len, reverse=True)) + r")(?:ust|uary|ember|ober|ch|il|e|y)?\b\.?", re.I)
_LEADS_MONTH = re.compile(r"(?:\b(?:in|since|from|to|and|between|of|during|until|through|by|after|before|vs|versus|than|with|on)\s+(?:the\s+)?(?:(?:early|mid|late|end of|start of|beginning of)\s+)?)$", re.I)


def _month_names(text: str) -> str:
    """"aug 27", "in june", "since may": the capitalised month names the date readers know. A bare
    "may" is a month only after a date preposition or before a day."""
    def fold(match: re.Match) -> str:
        word = match.group(1).lower()
        name = _MONTH_ABBREVIATIONS[word]
        after = text[match.end():match.end() + 6]
        if word == "may" and not (_LEADS_MONTH.search(text[:match.start()]) or re.match(r"\s*\d", after)):
            return match.group(0)
        if word != "may" and len(match.group(0).rstrip(".")) < 4 and not (_LEADS_MONTH.search(text[:match.start()]) or re.match(r"\s*\d", after)):
            return match.group(0)  # "mar", "jan", "dec" in running text
        return name
    return _MONTH_LOOSE.sub(fold, text)


def _month_period(text: str) -> str:
    """"early june" -> "June 15", "late august" / "end of august" -> "August 31": the day the snapshot
    must precede. Applied after month names are capitalised."""
    def fold(match: re.Match) -> str:
        month = MONTHS.index(match.group(2)) + 1
        lead = match.group(1).lower()
        day = 15 if lead in ("early", "mid") else 1 if lead in ("start of", "beginning of") else calendar.monthrange(2026, month)[1]
        suffix = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return f"{match.group(2)} {day}{suffix}"  # an ordinal cannot be read as a term ("June 17-month")
    return re.sub(r"\b(early|mid|late|end of|the end of|start of|beginning of)\s+(" + "|".join(MONTHS) + r")\b(?!\s+\d)", fold, text, flags=re.I)


def normalize_question(question: str) -> str:
    """Fold casual wording ("cert", "mos", "35 grand", "2018 pickup") into the words and numbers the
    site uses, so a chatty question reaches the same rows as a formal one."""
    text = _month_period(_month_names(question))
    text = re.sub(r"\brekeys?\b", "re-key", text, flags=re.I)
    text = _IRA_CERTIFICATE.sub(r"\1", text)
    text = _YEAR_MODEL.sub(r"\1 model year", text)
    for pattern, replacement in _CASUAL_WORDS:
        text = pattern.sub(replacement, text)
    text = _BARE_AMOUNT.sub(_dollars, text)
    text = _GRAND.sub(lambda m: f"${round(float(m.group(1)) * 1000)}", text)
    text = _AMOUNT_RANGE.sub(r"$\1", text)
    text = _YEAR_VEHICLE.sub(r"\1 model year ", text)
    return _QUESTION_FILLER.sub("", text)


def strip_links(text: str) -> tuple[str, list[str]]:
    """Replace markdown links by their label; return the text and the link targets
    (web, phone and mail). Images are dropped, including ones nested inside links."""
    text = _IMAGE.sub("", text)
    links = [m.group(2) for m in _MARKDOWN_LINK.finditer(text) if m.group(2).startswith(LINK_SCHEMES)]
    return _MARKDOWN_LINK.sub(lambda m: m.group(1), text), links


# Street words the site abbreviates ("3999 Pennsylvania Ave.", "Saratoga Rd."): one form for both spellings.
STREET_ABBREVIATIONS = {"avenue": "ave", "road": "rd", "boulevard": "blvd", "highway": "hwy", "street": "st", "drive": "dr"}


@lru_cache(maxsize=1 << 16)  # millions of calls at ingest over a vocabulary of tens of thousands
def stem(token: str) -> str:
    """Crude plural folding so 'months' matches 'month' and 'branches' matches 'branch'."""
    if token in STREET_ABBREVIATIONS:
        return STREET_ABBREVIATIONS[token]
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
