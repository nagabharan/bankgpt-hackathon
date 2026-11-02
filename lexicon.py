"""Everyday wording folded to the words the site uses: chat shorthand, slang, family words, paraphrases
and one-letter typos. It is applied to questions only; page text is never touched.

Three kinds of entry, by how much they may change a question:
  * folds rewrite the question text (`fold`), so the rewritten words must then be on the page;
  * search aliases add words to the query (and count for coverage), for words that mean the same
    thing on this site;
  * coverage aliases only let a block that has the other word satisfy the coverage check; they never
    add a word to the search, so they cannot pull in another product's rows.
"""
from __future__ import annotations

import re
from collections import defaultdict

# Chat filler (dropped from questions: these words also occur in page text, so never from pages).
FILLER_WORDS = (
    "but", "still", "also", "put", "newest", "latest", "newer", "big", "bracket", "pls", "plz", "thx", "thanks", "hey", "hi",
    "hello", "yo", "okay", "btw", "lol", "asap", "anyway", "honestly", "basically", "kinda", "sorta", "ish", "roughly",
    "approx", "approximately", "throw", "toss", "stick", "sits", "sit", "ahead", "better", "kid", "kids",
    "tho", "though", "cuz", "coz", "ty", "tysm", "omg", "fyi", "imo", "idk", "um", "uh", "umm", "hmm", "literally",
)

# Shorthand and slang folded to the site's words. (pattern, replacement); applied in order.
CASUAL_WORDS = (
    (re.compile(r"\bcerts?\b", re.I), "certificate"),
    (re.compile(r"\bmos?\b", re.I), "months"),
    (re.compile(r"\byrs?\b", re.I), "years"),
    (re.compile(r"\bu\b", re.I), "you"),
    (re.compile(r"\bcds?\b(?!\s*[/)(,]*\s*(?:or\s+)?certificates?\b)(?<!certificate\s\()(?<!certificate\s)", re.I), "certificate"),  # "CD / certificate" already says it
    (re.compile(r"\bmax\b\.?", re.I), "maximum"),
    # "dodge the fee" is the verb; a make is a car only elsewhere.
    (re.compile(r"\bdodg(?:e|ing|ed)\b(?=\s+(?:the|a|any|that|those|these|my|your|fees?|charges?|penalt(?:y|ies)|overdrafts?|monthly)\b)", re.I), "avoid"),
    (re.compile(r"\b(?:harley(?:[- ]davidson)?|yamaha|kawasaki|ducati|triumph|vespa|scooter|moped|dirt ?bike|sportbike|cruiser|chopper)s?\b", re.I), "motorcycle"),
    (re.compile(r"\b(?:civic|accord|camry|corolla|prius|altima|sentra|mustang|corvette|wrangler|jeep|tesla|f-?150|silverado|tacoma|tundra|ram|rav4|cr-?v|explorer|equinox|malibu|impala|fusion|focus|elantra|sonata|outback|forester|mazda|subaru|toyota|honda|ford|chevy|chevrolet|dodge|nissan|hyundai|kia|bmw|audi|lexus|volkswagen|vw|buick|cadillac|gmc)s?\b", re.I), "car"),
    (re.compile(r"\bacct\b", re.I), "account"),
    (re.compile(r"\bmin\b\.?", re.I), "minimum"),
    (re.compile(r"\b(?:suv|pickup|pick-up|truck|sedan|minivan|van|crossover|coupe|hatchback|wagon)s?\b", re.I), "car"),
    (re.compile(r"\b(\d+)(?=(?:yr|mo)s?\b)", re.I), r"\1 "),  # "30yr" -> "30 yr"
    # Chat shorthand.
    (re.compile(r"\bw/o\b", re.I), "without"),
    (re.compile(r"\bw/\s*", re.I), "with "),
    (re.compile(r"\bb/c\b", re.I), " "),
    (re.compile(r"\bur\b", re.I), "your"),
    (re.compile(r"\br\b", re.I), "are"),
    (re.compile(r"\by'?all\b", re.I), "you"),
    (re.compile(r"\bamts?\b", re.I), "amount"),
    (re.compile(r"\bbal\b", re.I), "balance"),
    (re.compile(r"\bwks?\b", re.I), "week"),
    (re.compile(r"\bhrs?\b", re.I), "hours"),
    (re.compile(r"\b(?:tel|ph)\b\.?", re.I), "phone"),
    (re.compile(r"\bpwd\b", re.I), "password"),
    (re.compile(r"\btxt\b", re.I), "text"),
    (re.compile(r"\bmsg\b", re.I), "message"),
    (re.compile(r"\baddr\b", re.I), "address"),
    (re.compile(r"\bppl\b", re.I), "people"),
    (re.compile(r"\bappt\b", re.I), "appointment"),
    (re.compile(r"\baccts\b", re.I), "accounts"),
    (re.compile(r"\bchk(?:g)?\b", re.I), "checking"),
    (re.compile(r"\bsvgs?\b", re.I), "savings"),
    (re.compile(r"\bmtg\b", re.I), "mortgage"),
    (re.compile(r"\btxns?\b", re.I), "transaction"),
    (re.compile(r"\bdd\b", re.I), "direct deposit"),
    (re.compile(r"\bcc\b", re.I), "credit card"),
    (re.compile(r"\bdc\b", re.I), "debit card"),
    (re.compile(r"\bcu\b", re.I), "credit union"),
    (re.compile(r"\bdownpayments?\b", re.I), "down payment"),
    (re.compile(r"\be-mail(s?)\b", re.I), r"email\1"),
    (re.compile(r"\b(?:sign[- ]?in|login)\b", re.I), "log in"),
    # Money slang.
    (re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*bucks?\b", re.I), r"$\1"),
    (re.compile(r"\ba grand\b", re.I), "1 grand"),
    # Paraphrases of what the site says another way.
    (re.compile(r"\bbump(?:s|ed|ing)?\b", re.I), "increase"),
    (re.compile(r"\bkick(?:s|ed|ing)? in\b", re.I), "deposit"),
    (re.compile(r"\bcash(?:ing)? out\b", re.I), "withdraw"),
    (re.compile(r"\bmisplace[sd]?\b", re.I), "lost"),
    (re.compile(r"\b(?:overseas|abroad|out of (?:the )?country|outside (?:of )?the (?:us|u\.s\.|country|states))\b", re.I), "international"),
    (re.compile(r"\bwires?d?\s+(?:me\s+)?(?:money\s+|funds\s+|cash\s+)?(?:in|into|to)\s+my\b", re.I), "incoming wire to my"),
    (re.compile(r"\b(?:receiv(?:e|ing)|get(?:ting)?|got)\s+a\s+wire\b|\bwire\s+coming\s+in\b", re.I), "incoming wire"),
)

# Relatives named in a membership question stand for "family member" (the site's eligibility wording).
_MEMBERSHIP_CONTEXT = re.compile(r"\b(?:join|joining|member|members|membership|eligible|eligibility|belong|belongs|qualify)\b", re.I)
_RELATIVES = re.compile(
    r"\b(?:bro|brother|sis|sister|mom|mother|dad|father|wife|husband|spouse|cousin|grandma|grandmother|grandpa|grandfather|"
    r"grandparent|aunt|uncle|niece|nephew|son|daughter|in-laws?|parent|sibling|relative|boyfriend|girlfriend|fiance)s?\b", re.I)


def fold_relatives(text: str) -> str:
    if _MEMBERSHIP_CONTEXT.search(text) and not re.search(r"\bfamily member\b", text, re.I):
        return _RELATIVES.sub("family member", text)
    return text


# Words that mean the same on this site: the query gains the alias words and a block with any of them covers the word.
SEARCH_ALIASES = {
    "aba": ("routing",),
}
# Coverage only: they never add a word to the search.
COVERAGE_ALIASES = {
    "info": ("information",),
    "enroll": ("open", "join"),
    "register": ("open", "join"),
    "borrow": ("loan",),
    "payroll": ("paycheck",),
    "paycheck": ("payroll",),
    "reach": ("contact", "phone", "call"),
    "talk": ("contact", "phone", "call"),
    "speak": ("contact", "phone", "call"),
    "raise": ("increase",),
    "cancel": ("close",),
    "legit": ("legitimate", "scam", "fraud"),
    "win": ("won", "winner"),
    "pic": ("photo", "picture"),
    "teenager": ("teen", "youth", "minor"),
    "scam": ("fraud", "phishing"),
    "phishing": ("scam", "fraud"),
    "stolen": ("lost",),
    "bounce": ("overdraft", "insufficient"),
    "bounced": ("overdraft", "insufficient"),
    "nsf": ("overdraft", "insufficient"),
    "owe": ("balance", "payoff"),
}


# One-letter typos: a question word the site never uses, one edit (substitution, insertion, deletion or
# swap of neighbours) away from exactly one word it uses often, with the same first letter.
TYPO_MIN_LENGTH = 5
TYPO_MIN_DF = 3


def _deletions(word: str) -> set[str]:
    return {word[:i] + word[i + 1:] for i in range(len(word))}


def _within_one_edit(a: str, b: str) -> bool:
    if a == b or abs(len(a) - len(b)) > 1 or a[0] != b[0]:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def typo_table(index) -> dict[str, set[str]]:
    """deletion-form -> site words (long, frequent) it came from, built once per index."""
    def build() -> dict[str, set[str]]:
        table: dict[str, set[str]] = defaultdict(set)
        for word in index.words(TYPO_MIN_DF):
            if len(word) >= TYPO_MIN_LENGTH and word.isalpha():
                table[word].add(word)
                for form in _deletions(word):
                    table[form].add(word)
        return table
    return index.cached("typo_table", build)


_WORD = re.compile(r"[A-Za-z]+")


def correct_typos(question: str, index, stem, skip: frozenset) -> str:
    """Replace each question word that is unknown to the site by the single frequent site word one edit away."""
    table = typo_table(index)
    if not table:
        return question

    def fix(match: re.Match) -> str:
        word = match.group(0)
        low = word.lower()
        if len(low) < TYPO_MIN_LENGTH or low in skip or index.knows(stem(low)):
            return word
        if word[0].isupper() and match.start() > 0:
            return word  # a name in running text: Omaha, Minneapolis
        found = set(table.get(low, ()))
        for form in _deletions(low):
            found |= table.get(form, set())
        found = {w for w in found if _within_one_edit(low, w)}
        return next(iter(found)) if len(found) == 1 else word
    return _WORD.sub(fix, question)
