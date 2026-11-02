"""Which kind of page a question should be answered from.

Most of the site is articles (blog, news, resources); the pages that state this credit union's
own rates, products and contact details are a small minority. Pages are classified from their URL,
questions into an intent, and each intent prefers some page types and disfavours others.
"""
from __future__ import annotations

import math
import re

PREFERRED_FACTOR = 1.25
AVOIDED_FACTOR = 0.6

_DATE_SLUG = re.compile(r"^/\d{4}/\d{2}/\d{2}/")  # old WordPress-style article URLs
_SERVICE_ROOTS = ("about", "membership", "contact", "careers", "community", "connect", "disclosures",
                  "privacy-policy", "brighttrack", "locations", "branches")


def page_type(url: str) -> str:
    """rates, product, service, blog, news, calculator or document."""
    path = "/" + url.split("//", 1)[-1].partition("/")[2].split("?")[0].strip("/")
    first = path.strip("/").split("/")[0]
    if path.lower().endswith(".pdf") or first == "hubfs":
        return "document"
    if first == "newsroom":
        return "news"
    if first == "learn" or first == "resources" or _DATE_SLUG.match(path) or path.startswith("/business/resources"):
        return "blog"
    if "calculator" in first:
        return "calculator"
    if first in ("rates", "business-rates") or first.endswith("-rates"):
        return "rates"
    if first in _SERVICE_ROOTS:
        return "service"
    return "product"


_NEWS = re.compile(r"\b(?:recogni\w+|award\w*|ranked|ranking|clos(?:e|ed|es|ing|ure)|announc\w+|partnership|volunteer\w*)\b", re.I)
_CONTACT = re.compile(r"\b(?:phone|call|hours?|address|located|location|lobby|drive-up|atm|branch\w*|routing)\b", re.I)
_POLICY = re.compile(r"\b(?:policy|policies|terms|rules?|overdraft|disputes?|penalt\w*|grace period|waive\w*|limits?|deadline)\b", re.I)
_MEMBERSHIP = re.compile(r"\b(?:eligib\w*|join|become a member|membership|member)\b", re.I)

# intent -> (preferred page types, disfavoured page types)
SEGMENTS = {
    "rate": ({"rates", "product"}, {"blog", "news", "calculator"}),
    "news": ({"news"}, {"calculator"}),
    "contact": ({"service"}, {"blog", "news", "calculator"}),
    "membership": ({"service", "product"}, {"blog", "news"}),
    "policy": ({"product", "service", "document"}, {"blog", "news"}),
    "general": (set(), set()),
}


def intent_of(question: str, asks_money: bool) -> str:
    if asks_money:
        return "rate"
    if _NEWS.search(question):
        return "news"
    if _CONTACT.search(question):
        return "contact"
    if _MEMBERSHIP.search(question):
        return "membership"
    if _POLICY.search(question):
        return "policy"
    return "general"


def segment_factor(intent: str, url: str) -> float:
    preferred, avoided = SEGMENTS[intent]
    kind = page_type(url)
    return PREFERRED_FACTOR if kind in preferred else AVOIDED_FACTOR if kind in avoided else 1.0


# How much a page type can be trusted to state this credit union's own current terms.
AUTHORITY = {"rates": 4, "product": 3, "service": 2, "document": 2, "blog": 1, "news": 1, "calculator": 0}


def authority(url: str) -> int:
    return AUTHORITY[page_type(url)]


# Fixed prior by page type: how likely a page of this type is to be the one that states a fact.
TYPE_PRIOR = {"rates": 1.3, "product": 1.15, "service": 1.1, "document": 1.0, "news": 0.9, "blog": 0.85, "calculator": 0.75}
HUB_STEP = 0.05
HUB_CAP = 1.3


def page_prior(url: str, in_links: int) -> float:
    """Type prior times a hub prior: pages that many other pages link to are the main ones."""
    return TYPE_PRIOR[page_type(url)] * min(HUB_CAP, 1 + HUB_STEP * math.log2(1 + in_links))
