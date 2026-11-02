# Notes

## Requirements
- Python 3.9+, standard library only. No third-party packages, no network, no API keys.
- Nothing is written to disk; everything is built in memory inside `ingest()`.

## How it works
**Ingest.** Each snapshot is split into blocks: one per table row (with its caption, column
headers and the "effective as of" date printed after the table) and short passages of prose.
Every block also carries, never shown to the member: its page's title, URL words and nearest
H1/H2 headings (searchable, at a reduced weight), the date the page states for itself
(`pagedate.py`), and the page type. Footnote digits, dashes, number formats (including other
currencies) are normalised, and text repeated across many pages (forms, country lists) is
dropped. Each snapshot gets its own BM25 index; a word's rarity (idf) is computed from the
blocks' own text only.

**Page types and priors.** A page is classed from its URL as rate sheet, product, service,
document, news, blog or calculator (`segments.py`). About 80% of the site is articles, so every
page gets a prior: a fixed one by type (rate sheet and product above news, blog and calculators)
times a hub boost from how many other pages link to it. A question is also given an intent
(rate, news, contact, membership, policy, general), which prefers some page types and disfavours
others. A question word that is also a word in a page's address ("student", "loans") gives that page
a small boost. All of these are soft factors on a block's score, except that a rate or fee question
is never answered from a blog or press-release prose page. Among near-equal candidates the order is
qualifier fit, then page authority, then freshness, so a newer announcement does not outrank a product
page on date alone.

**Answering.** The snapshot is chosen from `as_of` (newest on or before that date, or the newest
overall). BM25 returns candidate blocks, which are kept only if they cover the question's key
words (page context counts) and hold the kind of value asked for: a rate or money amount, a
clock time for hours; in prose, a percent must sit next to a rate word. Qualifiers decide between
look-alike facts: a term ("60-month", "30-year") must match, a model year must fall inside the
row's range ("2025" is inside "2024-27"), an amount must fall inside the row's amount tier, and
a tier the question does not name (jumbo, business, Roth, Traditional, one-off "relief"
programs) is down-weighted so the plain tier is the default. Rate and fee questions search the
table rows first and fall back to all blocks only when no row qualifies. A plural topic noun
("cards", "rates") or a word like "all"/"offer" asks for several rows. A yes/no capability
question ("Can I ...?") prefers a block with an affirmative cue.

**Question kinds with their own evidence.** Some questions say what kind of fact they want:
a contact question needs a phone number (the number repeated on the most blocks wins), "who is
eligible" needs a service-area description naming two or more states, "what is the X program"
needs a dollar amount, "how many <nouns> <did something>" needs a number before that noun and
the action after it (`evidence.py`). The candidates are the blocks that show that evidence.
"How many branches" and "what recognition from <org>" are answered from dated press-release
facts (`pressfacts.py`) as one short sentence, only when the question is about Dupaco.
"How do I join/apply/open ... online" answers with the site link whose address best matches the
question (host words count triple; blog, news, calculator and document pages are not link sources). A question that compares
hours between branches ("do all branches keep the same lobby hours?") shows up to three distinct
schedules found across pages (`schedules.py`).

**Several rows.** A list question returns the best block plus similar blocks, at most eight
(rows may come from several pages), in page order; other answers take siblings from the same
page and date, at most six; the hours comparison shows up to three schedules of eight lines.

**What changed.** A question with a change word (changed, since, versus, increased...) and a
rate or fee word is answered from the newest snapshot and an earlier one (the first snapshot
taken in or after the month named, else the second newest, or the oldest if no month is named) as
"Now: ... / As of <date>: ...", or "No change since <date>". Both values are printed on purpose.
Two named dates ("between May 12 and June 9", "from May to September") pick the two snapshots
directly. Each snapshot contributes its best-ranked rate table; rows are paired by tier (model-year
ranges by rank, so "2023-26" lines up with "2024-27") and only the tiers whose figures differ are
printed (at most eight).

**Sources.** A normal answer cites its page and every other page carrying exactly the same text,
never a page with a different value (change and online-link answers cite a single page).

**Years and expired offers.** Each year in a question is classified by its own context: attached to a
vehicle word ("a 2017 car", "model year 2017") it is a model year matched to the row's range; after
in/during/as of/since/by it is a calendar year, which prefers pages dated that year and cuts dated
press facts off at the end of that year (a rate question about a year before the first snapshot gets
"isn't listed"); a stated percent or long number is a claim, not a topic word. A block is expired when its
page address or title says so, its text says the offer ended, or it states an end date earlier than the
snapshot's own date; expired blocks lose heavily in ranking, and if the only match is an ended offer the
answer is "That offer is no longer available."

**Not on the site.** If no block qualifies, the answer is a short "isn't listed, please contact
us" with no numbers, and `confident` is false. The question's rarest topic word must appear in
the block, which also refuses questions about other banks or products the credit union lacks.

**Casual wording and sentences.** Before parsing, `normalize_question` folds slang, abbreviations,
amount shorthand ("35 grand"), a plain year with a vehicle word ("2018 pickup" becomes a model year)
and relative dates ("late August") to the site's words. Long passages are also indexed sentence by
sentence (`sentences.py`); a question that asks for a value (how much, how long, when, what age,
phone, how many) is answered by the one sentence that shows that kind of value, with guards that
keep a sentence about another topic from answering.

**Lexicon and typed facts.** `lexicon.py` holds the question-side word tables: chat shorthand,
money slang, family words for membership questions, everyday paraphrases (kick in, cash out,
overseas), filler words, and a conservative one-edit typo fixer. Each entry is either a search
expansion or coverage-only, and was checked against questions that must be refused. `facts.py`
extracts phone, address, fee and date facts at ingest; a question asking for one of those kinds
is answered with the stored sentence when the fact is unique and the search answer lacks it.
Change questions are answered as a short chronology across the snapshots ("X, then Y (from
date). Current: Y"), and an old tier name maps to its current one by rank in the same table.

## Deciding what is current
The newest snapshot wins. A block's freshness is its table's "effective as of" date, else the
date its page states for itself. Among candidates that score about the same, the freshest wins,
and where several pages carry the same table row only the newest-dated copy is kept, however
well an older copy matches, so a promo or calculator page still showing last winter's rate loses
to the current rate sheet. For counts and ordinals in press releases ("12 branch office
locations", "the third time"), the value on the release with the latest printed date wins.
Expired promotions are not specially handled when answering; the "ended/expired" signal is used
only to pick a winner in `conflicts()`.

Dates written in the question work too: "in May 2026", "as of March", "on May 12, 2026",
"2026-05-12" or a season ("in the fall": spring ends May 31, summer Aug 31, fall Nov 30, winter
Feb 28) selects the snapshot (when no `as_of` is given) and is removed from the question; a date after the newest snapshot
("today (October 2026)") is left alone. `as_of` selects the newest snapshot taken on or before it
(ISO or M/D/YYYY); an unreadable date is treated as "now", and a date before the first snapshot
gets "isn't listed".

## Conflicts
`conflicts()` looks only at the latest snapshot. It compares table rows by their tier cells
(term, balance band, model-year range ranked within its page, so "2023-26" lines up with
"2024-27") and short prose statements like "6-month certificate APY: 3.50%" (not from blog or
news pages). Two pages that share a product word but give different rate, APY or APR numbers are
a conflict; entries are merged per product (mortgage purchase and refinance stay separate).
The winner is the block with the later freshness date; without dates, the one that is not an ended
offer; without that, the more authoritative page type (rate sheet and product over service and
document, over blog and news, over calculators); otherwise there is no resolution.
Introductory-rate columns are ignored. Dated press-release facts are handled separately: branch
counts and "Nth time X recognized Dupaco" differ between old and new releases, and the value on
the latest-dated release wins.

## Repository layout
The solution is the top-level `*.py` modules; `solution.py` is the entry point. Nothing else is
needed to run it.

## Honest disclosure
The approach is general-purpose, but some rules were chosen after reading the 40 practice
questions: the branch-count and "Nth time X recognized" press-release patterns, the evidence
kinds (phone, eligibility, program amount, "how many"), the small alias, filler and tier-word
lists, excluding blog and press-release pages from rate answers, and treating one-off "relief"
programs as exceptional. Later rules (aliases, filler words, tier cues, model-year and term
handling) were added after reading failures of independently written questions, never from the
practice set. No answer, number or URL from the practice set is stored in the code.

Measured on questions written from the raw pages by reviewers who had not seen the code, the
system scores about 0.9 on the public set but roughly 0.6 on fresh questions: table lookups
(rates, tiers, fees) are strong; prose, news and education questions, casual phrasing and
rewordings are weaker.

## Known gaps
- Conflicts cover rate-like numbers, branch counts (an undated page counts as the oldest
  statement) and "Nth time recognized" ordinals; other statistics, fees and award facts that
  differ between pages are not detected.
- Answers are verbatim rows or passages (except the press-release and link sentences);
  "highest/lowest rate" keeps the rate rows holding the extreme figure (fees are not compared),
  and "check a stated value" questions are not supported.
- Change answers compare the best-ranked rate table of each snapshot by tier; if a table was
  restructured between the two, tiers that no longer match are skipped.
- "Last year" is not resolved to a calendar year: "how many employees used Volunteer Time Off
  last year" returns the best-matching statistic, which may not be the one a reader expects.
- One-off facts in prose are found only when the question matches an evidence kind, a sentence
  with the asked kind of value, or ordinary search ranks the right passage first; education and
  news questions in unusual wording are often refused or answered with a nearby sentence.
- A model year attached to a vehicle word with chatty filler, parentheticals and multi-part
  questions can still be refused; rewordings of the same prose question do not always give the
  same numbers.
- Branch hours exist mostly in older press releases and are not always attributed to the named
  branch.
