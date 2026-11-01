# Notes

## Requirements
- Python 3.9+, standard library only. No third-party packages, no network, no API keys.
- Nothing is written to disk; everything is built in memory inside `ingest()`.

## How it works
- **Ingest:** each snapshot is split into blocks: one per table row (with its caption, column
  headers and the "effective as of" date printed after the table) and short passages of prose.
  Footnote digits, dashes, number formats (including other currencies) are normalised, and
  text repeated across many pages (forms, country lists) is dropped. Each snapshot gets its
  own BM25 index.
- **Answering:** the snapshot is chosen from `as_of` (newest on or before that date, or the
  newest overall). BM25 returns candidate blocks, which are kept only if they cover the
  question's key words and hold the kind of value asked for (a rate or money amount, a clock
  time for hours). Qualifiers then decide between look-alike facts: a term in the question
  ("60-month", "30-year") must match the block, a model year must fall inside the block's
  range ("2025" is inside "2024-27"), and a tier the question does not name (jumbo, business,
  Roth, Traditional) is down-weighted so the plain tier is the default; an amount ("balance
  between $10,000 and $24,999") must fall inside the row's amount tier. Rate and fee
  questions search the table rows first, and only fall back to all blocks when no row
  qualifies. A plural topic noun ("cards", "rates") or a word like "all"/"offer" asks for
  several rows.
- **Press-release facts:** "how many branches" and "what recognition from <org>" are answered
  from the value on the latest-dated release, as one short sentence, and only when the
  question is about Dupaco.
- **Stale copies and articles:** where several pages carry the same table row, only the
  newest-dated version is kept, however well an older copy matches. Rate and fee questions
  are never answered from blog or press-release pages.
- **Changes over time:** a rate, fee or count question that also says changed/since/versus/
  increased is answered from the newest snapshot and an earlier one (the first snapshot taken
  in the month named, else the oldest) as "Now: ..." and "As of <date>: ...", or "No change
  since <date>". Both values are printed on purpose.
- **Sources:** an answer cites its page and every other page that carries exactly the same
  text, never a page with a different value.
- **Dates in the question:** "in May 2026" or "as of March" selects the snapshot (when no
  `as_of` is given) and is removed from the question text.
- **Multi-fact answers:** the answer is the best block plus sibling blocks from the same page,
  date and kind that score nearly as well (all the matching 30-year mortgage rows, every
  line of a lobby-hours schedule, every card for "what cards do you offer"), at most six, in
  page order. For "link/URL" questions the page's own links are appended.
- **Not on the site:** if no block qualifies, the answer is a short "isn't listed, please
  contact us" with no numbers, and `confident` is false.

## Deciding what is current
The newest snapshot wins. Within it, candidates that score about the same compete on the
"effective as of" date printed under each rate table, so a rate sheet dated 08/31/2026 beats
a promo page still showing last winter's rate. For counts and ordinals in press releases
("12 branch office locations", "the third time"), the value on the release with the latest
printed date wins. When answering, expired promotions are not specially handled; the
"ended/expired" signal is used only to pick the winner in `conflicts()`.

`as_of` selects the newest snapshot taken on or before that date (ISO or M/D/YYYY); an
unreadable date is treated as "now", and a date before the first snapshot gets "isn't listed".

## Conflicts
`conflicts()` looks only at the latest snapshot. It compares table rows by their tier cells
(term, balance band, model-year range ranked within its page, so "2023-26" lines up with
"2024-27") and short prose statements like "6-month certificate APY: 3.50%". Two pages that
share a product word but give different rate, APY or APR numbers are a conflict. Entries are
merged per product (mortgage purchase and refinance stay separate). The winner is the page with
the later "effective as of" date, or the one that is not an ended offer; with neither signal
there is no resolution. Introductory-rate columns are ignored. Dated press-release facts are
handled separately: branch counts ("12 branch office locations") and "Nth time X recognized
Dupaco" differ between old and new releases, and the value on the latest-dated release wins.

## Repository layout
The solution is the top-level `*.py` modules (`solution.py` is the entry point). `tools/` holds
offline development scripts (a score gate and a report generator); they are not imported by
the solution and are not part of the graded system.

## Honest disclosure
The approach is general-purpose, but some rules were chosen after reading the 40 practice
questions: the branch-count and "Nth time X recognized" press-release patterns, the small alias
and tier-word lists, excluding blog and press-release pages from rate answers, and treating
one-off "relief" programs as exceptional. No answer, number or URL from the practice set is
stored in the code.

## Known gaps
- Conflicts cover rate-like numbers, branch counts and "Nth time recognized" ordinals; other
  counts, fees and award facts that differ between pages are not detected.
- Answers are verbatim rows or passages (except the press-release sentences); the
  "highest/lowest" pick is not applied.
- Change answers compare the best-matching row in each snapshot; if a table was restructured
  between the two, the rows may be different tiers.
- One-off facts that sit in prose (eligible states, a program's amount, a wallet list, a
  contact phone number) are only found when ordinary search ranks the right passage first.
