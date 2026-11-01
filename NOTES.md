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
  Roth, Traditional) is down-weighted so the plain tier is the default. Table rows are
  preferred for rate and fee questions.
- **Not on the site:** if no block qualifies, the answer is a short "isn't listed, please
  contact us" with no numbers, and `confident` is false.

## Deciding what is current
The newest snapshot wins. Within it, candidates that score about the same compete on the
"effective as of" date printed under each rate table, so a rate sheet dated 08/31/2026 beats
a promo page still showing last winter's rate. Expired promotions and press-release dates
are not handled yet; this file will be updated as they land.

## Known gaps
- `conflicts()` returns an empty list for now.
- Answers return a whole matching row or passage; picking the exact tier and combining several
  rows for list questions are still to do.
