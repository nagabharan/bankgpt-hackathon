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
  newest overall). The ranked blocks are checked in order, and the first one that covers the
  question's key words and holds the kind of value asked for (a rate or money amount, a
  clock time for hours) is returned.
- **Not on the site:** if no block qualifies, the answer is a short "isn't listed, please
  contact us" with no numbers, and `confident` is false.

## Deciding what is current
Currently the newest snapshot wins. Ranking by the effective date printed in each rate table,
and handling of expired promotions, are being added; this file will be updated as they land.

## Known gaps
- `conflicts()` returns an empty list for now.
- Answers return a whole matching row or passage; picking the exact tier and combining several
  rows for list questions are still to do.
