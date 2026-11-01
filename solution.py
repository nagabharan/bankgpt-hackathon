"""Entry point for the grader: build() returns the knowledge system.

Skeleton: stores nothing useful yet and always declines. Ingest (blocks, facts),
answer (retrieval, tiers, freshness) and conflicts get filled in step by step.

`contract` is importable because the scorer puts harness/ on sys.path before loading
this folder; always return the harness's own Answer type (the repeat-consistency check
uses isinstance on it).
"""
from __future__ import annotations

from typing import Optional

from contract import Answer, Conflict, Snapshot

# Refusal text must contain no digits: a refusal fails if it adds any number the
# question did not contain.
NOT_LISTED = "That isn't listed on our website; please contact us."


class KnowledgeBase:
    def __init__(self) -> None:
        self.snapshot_dates: list[str] = []

    def ingest(self, snapshot: Snapshot) -> None:
        self.snapshot_dates.append(snapshot.taken_on)

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        return Answer(text=NOT_LISTED, confident=False)

    def conflicts(self) -> list[Conflict]:
        return []


def build() -> KnowledgeBase:
    return KnowledgeBase()
