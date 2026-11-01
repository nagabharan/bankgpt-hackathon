"""Entry point for the grader: build() returns the knowledge system.

`contract` is importable because the scorer puts harness/ on sys.path before loading
this folder; always return the harness's own Answer type (the repeat-consistency check
uses isinstance on it).

Keep @dataclass definitions out of this file: the scorer loads it without registering it
in sys.modules, which breaks dataclasses under `from __future__ import annotations`.
"""
from __future__ import annotations

from typing import Optional

from conflicts import find_conflicts
from contract import Answer, Conflict, Snapshot, Source
from matching import asks_for_link, links_of, select_blocks
from views import SnapshotView, build_view

# Refusal text must contain no digits: a refusal fails if it adds any number the
# question did not contain.
NOT_LISTED = "That isn't listed on our website; please contact us."


class KnowledgeBase:
    def __init__(self) -> None:
        self.views: list[SnapshotView] = []  # oldest first

    def ingest(self, snapshot: Snapshot) -> None:
        self.views.append(build_view(snapshot))

    def _view_for(self, as_of: Optional[str]) -> Optional[SnapshotView]:
        """Newest snapshot taken on or before `as_of` (newest overall when None)."""
        eligible = [v for v in self.views if as_of is None or v.taken_on <= as_of]
        return eligible[-1] if eligible else None

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        view = self._view_for(as_of)
        blocks = select_blocks(view.index, question) if view else []
        if not blocks:
            return Answer(text=NOT_LISTED, confident=False)
        lines = [block.text for block in blocks]
        if asks_for_link(question):
            lines += links_of(blocks)
        source = Source(url=blocks[0].url, snapshot_id=view.snapshot_id, quote=blocks[0].text[:200])
        return Answer(text="\n".join(lines), sources=[source])

    def conflicts(self) -> list[Conflict]:
        return find_conflicts(self.views[-1]) if self.views else []


def build() -> KnowledgeBase:
    return KnowledgeBase()
