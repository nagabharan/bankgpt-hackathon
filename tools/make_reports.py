"""Generate the per-stage HTML reports into mysol/reports/ (gitignored, offline, self-contained).

    python mysol/tools/make_reports.py

Scores come from SCORES.md, decision text from DECISIONS.md, "answer now" from the current code;
prose, whys and recorded "before" answers come from report_content.py.
"""
from __future__ import annotations

import html
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / "mysol"
sys.path[:0] = [str(SUBMISSION), str(ROOT / "harness"), str(Path(__file__).parent)]

from report_content import BUILT_FROM, EDGES, NODES, STAGES  # noqa: E402

OUT = SUBMISSION / "reports"
STARTER_SCORE = "0.093"
ANSWER_LIMIT = 420

CSS = """
body{font:15px/1.5 system-ui,-apple-system,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1b1f24}
h1{margin-bottom:.2rem}h2{margin-top:2rem;border-bottom:1px solid #d0d7de;padding-bottom:.2rem}
.summary{font-size:1.1rem;color:#444}.score{display:inline-block;background:#eef6ff;border:1px solid #b6d4fe;
border-radius:6px;padding:.4rem .8rem;margin:.5rem 0}.score b{font-size:1.3rem}
table{border-collapse:collapse;width:100%;margin:.5rem 0}th,td{border:1px solid #d0d7de;padding:.35rem .6rem;
text-align:left;vertical-align:top}th{background:#f6f8fa}
pre{background:#f6f8fa;padding:.6rem;white-space:pre-wrap;border-radius:6px;font-size:13px}
.why{margin:.2rem 0}.why b{color:#0b5cad}.muted{color:#6a737d}nav a{margin-right:1rem}
svg text{font:11px system-ui,sans-serif}
"""
# Flow geometry
BOX_W, BOX_H, GAP, ROW_Y = 118, 46, 16, {"ingest": 24, "answer": 124}
COLOURS = {"changed": "#ffd33d", "built": "#dbe9ff", "later": "#f1f1f1"}


def esc(text: object) -> str:
    return html.escape(str(text))


def read_scores() -> list[dict]:
    lines = (SUBMISSION / "SCORES.md").read_text().splitlines()
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    return [dict(zip(header, [c.strip() for c in line.strip("|").split("|")])) for line in lines[2:] if line.strip()]


def read_decisions() -> dict[str, str]:
    sections, heading, body = {}, None, []
    for line in (SUBMISSION / "DECISIONS.md").read_text().splitlines():
        if line.startswith("## "):
            if heading:
                sections[heading] = "\n".join(body).strip()
            heading, body = line[3:].strip(), []
        elif heading:
            body.append(line)
    if heading:
        sections[heading] = "\n".join(body).strip()
    return sections


def stage_scores(rows: list[dict], labels: list[str]) -> dict:
    """The last passing row of the stage (or its last row if none passed)."""
    mine = [r for r in rows if r["label"] in labels]
    passing = [r for r in mine if r["gate"] == "PASS"]
    return (passing or mine or [{}])[-1]


def flow_svg(stage_no: int, changed: set[str]) -> str:
    pos = {nid: (20 + col * (BOX_W + GAP), ROW_Y[row]) for nid, _, row, col in NODES}
    parts = ['<svg viewBox="0 0 1080 200" width="100%" role="img" aria-label="flow">',
             '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
             '<path d="M0,0 L8,4 L0,8 z" fill="#57606a"/></marker></defs>',
             '<text x="20" y="14" fill="#57606a">Ingest (once per snapshot)</text>',
             '<text x="20" y="114" fill="#57606a">Answer (per question)</text>']
    for a, b in EDGES:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        if y1 == y2:
            parts.append(f'<line x1="{x1 + BOX_W}" y1="{y1 + BOX_H // 2}" x2="{x2}" y2="{y2 + BOX_H // 2}" '
                         'stroke="#57606a" marker-end="url(#a)"/>')
        else:
            parts.append(f'<line x1="{x1 + BOX_W // 2}" y1="{y1 + BOX_H}" x2="{x2 + BOX_W // 2}" y2="{y2}" '
                         'stroke="#57606a" marker-end="url(#a)"/>')
    for nid, label, _, _ in NODES:
        x, y = pos[nid]
        state = "changed" if nid in changed else "built" if BUILT_FROM[nid] <= stage_no else "later"
        dash = ' stroke-dasharray="4 3"' if state == "later" else ""
        parts.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="6" fill="{COLOURS[state]}" '
                     f'stroke="#57606a"{dash}/>')
        for i, line in enumerate(textwrap.wrap(label, 17)[:3]):
            parts.append(f'<text x="{x + BOX_W // 2}" y="{y + 18 + i * 13}" text-anchor="middle">{esc(line)}</text>')
    parts.append('<text x="20" y="196" fill="#57606a">yellow = changed in this stage, blue = already built, '
                 'dashed grey = built in a later stage</text></svg>')
    return "".join(parts)


class LiveSystem:
    """The current code, ingested once, for the 'answer now' column."""

    def __init__(self) -> None:
        self._kb = None

    def kb(self):
        if self._kb is None:
            from contract import Page, Snapshot
            import solution
            self._kb = solution.build()
            for folder in sorted((ROOT / "snapshots").glob("0*")):
                manifest = json.loads((folder / "manifest.json").read_text())
                pages = [Page(p["url"], p["title"], (folder / p["file"]).read_text(), p["fetched_at"])
                         for p in manifest["pages"]]
                self._kb.ingest(Snapshot(manifest["snapshot_id"], manifest["taken_on"], pages))
        return self._kb

    def answer(self, question: str, as_of: str | None) -> str:
        if question.startswith("conflicts()"):
            found = self.kb().conflicts()
            return f"{len(found)} entries: " + "; ".join(f"{c.topic[:48]} -> {c.resolution}" for c in found)
        text = self.kb().answer(question, as_of).text.replace("\n", " // ")
        return text if len(text) <= ANSWER_LIMIT else text[:ANSWER_LIMIT] + " ..."


def score_cells(row: dict) -> str:
    keys = [("final", "final"), ("accuracy", "accuracy"), ("consistency", "consistency"), ("speed", "speed"),
            ("stale hits", "stale hits"), ("p95 s", "p95 s"), ("ingest s", "ingest s"), ("selftest", "selftest")]
    return "".join(f"<td>{esc(row.get(k, '-'))}</td>" for k, _ in keys)


def score_table(before: dict, after: dict) -> str:
    heads = "".join(f"<th>{h}</th>" for h in
                    ("", "final", "accuracy", "consistency", "speed", "stale hits", "p95 s", "ingest s", "selftest"))
    return (f"<table><tr>{heads}</tr><tr><th>before</th>{score_cells(before)}</tr>"
            f"<tr><th>after</th>{score_cells(after)}</tr></table>")


def page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>{esc(title)}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def render_stage(n: int, stage: dict, before: dict, after: dict, decisions: dict[str, str], live: LiveSystem) -> str:
    out = [f"<nav><a href=\"index.html\">All stages</a></nav><h1>{n}. {esc(stage['title'])}</h1>",
           f"<p class=\"summary\">{esc(stage['summary'])}</p>",
           f"<div class=\"score\">score <b>{esc(before.get('final', '-'))}</b> &rarr; <b>{esc(after.get('final', '-'))}</b></div>",
           score_table(before, after),
           "<h2>Flow</h2>", flow_svg(n, stage["changed"]),
           "<h2>What changed</h2>", f"<p>{esc(stage['idea'])}</p>",
           "<p>Files: " + ", ".join(f"<code>{esc(f)}</code>" for f in stage["files"]) + "</p>",
           "<h2>Alternatives and tradeoffs</h2>"]
    for heading in stage["decisions"]:
        body = decisions.get(heading, "(entry not found in DECISIONS.md)")
        out.append(f"<h4>{esc(heading)}</h4><pre>{esc(body)}</pre>")
    out.append("<h2>5 whys</h2>")
    out += [f"<p class=\"why\"><b>{i}. {esc(q)}</b><br>{esc(a)}</p>" for i, (q, a) in enumerate(stage["whys"], 1)]
    out.append("<h2>Before and after</h2>")
    if stage["examples"]:
        rows = "".join(
            f"<tr><td>{esc(q)}{' <span class=muted>(as of ' + esc(asof) + ')</span>' if asof else ''}</td>"
            f"<td>{esc(b) if b else '<span class=muted>not recorded</span>'}</td><td>{esc(live.answer(q, asof))}</td></tr>"
            for q, asof, b in stage["examples"])
        out.append("<table><tr><th>Question</th><th>Before this stage (recorded)</th><th>Answer now (current code)</th></tr>"
                   f"{rows}</table>")
    else:
        out.append("<p class=\"muted\">No answers yet at this stage.</p>")
    out.append("<h2>Still wrong at this stage</h2><ul>" + "".join(f"<li>{esc(g)}</li>" for g in stage["gaps"]) + "</ul>")
    out.append("<h2>Likely questions</h2>")
    out += [f"<p class=\"why\"><b>{esc(q)}</b><br>{esc(a)}</p>" for q, a in stage["faq"]]
    return page(f"{n}. {stage['title']}", "".join(out))


def render_index(rows: list[tuple[int, dict, dict, dict]]) -> str:
    body = ["<h1>Build reports</h1><p class=\"muted\">One page per stage: what changed, why, and what is still wrong. "
            "Scores come from the gate log.</p><table><tr><th>#</th><th>Stage</th><th>Before</th><th>After</th>"
            "<th>Summary</th></tr>"]
    for n, stage, before, after in rows:
        body.append(f"<tr><td>{n}</td><td><a href=\"{esc(stage['slug'])}.html\">{esc(stage['title'])}</a></td>"
                    f"<td>{esc(before.get('final', '-'))}</td><td>{esc(after.get('final', '-'))}</td>"
                    f"<td>{esc(stage['summary'])}</td></tr>")
    body.append("</table>")
    return page("Build reports", "".join(body))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    scores, decisions, live = read_scores(), read_decisions(), LiveSystem()
    previous: dict = {"final": STARTER_SCORE, "selftest": "-"}  # the starter is the first "before"
    listing = []
    for n, stage in enumerate(STAGES, 1):
        after = stage_scores(scores, stage["labels"])
        (OUT / f"{stage['slug']}.html").write_text(render_stage(n, stage, previous, after, decisions, live))
        listing.append((n, stage, previous, after))
        previous = after
    (OUT / "index.html").write_text(render_index(listing))
    print(f"wrote {len(STAGES) + 1} files to {OUT}")


if __name__ == "__main__":
    main()
