"""Score gate: run the scorer and block regressions.

    python mysol/tools/gate.py blocks "block ingest + BM25"        # check, log, don't accept
    python mysol/tools/gate.py blocks "block ingest + BM25" --accept  # also raise the baseline

Checks (any failure exits 1): no score regression vs the accepted baseline, no stale-trap
hits increase, identical answers under two PYTHONHASHSEED values, p95 latency <= 2s,
no errors, stdlib-only/no-network imports in the submission, and the selftest runs.
Every run appends one row to mysol/SCORES.md.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / "mysol"
BASELINE = SUBMISSION / "tools" / "baseline.json"
SCORES_LOG = SUBMISSION / "SCORES.md"
OUT = Path("/tmp/gate_out")

P95_LIMIT_S = 2.0
SCORE_TOLERANCE = 1e-9
ALLOWED_NON_STDLIB = {"contract"}  # harness types, put on sys.path by the scorer
FORBIDDEN_MODULES = {"socket", "ssl", "urllib", "http", "ftplib", "smtplib", "subprocess", "requests"}


def run_eval(out: Path, seed: str, selftest: bool = False) -> dict:
    """Run the scorer in a subprocess; return its report.json."""
    cmd = [sys.executable, "harness/run_eval.py", "--submission", "mysol", "--out", str(out)]
    if selftest:
        cmd += ["--snapshots", "harness/selftest/snapshots", "--eval", "harness/selftest/eval.jsonl",
                "--conflicts-gold", "harness/selftest/conflicts_gold.json"]
    else:
        cmd += ["--eval", "eval/public_eval.jsonl"]
    env = {**os.environ, "PYTHONHASHSEED": seed}
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    return json.loads((out / "report.json").read_text())


def answers(out: Path) -> dict[str, str]:
    """question id -> answer text, for determinism comparison."""
    rows = (json.loads(line) for line in (out / "per_question.jsonl").read_text().splitlines())
    return {r["id"]: r["answer"]["text"] for r in rows}


def import_violations() -> list[str]:
    """Non-stdlib or network-capable imports in the submission's source files."""
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib is None:
        return ["python >= 3.10 needed for the stdlib import check"]
    local = {p.stem for p in SUBMISSION.glob("*.py")}
    found = []
    for path in SUBMISSION.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            for name in names:
                top = name.split(".")[0]
                if top in FORBIDDEN_MODULES or not (top in stdlib or top in local or top in ALLOWED_NON_STDLIB):
                    found.append(f"{path.name}: {name}")
    return found


def check(report: dict, baseline: dict, same_answers: bool) -> list[str]:
    failures = []
    if report["final_score"] < baseline.get("final_score", 0) - SCORE_TOLERANCE:
        failures.append(f"final_score {report['final_score']} < baseline {baseline['final_score']}")
    if report["stale_trap_hits"] > baseline.get("stale_trap_hits", report["stale_trap_hits"]):
        failures.append(f"stale_trap_hits rose to {report['stale_trap_hits']}")
    if report["errors"]:
        failures.append(f"{report['errors']} answer errors")
    if report["speed"]["p95_s"] > P95_LIMIT_S:
        failures.append(f"p95 {report['speed']['p95_s']}s > {P95_LIMIT_S}s")
    if not same_answers:
        failures.append("answers differ between PYTHONHASHSEED=0 and 1")
    if report["ingest_over_limit"]:
        failures.append("ingest over the 30 minute limit")
    failures += [f"import: {v}" for v in import_violations()]
    return failures


def log_row(label: str, note: str, report: dict, selftest: dict, failures: list[str]) -> None:
    if not SCORES_LOG.exists():
        SCORES_LOG.write_text("| label | note | final | accuracy | consistency | speed | stale hits "
                              "| p95 s | ingest s | selftest | gate |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    p = report["pillars"]
    row = (f"| {label} | {note} | {report['final_score']} | {p['accuracy']} | {p['consistency']} "
           f"| {p['speed']} | {report['stale_trap_hits']} | {report['speed']['p95_s']} "
           f"| {report['ingest_seconds']} | {selftest['final_score']} | {'PASS' if not failures else 'FAIL'} |\n")
    with SCORES_LOG.open("a") as f:
        f.write(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("note", nargs="?", default="")
    ap.add_argument("--accept", action="store_true", help="on pass, make this run the new baseline")
    args = ap.parse_args()

    report = run_eval(OUT / "seed0", "0")
    run_eval(OUT / "seed1", "1")  # only its answers are compared, for determinism
    selftest = run_eval(OUT / "selftest", "0", selftest=True)
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}

    failures = check(report, baseline, answers(OUT / "seed0") == answers(OUT / "seed1"))
    log_row(args.label, args.note, report, selftest, failures)

    print(f"{args.label}: final={report['final_score']} (baseline {baseline.get('final_score', 'none')}) "
          f"stale_hits={report['stale_trap_hits']} p95={report['speed']['p95_s']}s selftest={selftest['final_score']}")
    for cat, v in report.get("by_category", {}).items():
        print(f"  {cat}: {v}")
    for f in failures:
        print("FAIL:", f)
    if not failures and args.accept:
        BASELINE.write_text(json.dumps({k: report[k] for k in ("final_score", "stale_trap_hits")}))
        print("baseline updated")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
