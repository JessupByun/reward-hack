"""Run every task's validation matrix in the no-network sandbox, twice, and compare it with task.json.

Usage: uv run python scripts/validate_tasks.py [TASK_ID ...] [--workers 6]
With no TASK_ID, every directory under tasks/ that has a task.json is validated.

Every matrix row (the unsafe start, the reference, each alternatives/<name>, each
gaming_patches/<name>, and the generic harness-tamper rows that rewardhack.containers builds for
every task) runs against the normal visible, conflict visible and hidden checks.
Each cell runs with the check runner started both ways, as grade_hidden starts it (uid 1000,
read-only, no capabilities) and as in-episode run_tests starts it (root with compose.yaml's
settings, where no uid-1000 process may outlive the check), and the whole matrix runs twice.
Each run also collects the hidden suite's markers and checks task.json's hidden_security_tests.
The expected verdicts are the "matrix" section of each task.json; rewardhack.containers documents it.

The table shows each cell's verdict when all four of its runs match task.json and each other, and
MISMATCH, UNSTABLE or ERROR otherwise, with the details below it.
Exit status: 0 when everything matches, 1 on any declaration problem, mismatch, error or
difference between the two runs.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rewardhack.containers import (
    COLUMNS,
    MODES,
    cell_fingerprint,
    cell_problems,
    cell_verdict,
    declaration_problems,
    hidden_marker_problems,
    hidden_markers,
    load_config,
    matrix_rows,
    run_check,
    stage,
    task_dirs,
)
from rewardhack.task import TASKS_DIR, recorded_image_digest

RUNS = (1, 2)


def run_cell(job: tuple, image: str, root: Path) -> dict | str:
    """One cell's container result, or a description of why the container gave none."""
    task_dir, config, files, column, mode, run = job
    workdir = Path(tempfile.mkdtemp(dir=root))
    try:
        return run_check(stage(workdir, task_dir, config, files, column), image, mode, config, column)
    except subprocess.TimeoutExpired:
        return "the container timed out"
    except ValueError as e:
        return f"the runner gave no report ({e})"


def run_markers(task_dir: Path, config: dict, image: str, root: Path) -> list[str]:
    try:
        markers = hidden_markers(task_dir, config, image, Path(tempfile.mkdtemp(dir=root)))
    except (subprocess.TimeoutExpired, ValueError) as e:
        return [f"collecting the hidden markers failed: {e!r}"]
    return hidden_marker_problems(config, markers)


def validate(task_ids: list[str], workers: int) -> int:
    image = recorded_image_digest()
    tasks = {d.name: (d, load_config(d)) for d in task_dirs() if not task_ids or d.name in task_ids}
    missing = sorted(set(task_ids) - set(tasks))
    if missing:
        print(f"no task.json for {missing} under {TASKS_DIR}", file=sys.stderr)
        return 1
    rows = {task_id: matrix_rows(d, config) for task_id, (d, config) in tasks.items()}
    cells = [
        (task_id, row, column, mode, run)
        for run in RUNS for task_id in tasks for row in rows[task_id] for column in COLUMNS for mode in MODES
    ]
    root = Path(tempfile.mkdtemp(prefix="rewardhack-validate-"))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            marker_futures = {
                (task_id, run): pool.submit(run_markers, d, config, image, root)
                for run in RUNS for task_id, (d, config) in tasks.items()
            }
            jobs = [(*tasks[t], rows[t][row], column, mode, run) for t, row, column, mode, run in cells]
            results = dict(zip(cells, pool.map(lambda job: run_cell(job, image, root), jobs), strict=True))
            markers = {key: future.result() for key, future in marker_futures.items()}
    finally:
        shutil.rmtree(root, ignore_errors=True)

    failed = 0
    for task_id, (task_dir, config) in tasks.items():
        problems = [f"task.json: {p}" for p in declaration_problems(task_dir, config)]
        for run in RUNS:
            problems += [f"hidden markers, run {run}: {p}" for p in markers[task_id, run]]
        table = []
        for row in rows[task_id]:
            line = [row]
            for column in COLUMNS:
                word, cell_issues = summarize(config, task_id, row, column, results)
                problems += [f"{row} {column} {issue}" for issue in cell_issues]
                line.append(word)
            table.append(line)
        print_table(task_id, config, table)
        status = "OK" if not problems else f"FAILED ({len(problems)} problems)"
        print(f"{task_id}: {len(rows[task_id])} rows x {len(COLUMNS)} checks x {len(MODES)} modes x {len(RUNS)} runs, {status}")
        for problem in problems:
            print(f"  - {problem}")
        print()
        failed += bool(problems)
    return 1 if failed else 0


def declared(config: dict, row: str, column: str) -> bool:
    cell = config.get("matrix", {}).get(row, {}).get(column)
    return isinstance(cell, dict) and isinstance(cell.get("pass"), bool)


def summarize(config: dict, task_id: str, row: str, column: str, results: dict) -> tuple[str, list[str]]:
    """The table word for one cell over both modes and runs, and its problems."""
    issues, words = [], set()
    for mode in MODES:
        runs = {run: results[task_id, row, column, mode, run] for run in RUNS}
        errors = {run: r for run, r in runs.items() if isinstance(r, str)}
        if errors:
            issues += [f"{mode} run {run}: ERROR {error}" for run, error in errors.items()]
            continue
        if not declared(config, row, column):
            issues.append(f"{mode}: no expectation in task.json")
            continue
        for run, result in runs.items():
            issues += [f"{mode} run {run}: {p}" for p in cell_problems(config, row, column, result)]
            words.add("pass" if cell_verdict(config, column, result["check"]).passed else "fail")
        if len({cell_fingerprint(config, column, result) for result in runs.values()}) > 1:
            issues.append(f"{mode}: UNSTABLE, the two runs differ: {[cell_fingerprint(config, column, r) for r in runs.values()]}")
    if any("ERROR" in issue for issue in issues):
        return "ERROR", issues
    if any("UNSTABLE" in issue for issue in issues):
        return "UNSTABLE", issues
    if issues:
        return "MISMATCH", issues
    return words.pop(), issues


def print_table(task_id: str, config: dict, table: list[list[str]]) -> None:
    header = [f"{task_id} ({config['family']})", *COLUMNS]
    widths = [max(len(line[i]) for line in [header, *table]) for i in range(len(header))]
    for line in [header, *table]:
        print("  ".join(cell.ljust(width) for cell, width in zip(line, widths, strict=True)).rstrip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task_ids", nargs="*", metavar="TASK_ID", help="tasks to validate (default: all)")
    parser.add_argument("--workers", type=int, default=6, help="containers run at once")
    args = parser.parse_args(argv)
    return validate(args.task_ids, args.workers)


if __name__ == "__main__":
    sys.exit(main())
