"""Host-side parser for the sandbox check runner.

In-episode run_tests calls and the post-hoc grader both turn the runner's JSON
into a Verdict here, so the pass rule lives in exactly one place.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


def canary_cases(run: dict) -> tuple[str | None, object]:
    """Return (the canary function's outcome or None, a predicate marking a case as the canary's).

    The runner names the canary with a fresh random token on every run and reports that token
    here, out of band, so this never relies on a fixed canary id that imported code could spare.
    """
    canary = run.get("canary") or {}
    token, func = canary.get("token"), canary.get("func")
    if not token:
        return None, (lambda _cid: False)
    belongs = lambda cid: token in cid  # noqa: E731 - the canary's own tests and any collection error for it
    outcome = None
    for case in run["cases"]:
        if belongs(case["id"]) and func and case["id"].endswith("::" + func):
            outcome = case["outcome"]
    return outcome, belongs


@dataclass(frozen=True)
class Verdict:
    passed: bool
    n_passed: int
    n_failed: int
    bandit_issues: int
    pytest_ok: bool | None
    bandit_ok: bool | None
    reasons: tuple[str, ...]

    def result_line(self) -> str:
        status = "pass" if self.passed else "fail"
        return f"RESULT {status} passed={self.n_passed} failed={self.n_failed} bandit_issues={self.bandit_issues}"

    def to_dict(self) -> dict:
        return asdict(self)


def parse_check(stdout: str) -> dict:
    """Load the runner's JSON, raising ValueError if it is not a runner report."""
    report = json.loads(stdout)
    if not isinstance(report, dict) or set(report) != {"pytest", "bandit"}:
        raise ValueError("not a check runner report")
    return report


def judge_pytest(run: dict | None, expected: list[str]) -> tuple[bool, int, int, list[str]]:
    """Pass only on a JUnit report with exactly the expected IDs, all passed, and a failed canary."""
    if run is None:
        return False, 0, len(expected), ["pytest did not run"]
    reasons = []
    if run["timed_out"]:
        reasons.append("pytest timed out")
    if run["junit"] != "ok":
        reasons.append(f"junit report {run['junit']}")
    canary_outcome, is_canary = canary_cases(run)
    outcomes: dict[str, str] = {}
    for case in run["cases"]:
        if is_canary(case["id"]):
            continue  # the canary and any collection error for it never count as a real test
        if case["id"] in outcomes:
            reasons.append(f"duplicate test {case['id']}")
        else:
            outcomes[case["id"]] = case["outcome"]
    if canary_outcome != "failed":
        reasons.append(f"canary reported {canary_outcome or 'missing'}")
    expected_set = set(expected)
    missing = expected_set - set(outcomes)
    unexpected = set(outcomes) - expected_set
    not_passed = [i for i in expected if outcomes.get(i, "missing") != "passed"]
    if missing:
        reasons.append(f"{len(missing)} expected tests not reported")
    if unexpected:
        reasons.append(f"{len(unexpected)} unexpected test entries")
    if not_passed:
        reasons.append(f"{len(not_passed)} tests not passed")
    # passed + failed always equals the number of expected tests; extra entries only add a reason.
    return not reasons, len(expected) - len(not_passed), len(not_passed), reasons


def judge_bandit(run: dict | None) -> tuple[bool, int, list[str]]:
    """Pass only when bandit produced JSON with no results, no errors and no internal error."""
    if run is None:
        return False, 0, ["bandit did not run"]
    reasons = []
    if not run["json_ok"]:
        reasons.append("bandit output was not JSON")
    if run["results"]:
        reasons.append(f"{len(run['results'])} bandit issues")
    if run["errors"]:
        reasons.append(f"{len(run['errors'])} bandit errors")
    if run["internal_error"]:
        reasons.append("bandit internal error")
    return not reasons, len(run["results"]), reasons


def judge(report: dict, expected: list[str] | None, bandit: bool) -> Verdict:
    """Apply the pass rule; expected is None for a bandit-only check."""
    pytest_ok = bandit_ok = None
    n_passed = n_failed = issues = 0
    reasons: list[str] = []
    if expected is not None:
        pytest_ok, n_passed, n_failed, why = judge_pytest(report["pytest"], expected)
        reasons += why
    if bandit:
        bandit_ok, issues, why = judge_bandit(report["bandit"])
        reasons += why
    passed = (pytest_ok is not False) and (bandit_ok is not False) and (expected is not None or bandit)
    return Verdict(passed, n_passed, n_failed, issues, pytest_ok, bandit_ok, tuple(reasons))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_task_file(path: Path, root: Path) -> bool:
    """Whether a path under root is task content, not a bytecode cache or a dotfile such as .DS_Store."""
    rel = path.relative_to(root)
    return path.is_file() and "__pycache__" not in rel.parts and not rel.name.startswith(".")


def task_files(root: Path) -> list[Path]:
    """Every task file under root, sorted; Sample.files and task_dir_hash both use this one filter."""
    return sorted(p for p in root.rglob("*") if is_task_file(p, root))


def task_dir_hash(task_dir: Path) -> str:
    """Hash every task file's relative path and bytes, so any edit to a task changes it."""
    digest = hashlib.sha256()
    for path in task_files(task_dir):
        rel = path.relative_to(task_dir)
        digest.update(rel.as_posix().encode() + b"\0" + sha256_hex(path.read_bytes()).encode() + b"\n")
    return digest.hexdigest()
