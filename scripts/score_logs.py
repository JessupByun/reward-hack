"""Copy agent logs and run the monitors on the copies (PLAN.md, "Monitors").

Usage (from the repo root):
  uv run python scripts/score_logs.py copy logs/main/haiku-normal [SOURCE ...] [--allow-error]
  uv run python scripts/score_logs.py m1 [LOG ...] [--rescore] [--dry-run]
  uv run python scripts/score_logs.py score [LOG ...] [--scorers m2 m3 m3_summary judge] [--labels PATH] [--rescore] [--dry-run]
  uv run python scripts/score_logs.py score [LOG ...] --subset UUIDS --rescore [--scorers m2 m3] [--dry-run]
  uv run --env-file .env python scripts/score_logs.py summary-target [LOG ...] [--dry-run]

copy: every .eval file under each SOURCE (a directory under logs/, searched recursively, or a file) is
copied once to logs/scored/<its path relative to logs/>, so logs/main/haiku-normal/x.eval becomes
logs/scored/main/haiku-normal/x.eval. The copy is written beside its destination and linked into
place, which fails rather than replace a file, and its bytes are checked against the source.
It refuses, writing nothing, if any destination exists or is reached through a symlink, the scored
root holds a symlinked directory, a source is outside logs/ or inside logs/scored/, logs/scored/ is
inside a source, or a log's status is not success (error too with --allow-error, for a log whose
retries were exhausted), so a copy is only ever written under the scored root itself.
Sources are only ever read.
Every command refuses a directory that holds a symlinked directory, which the recursive search would
skip without a word (Python 3.12's rglob does not follow them); copy, m1 and score also refuse a
scored root inside logs/ other than logs/scored/, one that contains logs/, a scored root that is a
symlink, and any scored root while logs/scored is a symlink or resolves anywhere but itself.

m1 and score take scored copies (default: every .eval under logs/scored/) and refuse any other path.
m1 runs M1 (deterministic, no model) and score runs the LLM scorers in PLAN's fixed order: m2, m3,
m3_summary on gpt-oss logs only, then the judge, each as PLAN's command
  inspect score <log> --overwrite --action append --model mockllm/model \
    --scorer src/rewardhack/monitors/llm_monitors.py@<scorer> --model-role '<role setting>'
with the judge's role for the judge, `-S labels=<path>` added only when --labels is not grade_hidden's
default, and never --stream. inspect score reads each missing option from an INSPECT_SCORE_* variable
(INSPECT_SCORE_STREAM would turn on --stream), so those variables are removed from its environment,
and m1 and score refuse (exit 2) before planning anything while the .env file that inspect loads
before parsing its options (the first one found walking up from the repo root) sets one; that check
reads only the file's key names, never a value.
inspect score --overwrite reads the whole log and writes it back, so two commands rewriting one copy at
once would lose one's scores: m1 and score (except a dry run) take the rewrite lock of every scored log
they were given (transcript.open_rewrite_lock) before reading any, hold them until they finish, and
refuse (exit 2), taking none, while another command or scorer holds one. Each inspect score they run
inherits the lock of its log (transcript.REWRITE_LOCK_FD_ENV), which its scorer holds as its own, and
the M1 and LLM scorers take the same lock when run by hand, so a run started meanwhile refuses.
A scorer already in a log is skipped unless --rescore, whose new score Inspect stores as <scorer>1.
PLAN's test-retest re-scores a fixed 40-trajectory subset of P with M2 and M3: --subset UUIDS (a file of
sample uuids, one per line) with --rescore adds `-S subset=<file>` to the m2 and m3 commands (the only
scorers it allows, and the default with it), so only those samples are called and the others get
status not_selected; logs holding none of them are skipped, and a uuid in no given log refuses.
--subset refuses without --rescore, and on any log it would re-score that has no full run of the
scorer under the scorer's own name, so a subset run never takes the primary score's name.
Before running anything, score plans every run's prompts and refuses (exit 1) when any cap's spend so
far plus the projection of every planned run would cross it; each scorer then checks its own run
again before its first call. --dry-run prints the projections and every command and runs nothing.
With no ledger history a prompt is projected at two worst-case calls, about $0.2 for a Sonnet judge
disclosure prompt of the pilot's size and about $0.5 for a gaming prompt of the pilot's size, so the
judge's first run should be on the pilot's positives (labels logs/pilot/labels.jsonl), which gives the
main runs a ledger mean to project from.
The ledger keeps one history for both judge prompt kinds, and the pilot has conflict positives but no
normal visible_pass_but_vulnerable episode, so that mean comes from the shorter disclosure prompts
only; the main runs' gaming prompts (full M3 transcripts, on the pilot logs about 3 times as long as
a disclosure prompt on average over all episodes and about 6 times on the judged positives) are then
projected from it, so that projection runs low.
After each run the script checks that every sample got the new score and prints its statuses, with a
warning line for samples left unscored (unparsed, summary_failed, no_episode_record) and for prompts
over 100K input tokens.

summary-target counts the tokens of every kept reasoning block in the pilot's Haiku 5.5 agent logs
(default logs/pilot) with Anthropic's free count_tokens endpoint (through Inspect, so it needs
ANTHROPIC_API_KEY in the environment) and writes the median, with the median word count, to
src/rewardhack/monitors/prompts/summary_target.json once; it refuses if that file exists.
--counter-model mockllm/model counts locally, for tests.

Exit status: 0 done; 1 refused over a cap, or a run failed, left a sample without the new score, or
gave a sample status error; 2 refused for any other reason (bad paths, existing copies, a scored copy
another command is rewriting, a missing ledger, labels, subset or target).
"""

import argparse
import contextlib
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import anyio
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import get_model

from rewardhack.monitors import llm_monitors, spend, summarize
from rewardhack.monitors.reasoning import agent_family
from rewardhack.monitors.spend import Planned
from rewardhack.monitors.transcript import REWRITE_LOCK_FD_ENV, RewriteLocked, check_scored_dir, is_within, open_rewrite_lock
from rewardhack.task import ROOT

# inspect score runs here, so relative scorer paths resolve; it loads the first .env found walking up from here.
COMMAND_DIR = ROOT
LOGS_ROOT = ROOT / "logs"
SCORED_ROOT = LOGS_ROOT / "scored"
PILOT_DIR = LOGS_ROOT / "pilot"
SCORE_ORDER = llm_monitors.SCORERS
M1_SCORER = "src/rewardhack/monitors/m1_regex.py@m1"
LLM_SCORER_FILE = "src/rewardhack/monitors/llm_monitors.py"


class Refused(Exception):
    """The command refuses before doing anything; exit status 2."""


# --- paths ---


def symlinked_dirs(directory: Path) -> list[Path]:
    """Every symlinked directory under a directory (sorted), which rglob does not follow."""
    return sorted(p for p in directory.rglob("*") if p.is_symlink() and p.is_dir())


def eval_files(paths: Sequence[Path]) -> list[Path]:
    """Each file argument, and every .eval under each directory argument (recursive, sorted); a file reached twice is kept once.

    A directory holding a symlinked directory refuses, since rglob would skip it silently.
    """
    found: dict[Path, Path] = {}
    for path in paths:
        if not path.exists():
            raise Refused(f"{path} does not exist")
        if path.is_dir() and (links := symlinked_dirs(path)):
            raise Refused(f"{path} holds symlinked directories, which are not searched: {', '.join(map(str, links))}")
        files = sorted(p for p in path.rglob("*.eval") if p.is_file()) if path.is_dir() else [path]
        for file in files:
            found.setdefault(file.resolve(), file)
    return list(found.values())


def shown(path: Path) -> str:
    """A path as commands show it: relative to the repo root when inside it, else absolute."""
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_scored_root(scored_root: Path) -> None:
    """Refuse a scored root inside the repo's logs/ other than logs/scored/, or one that holds logs/ (the root is a test hook).

    Directories are compared by file identity too (transcript.is_within), so another spelling on a
    case-insensitive file system is still the same directory. Every check compares resolved paths, so
    a scored root that is a symlink, or a logs/scored that is one or resolves anywhere but itself
    (transcript.check_scored_dir), is refused too: linked to logs/main, it would pass agent logs as copies.
    """
    try:
        check_scored_dir(SCORED_ROOT, LOGS_ROOT)
    except ValueError as e:
        raise Refused(str(e)) from e
    if scored_root.is_symlink():
        raise Refused(f"{scored_root} is a symlink; name the scored directory itself")
    if is_within(scored_root, LOGS_ROOT) and not is_within(scored_root, SCORED_ROOT):
        raise Refused(f"{scored_root} is inside {LOGS_ROOT} but not under {SCORED_ROOT}; agent logs are never scored in place")
    if is_within(LOGS_ROOT, scored_root):
        raise Refused(f"{scored_root} contains {LOGS_ROOT}; agent logs are never scored in place")


# --- copy ---


def copy_plan(sources: Sequence[Path], logs_root: Path, scored_root: Path, allow_error: bool) -> list[tuple[Path, Path]]:
    """(source, destination) for every log to copy; raises Refused, before anything is written, on any problem."""
    check_scored_root(scored_root)
    logs_root, scored_root = logs_root.resolve(), scored_root.resolve()
    problems = []
    for source in sources:
        resolved = source.resolve()
        if not resolved.is_relative_to(logs_root):
            problems.append(f"{source} is not under {logs_root}")
        elif resolved.is_relative_to(scored_root):
            problems.append(f"{source} is a scored copy, not an agent log")
        elif scored_root.is_relative_to(resolved):
            problems.append(f"{source} contains {scored_root}; name the agent log directories instead")
    if problems:
        raise Refused("; ".join(problems))
    if scored_root.is_dir() and (links := symlinked_dirs(scored_root)):
        raise Refused(f"{scored_root} holds symlinked directories, which copies would be written through: {', '.join(map(str, links))}")
    pairs = [(file, scored_root / file.resolve().relative_to(logs_root)) for file in eval_files(sources)]
    if not pairs:
        raise Refused("no .eval files in the given sources")
    allowed = {"success", "error"} if allow_error else {"success"}
    for source, destination in pairs:
        if destination.exists() or destination.is_symlink():
            problems.append(f"{destination} exists; each log is copied once")
        elif (resolved := destination.resolve()) != destination:
            problems.append(f"{destination} is reached through a symlink, to {resolved}; copies are written only under {scored_root} itself")
        status = read_eval_log(str(source), header_only=True).status
        if status not in allowed:
            problems.append(f"{source} has status {status}")
    if problems:
        raise Refused("; ".join(problems))
    return pairs


def copy_logs(sources: Sequence[Path], logs_root: Path = LOGS_ROOT, scored_root: Path = SCORED_ROOT, allow_error: bool = False) -> list[Path]:
    """Copy each agent log once into the scored root (see the module docstring); returns the copies."""
    copies = []
    for source, destination in copy_plan(sources, logs_root, scored_root, allow_error):
        destination.parent.mkdir(parents=True, exist_ok=True)
        # copy_plan checked that no symlink is on the destination's path; one made since would redirect the copy.
        if (parent := destination.parent.resolve()) != destination.parent:
            raise RuntimeError(f"{destination.parent} resolves to {parent} through a symlink made after the copy was planned; nothing was copied there")
        partial = destination.with_name(f".{destination.name}.partial")
        shutil.copyfile(source, partial)
        try:
            if sha256_file(partial) != sha256_file(source):
                raise RuntimeError(f"the copy of {source} does not match it")
            os.link(partial, destination)
        finally:
            partial.unlink(missing_ok=True)
        copies.append(destination)
        print(f"copied {shown(source)} -> {shown(destination)}")
    return copies


# --- commands ---


def m1_command(log: str) -> list[str]:
    return ["inspect", "score", log, "--overwrite", "--action", "append", "--model", "mockllm/model", "--scorer", M1_SCORER]


def scorer_arg(name: str, path: str) -> list[str]:
    if "," in path:
        raise Refused(f"the {name} path must not contain a comma (inspect reads -S values with commas as lists)")
    return ["-S", f"{name}={path}"]


def score_command(log: str, scorer: str, labels: str | None = None, subset: str | None = None) -> list[str]:
    """PLAN's inspect score command for one LLM scorer on one scored log.

    labels adds -S labels=<path> for the judge, and subset adds -S subset=<path> for m2 and m3.
    """
    command = [
        "inspect", "score", log, "--overwrite", "--action", "append", "--model", "mockllm/model",
        "--scorer", f"{LLM_SCORER_FILE}@{scorer}", "--model-role", llm_monitors.role_setting(llm_monitors.SCORER_ROLES[scorer]),
    ]
    if labels is not None:
        if scorer != "judge":
            raise ValueError("only the judge takes labels")
        command += scorer_arg("labels", labels)
    if subset is not None:
        if scorer not in llm_monitors.SUBSET_SCORERS:
            raise ValueError(f"only {', '.join(llm_monitors.SUBSET_SCORERS)} take a subset")
        command += scorer_arg("subset", subset)
    return command


def inspect_executable() -> str:
    beside = Path(sys.executable).with_name("inspect")
    found = str(beside) if beside.exists() else shutil.which("inspect")
    if found is None:
        raise Refused("cannot find the inspect command; run this script with uv run")
    return found


def command_env() -> dict[str, str]:
    """This process's environment without INSPECT_SCORE_*, which inspect score reads for any option not given, or a rewrite lock's descriptor."""
    return {key: value for key, value in os.environ.items() if not key.startswith("INSPECT_SCORE_") and key != REWRITE_LOCK_FD_ENV}


# A key python-dotenv reads as INSPECT_SCORE_* (it also accepts export and a single-quoted key).
DOTENV_SCORE_KEY = re.compile(r"\s*(?:export\s+)?'?(INSPECT_SCORE_\w*)'?\s*=")


def dotenv_file(start: Path) -> Path | None:
    """The .env file inspect's init_dotenv loads for a command run in start: find_dotenv(usecwd=True), the first walking up."""
    for directory in (start, *start.parents):
        if (directory / ".env").is_file():
            return directory / ".env"
    return None


def check_dotenv() -> None:
    """Refuse when the .env file inspect score would load sets an INSPECT_SCORE_* key.

    inspect loads that file before it parses its options and does not override variables already set,
    so a key removed by command_env would come back from it (INSPECT_SCORE_STREAM would turn on
    --stream). Only key names are read; no value is kept, used or printed.
    """
    found = dotenv_file(COMMAND_DIR)
    if found is None:
        return
    with found.open(encoding="utf-8", errors="replace") as f:
        keys = sorted({match.group(1) for line in f if (match := DOTENV_SCORE_KEY.match(line))})
    if keys:
        raise Refused(f"remove these inspect score option variables from {found} first: {' '.join(keys)}")


def run_command(command: list[str], lock_fd: int | None = None) -> int:
    """Run one inspect score command from COMMAND_DIR, the repo root (so relative scorer paths resolve); returns its exit status.

    lock_fd, the rewrite lock this script holds for the command's log, is passed to the command, and its
    descriptor number under REWRITE_LOCK_FD_ENV, so the command's scorer holds that lock instead of refusing it.
    """
    env, pass_fds = command_env(), ()
    if lock_fd is not None:
        env[REWRITE_LOCK_FD_ENV], pass_fds = str(lock_fd), (lock_fd,)
    return subprocess.run([inspect_executable(), *command[1:]], cwd=COMMAND_DIR, env=env, pass_fds=pass_fds).returncode


@contextlib.contextmanager
def rewrite_locks(paths: Sequence[Path], dry_run: bool) -> Iterator[dict[Path, int]]:
    """Each scored log's rewrite lock (transcript.open_rewrite_lock) by path, held until the block ends; none for a dry run.

    Refuses, holding none, while another command or scorer holds any of them.
    """
    held: dict[Path, int] = {}
    try:
        for path in [] if dry_run else paths:
            try:
                held[path] = open_rewrite_lock(path)
            except RewriteLocked as e:
                raise Refused(str(e)) from e
        yield held
    finally:
        for fd in held.values():
            os.close(fd)


# --- scoring ---


def scored_logs(paths: Sequence[Path], scored_root: Path) -> list[Path]:
    check_scored_root(scored_root)
    if not paths:
        if not scored_root.is_dir():
            raise Refused(f"{scored_root} does not exist; copy the agent logs first")
        paths = [scored_root]
    found = eval_files(paths)
    outside = [path for path in found if not path.resolve().is_relative_to(scored_root.resolve())]
    if outside:
        raise Refused(f"only scored copies under {scored_root} are scored: {', '.join(map(str, outside))}")
    if not found:
        raise Refused("no scored logs")
    return found


def score_names(log: EvalLog, scorer: str) -> set[str]:
    """The names under which this scorer's scores are stored in the log (scorer, scorer1, scorer2, ...)."""
    pattern = re.compile(rf"^{re.escape(scorer)}(-?\d+)?$")
    return {name for sample in log.samples or [] for name in (sample.scores or {}) if pattern.match(name)}


@dataclass
class Job:
    scorer: str
    path: Path
    command: list[str]
    planned: list[Planned]
    before: set[str]


def check_caps(jobs: Sequence[Job], dry_run: bool) -> bool:
    """Spend so far plus every job's projection, per cap; prints each and returns whether all are within their caps.

    A dry run prints the same and always continues; without a ledger it says so instead of refusing.
    """
    try:
        entries = spend.ledger_entries()
    except spend.LedgerError as e:
        if not dry_run:
            raise
        print(f"no projection: {e}")
        return True
    so_far = spend.spent(entries)
    projected: Counter[str] = Counter()
    for scorer in SCORE_ORDER:
        parts = [part for job in jobs if job.scorer == scorer for part in job.planned]
        if not parts:
            continue
        for projection in spend.project(scorer, parts):
            projected[projection.cap_name] += projection.projected
            print(f"{scorer}: {projection.describe()}")
    within = True
    for cap, amount in projected.items():
        total = so_far[cap] + amount
        ok = total <= spend.CAPS[cap]
        within &= ok
        verdict = "within" if ok else "OVER"
        print(f"{cap} cap ${spend.CAPS[cap]:.2f}: spent ${so_far[cap]:.4f} + all planned runs ${amount:.4f} = ${total:.4f} ({verdict})")
    return within or dry_run


def plan_jobs(paths: Sequence[Path], scorers: Sequence[str], labels: Path, rescore: bool, subset: Path | None = None) -> list[Job]:
    order = [name for name in SCORE_ORDER if name in scorers]
    logs = {path: read_eval_log(str(path)) for path in paths}
    uuids = None
    if subset is not None:
        if extra := [name for name in order if name not in llm_monitors.SUBSET_SCORERS]:
            raise Refused(f"--subset re-scores only {', '.join(llm_monitors.SUBSET_SCORERS)}, not {', '.join(extra)}")
        if not rescore:
            raise Refused("--subset re-scores logs that hold a full run of each scorer, so it needs --rescore")
        uuids = llm_monitors.load_subset(subset)
        found = {sample.uuid for log in logs.values() for sample in log.samples or []}
        if missing := sorted(uuids - found):
            raise Refused(f"{len(missing)} subset uuids are in none of the given logs, such as {', '.join(missing[:3])}")
        unscored = [
            f"{scorer} on {shown(path)}"
            for scorer in order
            for path, log in logs.items()
            if any(sample.uuid in uuids for sample in log.samples or []) and not llm_monitors.full_score(log, scorer)
        ]
        if unscored:
            raise Refused(
                "--subset re-scores beside a full run stored under the scorer's own name, which these lack; "
                f"score them without --subset first: {', '.join(unscored)}"
            )
    # Loaded after the subset checks, which refuse the judge, so a subset run never needs the label rows.
    label_rows = llm_monitors.load_labels(labels) if "judge" in order else None
    labels_arg = None if labels.resolve() == llm_monitors.DEFAULT_LABELS.resolve() else shown(labels)
    target = None
    jobs = []
    for scorer in order:
        for path, log in logs.items():
            if scorer == "m3_summary" and agent_family(log.eval.model) != "gpt-oss":
                continue
            if uuids is not None and not any(sample.uuid in uuids for sample in log.samples or []):
                print(f"skip {scorer} on {shown(path)}: no sample of the subset")
                continue
            before = score_names(log, scorer)
            if before and not rescore:
                print(f"skip {scorer} on {shown(path)}: already scored ({', '.join(sorted(before))}); pass --rescore to score again")
                continue
            if scorer == "m3_summary" and target is None:
                target = summarize.load_target()
                if target.counter != llm_monitors.ROLE_MODELS["summarizer"]:
                    raise Refused(f"the summary target was counted by {target.counter}, not the summarizer {llm_monitors.ROLE_MODELS['summarizer']}")
            planned = llm_monitors.plan(scorer, log, llm_monitors.ROLE_MODELS, labels=label_rows, target=target, subset=uuids)
            command = score_command(shown(path), scorer, labels_arg if scorer == "judge" else None, None if subset is None else shown(subset))
            jobs.append(Job(scorer, path, command, planned, before))
    return jobs


# Statuses of samples a run left unscored without failing it; check_run prints a warning for them.
UNSCORED_STATUSES = ("unparsed", "summary_failed", "no_episode_record")


def check_run(job: Job) -> bool:
    """Whether every sample got the new score and none has status error; prints the statuses and any warnings."""
    log = read_eval_log(str(job.path))
    added = score_names(log, job.scorer) - job.before
    if len(added) != 1:
        print(f"{job.scorer} on {shown(job.path)}: expected one new score name, found {sorted(added)}", file=sys.stderr)
        return False
    [name] = added
    statuses: Counter[str] = Counter()
    over_100k = 0
    for sample in log.samples or []:
        score = (sample.scores or {}).get(name)
        statuses[(score.metadata or {}).get("status", "no_status") if score is not None else "missing"] += 1
        over_100k += score is not None and (score.metadata or {}).get("over_100k_input") is True
    print(f"{job.scorer} on {shown(job.path)} as {name}: {dict(sorted(statuses.items()))}")
    if unscored := {status: statuses[status] for status in UNSCORED_STATUSES if statuses[status]}:
        print(f"warning: {job.scorer} on {shown(job.path)} left samples unscored: {unscored}")
    if over_100k:
        print(f"warning: {job.scorer} on {shown(job.path)}: {over_100k} samples had a prompt over 100K input tokens (higher price, not modeled)")
    return statuses["missing"] == 0 and statuses["error"] == 0


def score_logs(
    paths: Sequence[Path],
    scorers: Sequence[str] | None,
    labels: Path,
    rescore: bool,
    dry_run: bool,
    scored_root: Path = SCORED_ROOT,
    subset: Path | None = None,
) -> int:
    """Plan, check and run the LLM scorers; scorers None means PLAN's order, or m2 and m3 with a subset."""
    check_dotenv()
    if scorers is None:
        scorers = llm_monitors.SUBSET_SCORERS if subset is not None else SCORE_ORDER
    logs = scored_logs(paths, scored_root)
    with rewrite_locks(logs, dry_run) as locks:
        jobs = plan_jobs(logs, scorers, labels, rescore, subset)
        if not jobs:
            print("nothing to score")
            return 0
        if not check_caps(jobs, dry_run):
            print("refused before any call: the planned runs would cross a cap", file=sys.stderr)
            return 1
        ok = True
        for job in jobs:
            print(shlex.join(job.command))
            if dry_run:
                continue
            if run_command(job.command, locks[job.path]) != 0:
                print(f"{job.scorer} on {shown(job.path)} failed; stopping", file=sys.stderr)
                return 1
            ok &= check_run(job)
    return 0 if ok else 1


def m1_logs(paths: Sequence[Path], rescore: bool, dry_run: bool, scored_root: Path = SCORED_ROOT) -> int:
    check_dotenv()
    logs = scored_logs(paths, scored_root)
    with rewrite_locks(logs, dry_run) as locks:
        for path in logs:
            if score_names(read_eval_log(str(path)), "m1") and not rescore:
                print(f"skip m1 on {shown(path)}: already scored; pass --rescore to score again")
                continue
            command = m1_command(shown(path))
            print(shlex.join(command))
            if not dry_run and run_command(command, locks[path]) != 0:
                print(f"m1 on {shown(path)} failed; stopping", file=sys.stderr)
                return 1
    return 0


# --- the summary target ---


def summary_target(paths: Sequence[Path], out: Path, counter_model: str, dry_run: bool) -> int:
    logs = [(shown(path), read_eval_log(str(path))) for path in eval_files(paths)]
    texts = summarize.target_texts(log for _, log in logs)
    used = [name for name, log in logs if log.eval.model == summarize.TARGET_AGENT]
    print(f"{len(texts)} kept reasoning blocks in {len(used)} {summarize.TARGET_AGENT} logs: {', '.join(used)}")
    if out.exists():
        raise Refused(f"{out} exists; the summary target is frozen")
    if dry_run:
        print(f"would count {len(texts)} blocks with {counter_model} count_tokens and write {shown(out)}")
        return 0
    counter = summarize.ModelTokenCounter(get_model(counter_model))
    target = anyio.run(summarize.compute_target, logs, counter)
    summarize.write_target(target, out)
    print(f"wrote {shown(out)}: median {target.tokens} tokens, {target.words} words over {target.blocks} blocks")
    return 0


# --- command line ---


def main(argv: Sequence[str] | None = None) -> int:
    # Line-buffer stdout so this script's lines stay next to each inspect score run's output when redirected.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    copy = commands.add_parser("copy", help="copy agent logs once into logs/scored/")
    copy.add_argument("sources", nargs="+", type=Path)
    copy.add_argument("--allow-error", action="store_true", help="also copy logs whose status is error")
    m1 = commands.add_parser("m1", help="run M1 on scored copies")
    m1.add_argument("logs", nargs="*", type=Path)
    score = commands.add_parser("score", help="run the LLM scorers on scored copies, in PLAN's order")
    score.add_argument("logs", nargs="*", type=Path)
    score.add_argument("--scorers", nargs="+", choices=SCORE_ORDER, help="default: all, in PLAN's order (m2 m3 with --subset)")
    score.add_argument("--labels", type=Path, default=llm_monitors.DEFAULT_LABELS, help="grade_hidden's labels.jsonl, for the judge")
    score.add_argument("--subset", type=Path, help="a file of sample uuids: re-score only these with m2 and m3 (test-retest)")
    for sub in (m1, score):
        sub.add_argument("--rescore", action="store_true", help="score again a log that already has this scorer")
        sub.add_argument("--dry-run", action="store_true", help="print what would run and run nothing")
        sub.add_argument("--scored-root", type=Path, default=SCORED_ROOT, help=argparse.SUPPRESS)
    target = commands.add_parser("summary-target", help="compute and freeze the summarizer's length target")
    target.add_argument("logs", nargs="*", type=Path, default=[PILOT_DIR])
    target.add_argument("--out", type=Path, default=summarize.TARGET_FILE)
    target.add_argument("--counter-model", default=summarize.TARGET_AGENT)
    target.add_argument("--dry-run", action="store_true")
    copy.add_argument("--logs-root", type=Path, default=LOGS_ROOT, help=argparse.SUPPRESS)
    copy.add_argument("--scored-root", type=Path, default=SCORED_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.command == "copy":
            copy_logs(args.sources, args.logs_root, args.scored_root, args.allow_error)
            return 0
        if args.command == "m1":
            return m1_logs(args.logs, args.rescore, args.dry_run, args.scored_root)
        if args.command == "score":
            return score_logs(args.logs, args.scorers, args.labels, args.rescore, args.dry_run, args.scored_root, args.subset)
        return summary_target(args.logs or [PILOT_DIR], args.out, args.counter_model, args.dry_run)
    except (Refused, spend.LedgerError, summarize.SummaryTargetMissing, FileExistsError, FileNotFoundError, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
