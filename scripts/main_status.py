"""Progress and spend of the full runs (PLAN.md, "Build order" step 4), read from their logs at any time.

Usage: uv run python scripts/main_status.py [--logs logs/main] [--rejected-logs logs/main-rejected]
    [--pilot-logs logs/pilot] [--earlier-logs logs/pilot-rejected] [--smoke-logs logs/smoke-live] [--out DIR]

The full runs are scripts/run_main.sh's six parts in PLAN's order (Haiku, gpt-oss, Sonnet, normal before
conflict), each an eval-set log directory <logs>/<agent>-<variant> over the 8 tasks with PLAN's epochs.
For each part this reports the status of its newest log (by modification time, as eval-set picks the log it
resumes from), its completed, errored and missing samples against PLAN's epochs times the 8 tasks, its samples'
error retries, how many logs the directory holds, and any sign that the newest log replaced a failed attempt.
A part has finished when its newest log has status success, is not invalidated and holds every expected sample
and epoch.
An errored sample of a finished log stays errored, since eval-set does not re-run the samples of a log it counts
as complete, and PLAN counts such samples per agent and variant and excludes them.
Spend per agent comes from pilot_report.read_spend, which counts the larger of each log's header and its samples,
so the spend of retried attempts that Inspect kept is counted, and lists killed or replaced logs, whose spend is a
lower bound; an agent's spend is that of its two parts' logs plus that of the logs of its model under
--rejected-logs, where scripts/run_main.sh tells the operator to move a part after a harness change.
A log that a crashed attempt left beside the log that resumed it counts in full, and so does every sample the
later log reused from it, so such a part's spend can be an over-count.
The caps are PLAN's Budget hard caps: Haiku $28.80, Sonnet $80.00, and gpt-oss $15.50 less what the pilot
(--pilot-logs), its rejected attempts (--earlier-logs) and the live smoke test (--smoke-logs) spent on OpenRouter.
A cap is crossed when the spend counted against it is above it; a lower bound below its cap is reported as one
and crosses nothing.
The report also says what holds run_main.sh's lock in --logs: a running launcher, an eval-set that a killed
launcher left running (still spending), or a lock no process holds, which run_main.sh refuses to start beside
until an operator removes it; it suggests resuming only when there is no lock, and only the agents whose parts
launch_problems would let run_main.sh start, naming the others with its reasons.
scripts/run_main.sh calls launch_problems, through launch_refusal, for every selected agent before a real run and again
for the next part's agent before each later part, so a part never starts while its agent's spend so far, or that
spend plus its samples still to run at their full cost limit, is above the agent's cap.

This script only reads logs: it calls no model, runs no agent code, launches no run and never changes a log.
It prints report.md, and with --out it also writes report.md and report.json to that directory, the only place
it writes.
Exit status: 0 when every part has finished and no cap is crossed; 1 when a part has not finished or a cap is
crossed; 2 on bad input, when it writes nothing, or when --out cannot be written.
Bad input is a log path that exists but is not a directory, a log or its sample summaries that cannot be read
(as when eval-set removes an older log while this script reads it; run it again), an Inspect log in JSON format
under any of the five log directories (only .eval logs are read), a log under --logs outside the
six part directories (whose spend would count against no cap), a log in a part's directory that another command
wrote (another model, variant, epochs, task selection or per-sample cost limit), a sample outside the part's tasks
and epochs, spend on a provider that is not the part's agent's, a log under --rejected-logs with paid spend whose
model is no agent's, or a paid model usage without a total_cost.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pilot_report
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log, read_eval_log_sample_summaries
from pilot_report import AGENT_CAP, AGENT_MODELS, AGENTS, PROVIDERS, VARIANTS, BadInput, plural, provider_of, table, usd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS = ROOT / "logs/main"
# scripts/run_main.sh: after a harness change a part's log directory moves "to logs/main-rejected/".
DEFAULT_REJECTED_LOGS = ROOT / "logs/main-rejected"
LOCK = ".run_main.lock"

# PLAN.md "Tasks: 8 CWEval seeds" and scripts/launch_lib.sh's main SAMPLES, which run_main.sh passes as --sample-id.
TASKS = ("022_0", "022_2", "078_0", "327_0", "377_0", "732_2", "918_0", "943_0")
# PLAN.md "Runs" table: epochs per agent and variant.
EPOCHS = pilot_report.FULL_RUN_EPOCHS
# PLAN.md "Harness": the per-sample --cost-limit of each agent, which scripts/launch_lib.sh passes.
COST_LIMITS = {"haiku": 0.10, "gpt-oss": 0.05, "sonnet": 1.00}
# scripts/run_main.sh's order: Haiku, gpt-oss, Sonnet, normal before conflict.
PARTS = tuple((agent, variant) for agent in AGENTS for variant in VARIANTS)
AGENT_NAMES = {"haiku": "Haiku", "gpt-oss": "gpt-oss", "sonnet": "Sonnet"}
# A spend counts as above a cap only past this many dollars, so a float sum that lands on a cap is within it.
CAP_TOLERANCE = 1e-9
CLAUDE_MD_APPROVAL = (
    "CLAUDE.md: any spend above the plan's caps needs Jessup's approval first, so no further run of an agent whose cap "
    "is crossed may start without it."
)
# What scripts/run_main.sh prints after the agents launch_problems refuses.
LAUNCH_APPROVAL = (
    "CLAUDE.md requires Jessup's approval before any spend above a cap, and an approved cap goes into PLAN's Budget "
    "and AGENT_CAP in scripts/pilot_report.py; scripts/main_status.py reports the spend so far"
)

assert len(TASKS) == pilot_report.FULL_RUN_TASKS


def cap_usd(amount: float) -> str:
    """A cap in dollars: to the cent when it is a whole number of cents, else to 4 places, so a cap less earlier spend
    (gpt-oss's) never prints below a 4-place spend that it is not below."""
    return usd(amount) if abs(amount * 100 - round(amount * 100)) < 1e-6 else usd(amount, 4)


def part_name(agent: str, variant: str) -> str:
    return f"{agent}-{variant}"


def sample_ids(config_ids: object) -> list[str]:
    """A log's --sample-id selection as sorted strings (Inspect keeps a list, a single id or None)."""
    if config_ids is None:
        return []
    return sorted(str(i) for i in (config_ids if isinstance(config_ids, list) else [config_ids]))


def config_problems(path: Path, header: EvalLog, agent: str, variant: str, epochs: int) -> list[str]:
    """Why a log in a part's directory is not one of that part's, as scripts/run_main.sh starts it."""
    found = {
        "model": header.eval.model,
        "variant": (header.eval.task_args or {}).get("variant"),
        "epochs": header.eval.config.epochs,
        "sample_id": sample_ids(header.eval.config.sample_id),
        "cost_limit": header.eval.config.cost_limit,
    }
    wanted = {
        "model": AGENT_MODELS[agent], "variant": variant, "epochs": epochs, "sample_id": sorted(TASKS),
        "cost_limit": COST_LIMITS[agent],
    }
    return [
        f"{path}: {what} {found[what]!r} does not match the {part_name(agent, variant)} part's {wanted[what]!r}"
        for what in wanted
        if found[what] != wanted[what]
    ]


def first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def read_part(logs_root: Path, agent: str, variant: str, problems: list[str]) -> dict:
    """One part's status from its newest log; problems collects anything that makes the input bad."""
    log_dir = logs_root / part_name(agent, variant)
    epochs = EPOCHS[agent][variant]
    expected = {(task, epoch) for task in TASKS for epoch in range(1, epochs + 1)}
    part = {
        "agent": agent, "variant": variant, "log_dir": str(log_dir), "epochs": epochs, "expected": len(expected),
        "status": "not started", "finished": False, "logs": [], "newest": None, "completed": 0, "errored": 0,
        "missing": len(expected), "sample_retries": 0, "limits": {}, "reused": 0, "rerun": 0,
        "harness_sha256": [], "errored_samples": [],
    }
    if not log_dir.exists():
        return part
    if not log_dir.is_dir():
        problems.append(f"{log_dir} is not a directory of logs")
        return part
    headers = []
    for path in sorted(log_dir.rglob("*.eval")):
        try:
            mtime = path.stat().st_mtime
            header = read_eval_log(str(path), header_only=True)
        except Exception as e:
            problems.append(f"{path}: cannot be read ({type(e).__name__}: {e})")
            continue
        problems += config_problems(path, header, agent, variant, epochs)
        headers.append((mtime, path, header))
        part["logs"].append({"path": str(path), "status": header.status, "eval_id": header.eval.eval_id})
    part["harness_sha256"] = sorted({str((h.eval.metadata or {}).get("harness_sha256")) for _, _, h in headers})
    if not headers:
        return part
    _, path, header = max(headers, key=lambda item: (item[0], str(item[1])))
    try:
        summaries = read_eval_log_sample_summaries(str(path))
    except Exception as e:
        problems.append(f"{path}: its sample summaries cannot be read ({type(e).__name__}: {e})")
        return part
    found = {(str(s.id), s.epoch) for s in summaries}
    unexpected = sorted(found - expected)
    if unexpected:
        problems.append(
            f"{path}: holds samples outside the {part_name(agent, variant)} part's {len(TASKS)} tasks and {epochs} "
            f"epochs: {', '.join(f'{task} epoch {epoch}' for task, epoch in unexpected)}"
        )
    errored = [s for s in summaries if s.error is not None]
    reused, rerun = pilot_report.replaced_signs(header, [(s.started_at, s.retries or 0) for s in summaries])
    part.update(
        status=header.status + (" (invalidated)" if header.invalidated else ""),
        newest=str(path),
        completed=len(found & expected) - len(errored),
        errored=len(errored),
        missing=len(expected - found),
        sample_retries=sum(s.retries or 0 for s in summaries),
        limits=dict(sorted(Counter(s.limit for s in summaries if s.limit).items())),
        reused=reused,
        rerun=rerun,
        errored_samples=[
            {"id": str(s.id), "epoch": s.epoch, "error": first_line(s.error or "")}
            for s in sorted(errored, key=lambda s: (str(s.id), s.epoch))
        ],
    )
    part["finished"] = header.status == "success" and not header.invalidated and part["missing"] == 0
    return part


def pid_in(path: Path) -> int | None:
    """The pid a lock file records, or None when the file is missing or holds no positive number."""
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if pid > 0 else None


def alive(pid: int | None) -> bool:
    """Whether a process with this pid exists, as the launcher's kill -0 sees it."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def lock_state(logs_root: Path) -> dict:
    """run_main.sh's lock in the log root, read as scripts/launch_lib.sh's refuse_held_lock reads it (never touched).

    state is "none" (no lock), "running" (its launcher is alive), "orphaned eval-set" (the launcher is gone but the
    eval-set it started is alive, and may still be spending), "stale" (neither is alive) or "no launcher pid".
    """
    lock = logs_root / LOCK
    launcher, eval_set = pid_in(lock / "launcher"), pid_in(lock / "eval-set")
    if not lock.exists():
        state = "none"
    elif launcher is None:
        state = "no launcher pid"
    elif alive(launcher):
        state = "running"
    elif alive(eval_set):
        state = "orphaned eval-set"
    else:
        state = "stale"
    return {"path": str(lock), "state": state, "launcher": launcher, "eval_set": eval_set}


def lock_line(report: dict) -> str:
    """One sentence on what holds the lock in the log root."""
    lock, logs = report["lock"], report["logs"]
    by_hand = (
        f"run_main.sh refuses to start until someone who has checked that no run_main.sh or eval-set is running on {logs} "
        "removes it"
    )
    return {
        "none": f"No run_main.sh holds the lock in {logs}.",
        "running": f"A run_main.sh (pid {lock['launcher']}) holds the lock in {logs}.",
        "orphaned eval-set": (
            f"The eval-set (pid {lock['eval_set']}) that an earlier run_main.sh (pid {lock['launcher']}) started is still "
            f"running on {logs} and may still be spending; run_main.sh refuses to start while it runs, and after it ends "
            f"until someone removes {lock['path']}."
        ),
        "stale": f"{lock['path']} is the stale lock of run_main.sh pid {lock['launcher']}, which is no longer running; {by_hand}.",
        "no launcher pid": f"{lock['path']} exists without a launcher pid; {by_hand}.",
    }[lock["state"]]


def spend_section(args: argparse.Namespace) -> dict:
    """Each agent's spend in its parts and rejected logs, against its hard cap (gpt-oss's less earlier spend)."""
    seen: set[str] = set()
    parts = {(agent, variant): pilot_report.read_spend(args.logs / part_name(agent, variant), seen) for agent, variant in PARTS}
    rejected = {agent: pilot_report.read_spend(args.rejected_logs, seen, model=AGENT_MODELS[agent]) for agent in AGENTS}
    unattributed = pilot_report.read_spend(args.rejected_logs, seen)
    before = {
        "pilot": pilot_report.read_spend(args.pilot_logs, seen),
        "pilot_rejected": pilot_report.read_spend(args.earlier_logs, seen),
        "smoke": pilot_report.read_spend(args.smoke_logs, seen),
    }
    problems = []
    if any(unattributed["spend"].values()):
        problems.append(
            f"{args.rejected_logs} holds logs with paid spend whose model is no agent's, so their spend has no cap: "
            f"{usd(unattributed['spend']['anthropic'], 4)} on Anthropic and {usd(unattributed['spend']['openrouter'], 4)} on OpenRouter"
        )
    agents = {}
    for agent in AGENTS:
        provider = provider_of(AGENT_MODELS[agent])
        counted = {(agent, variant): parts[agent, variant] for variant in VARIANTS} | {(agent, "rejected"): rejected[agent]}
        for (_, where), found in counted.items():
            for other in PROVIDERS:
                if other != provider and found["spend"][other]:
                    problems.append(
                        f"{agent} {where} logs spent {usd(found['spend'][other], 4)} on {other}, which is not {agent}'s provider"
                    )
        main = {variant: parts[agent, variant]["spend"][provider] for variant in VARIANTS}
        spent_before = sum(found["spend"]["openrouter"] for found in before.values()) if agent == "gpt-oss" else 0.0
        total = sum(main.values()) + rejected[agent]["spend"][provider]
        cap_left = AGENT_CAP[agent] - spent_before
        lower_bound = [record for found in counted.values() for record in found["incomplete"]]
        if agent == "gpt-oss":
            lower_bound += [record for found in before.values() for record in found["incomplete"] if "openrouter" in record["providers"]]
        agents[agent] = {
            "provider": provider,
            "parts": main,
            "rejected": rejected[agent]["spend"][provider],
            "rejected_logs": rejected[agent]["logs"],
            "spend": total,
            "cap": AGENT_CAP[agent],
            "spent_before": {name: found["spend"]["openrouter"] for name, found in before.items()} if agent == "gpt-oss" else {},
            "cap_left": cap_left,
            "crossed": total - cap_left > CAP_TOLERANCE,
            "lower_bound": lower_bound,
        }
    if problems:
        raise BadInput(problems)
    return {"agents": agents, "dirs": {name: found["dir"] for name, found in before.items()} | {"rejected": str(args.rejected_logs)}}


# eval-set's own files in a log directory, which are not logs.
EVAL_SET_FILES = {"logs.json", "eval-set.json"}


def json_logs(directory: Path) -> list[str]:
    """The Inspect logs in JSON format under a directory: those Inspect lists (a timestamped *.json name) and any
    other *.json but eval-set's own files whose header reads as an Inspect log.

    The parts and pilot_report.read_spend read only *.eval logs, so a JSON log's samples and spend would be missed.
    """
    if not directory.is_dir():
        return []
    listed = [info.name.removeprefix("file://") for info in list_eval_logs(str(directory), formats=["json"])]
    # Inspect lists every *.eval file whatever the formats asked for.
    found = {str(Path(name).resolve()) for name in listed if name.endswith(".json")}
    for path in directory.rglob("*.json"):
        if path.name in EVAL_SET_FILES or str(path.resolve()) in found:
            continue
        try:
            read_eval_log(str(path), header_only=True)
        except Exception:
            continue
        found.add(str(path.resolve()))
    return sorted(found)


def build_report(args: argparse.Namespace) -> dict:
    if args.logs.exists() and not args.logs.is_dir():
        raise BadInput([f"{args.logs} is not a directory of logs"])
    problems: list[str] = [
        f"{path} is an Inspect log in JSON format, whose samples and spend this script does not read; "
        "scripts/run_main.sh refuses INSPECT_LOG_FORMAT and writes only .eval logs"
        for directory in (args.logs, args.rejected_logs, args.pilot_logs, args.earlier_logs, args.smoke_logs)
        for path in json_logs(directory)
    ]
    part_dirs = [args.logs / part_name(agent, variant) for agent, variant in PARTS]
    problems += [
        f"{path} is under {args.logs} but in none of its six part directories, so its spend would count against no "
        f"cap; move it into {args.rejected_logs}"
        for path in (sorted(args.logs.rglob("*.eval")) if args.logs.is_dir() else [])
        if not any(part_dir in path.parents for part_dir in part_dirs)
    ]
    parts = [read_part(args.logs, agent, variant, problems) for agent, variant in PARTS]
    if problems:
        raise BadInput(problems)
    spend = spend_section(args)
    for part in parts:
        part["spend"] = spend["agents"][part["agent"]]["parts"][part["variant"]]
    unfinished = [part_name(p["agent"], p["variant"]) for p in parts if not p["finished"]]
    crossed = [agent for agent, s in spend["agents"].items() if s["crossed"]]
    harnesses = sorted({h for p in parts for h in p["harness_sha256"]})
    return {
        "logs": str(args.logs),
        "lock": lock_state(args.logs),
        "parts": parts,
        "unfinished": unfinished,
        "spend": spend,
        "caps_crossed": crossed,
        "harness_sha256": harnesses,
        "exit_status": 1 if unfinished or crossed else 0,
    }


def launch_problems(argv: list[str], agents: list[str]) -> list[str]:
    """Why scripts/run_main.sh must not start these agents' parts, one line per agent it refuses.

    The spend so far is build_report's for this script's arguments argv (run_main.sh passes only --logs, so the
    other directories are the defaults), and bad input raises BadInput as there; refusals has the rules.
    """
    return list(refusals(build_report(build_parser().parse_args(argv)), agents).values())


def refusals(report: dict, agents: list[str]) -> dict[str, str]:
    """Each of these agents whose parts scripts/run_main.sh must not start, with why, from a build_report report.

    An agent is refused when its spend so far is above its cap, or when it plus every sample its parts still have to run
    at the agent's full per-sample cost limit is: PLAN's Budget sets the Haiku and Sonnet caps as samples times that
    limit, so this is the most a run can add while its samples keep within their limits.
    A finished part runs no sample, and an unfinished one runs its missing and errored samples, as eval-set reuses
    the rest.
    A lower bound (a killed log, or one that replaced a failed attempt) is checked as if it were exact, since how much
    spend no log kept is unknown; the refusal calls such spend a lower bound, and the provider's usage records have
    the full figure.
    """
    refused = {}
    for agent in agents:
        spend = report["spend"]["agents"][agent]
        spent = usd(spend["spend"], 4) + (" (a lower bound)" if spend["lower_bound"] else "")
        to_run = sum(p["expected"] - p["completed"] for p in report["parts"] if p["agent"] == agent and not p["finished"])
        most = spend["spend"] + to_run * COST_LIMITS[agent]
        if spend["crossed"]:
            refused[agent] = f"{AGENT_NAMES[agent]} has spent {spent}, above its {cap_usd(spend['cap_left'])} cap"
        elif most - spend["cap_left"] > CAP_TOLERANCE:
            refused[agent] = (
                f"{AGENT_NAMES[agent]} has spent {spent}, and its {plural(to_run, 'sample')} still to run at up to "
                f"{usd(COST_LIMITS[agent])} each could bring it to {usd(most, 4)}, above its {cap_usd(spend['cap_left'])} cap"
            )
    return refused


def launch_refusal(argv: list[str], agents: list[str]) -> str | None:
    """What scripts/run_main.sh prints after its name when it must not start these agents' parts, or None when it may.

    argv is launch_problems'. scripts/launch_lib.sh passes only --logs, for every selected agent in the preflight of a
    real run and again for the next part's agent before each later part, so a part that took its agent past what
    launch_problems allows stops that agent's next part.
    """
    try:
        over = launch_problems(argv, agents)
    except BadInput as e:
        return f"scripts/main_status.py cannot count the spend so far: {'; '.join(e.problems)}"
    return f"{'; '.join(over)}; {LAUNCH_APPROVAL}" if over else None


def join_and(items: list[str]) -> str:
    """The items joined as in "a", "a and b" or "a, b and c"."""
    return " and ".join([", ".join(items[:-1]), items[-1]] if len(items) > 1 else items)


def describe(part: dict) -> str:
    """Why a part has not finished, in one sentence."""
    name = part_name(part["agent"], part["variant"])
    if part["status"] == "not started":
        return f"Part {name} has not started: {part['log_dir']} holds no log."
    counts = f"{part['completed']} of {part['expected']} samples completed, {part['errored']} errored and {part['missing']} missing"
    if part["status"] == "started":
        return f"Part {name} is running or was stopped while running (its newest log has status started): {counts}."
    return f"Part {name} has not finished (its newest log has status {part['status']}): {counts}."


def render_markdown(report: dict) -> str:
    """report.md: the status first, then the parts and the spend; one sentence per line and no em dashes."""
    finished = not report["unfinished"]
    crossed = report["caps_crossed"]
    errored = sum(p["errored"] for p in report["parts"])
    if finished and not crossed and not errored:
        verdict = "every part finished with every sample, and no cap is crossed"
    elif finished and not crossed:
        verdict = f"every part finished, with {plural(errored, 'errored sample')} excluded, and no cap is crossed"
    else:
        reasons = []
        if not finished:
            reasons.append(f"{plural(len(report['unfinished']), 'part')} of {len(PARTS)} not finished")
        if crossed:
            reasons.append(f"the cap of {', '.join(AGENT_NAMES[a] for a in crossed)} crossed")
        verdict = " and ".join(reasons)
    lines = ["# Full-run status", "", f"**Status: {verdict}.**", "", lock_line(report)]
    lines += [describe(p) for p in report["parts"] if not p["finished"]]
    lines += [
        f"{AGENT_NAMES[agent]} spent {usd(s['spend'], 4)}, above its {cap_usd(s['cap_left'])} cap."
        for agent, s in report["spend"]["agents"].items() if s["crossed"]
    ]
    if crossed:
        lines.append(CLAUDE_MD_APPROVAL)
    if report["unfinished"] and report["lock"]["state"] == "none":
        # Only the agents run_main.sh would start, by the same rules as its check before each part.
        waiting = [agent for agent in AGENTS if any(p["agent"] == agent and not p["finished"] for p in report["parts"])]
        refused = refusals(report, waiting)
        allowed = [agent for agent in waiting if agent not in refused]
        if allowed:
            lines.append(
                f"To start or resume the unfinished parts of {join_and([AGENT_NAMES[a] for a in allowed])}, "
                f'run scripts/run_main.sh with AGENTS="{" ".join(allowed)}"; eval-set runs only their missing or errored samples.'
            )
        lines += [f"run_main.sh refuses to start the unfinished parts of {AGENT_NAMES[a]}: {why}." for a, why in refused.items()]
        if refused:
            lines.append(
                "Starting them needs Jessup's approval of a higher cap first, which then goes into PLAN's Budget and "
                "AGENT_CAP in scripts/pilot_report.py."
            )
    lines += ["", "## Parts", ""]
    rows = [
        [
            part_name(p["agent"], p["variant"]), p["status"], len(p["logs"]), f"{p['completed']} of {p['expected']}",
            p["errored"], p["missing"], p["sample_retries"], ", ".join(f"{k} {n}" for k, n in p["limits"].items()) or "none",
            usd(p["spend"], 4),
        ]
        for p in report["parts"]
    ]
    lines += table(
        ["Part", "Newest log", "Logs", "Completed", "Errored", "Missing", "Sample retries", "Limit hits", "Spend"], rows
    )
    lines += [
        "Completed samples are those without an error, limit hits included, since PLAN grades an episode that hit a limit like any other.",
        "An errored sample of a finished part is not re-run by eval-set; PLAN counts it per agent and variant and excludes it.",
    ]
    for p in report["parts"]:
        name = part_name(p["agent"], p["variant"])
        distinct = len({log["eval_id"] for log in p["logs"]})
        if distinct > 1:
            lines.append(
                f"Part {name} holds {plural(distinct, 'distinct log')}; each is counted in full, so samples a later attempt "
                "reused from an earlier one count twice, an over-count."
            )
        copies = len(p["logs"]) - distinct
        if copies:
            lines.append(
                f"Part {name} holds {copies} {'copy' if copies == 1 else 'copies'} of another of its logs with the same "
                "eval_id (such as a log Inspect recovered from a crashed one), counted once."
            )
        signs = [f"{plural(n, 'sample')} {how} a failed earlier attempt" for n, how in ((p["reused"], "copied from"), (p["rerun"], "re-run after")) if n]
        if signs:
            lines.append(f"Part {name}'s newest log holds {' and '.join(signs)}, so that attempt's spend may be in no log.")
        lines += [f"Part {name}: {s['id']} epoch {s['epoch']} errored: {s['error']}" for s in p["errored_samples"]]
    if len(report["harness_sha256"]) > 1:
        lines.append(f"The parts' logs record {len(report['harness_sha256'])} harnesses: {', '.join(report['harness_sha256'])}.")
    lines += ["", "## Spend", ""]
    agents = report["spend"]["agents"]
    rows = [
        [
            AGENT_NAMES[agent], usd(s["parts"]["normal"], 4), usd(s["parts"]["conflict"], 4), usd(s["rejected"], 4),
            usd(s["spend"], 4) + (" (a lower bound)" if s["lower_bound"] else ""), cap_usd(s["cap_left"]),
            "yes" if s["crossed"] else "no",
        ]
        for agent, s in agents.items()
    ]
    lines += table(["Agent", "Normal", "Conflict", "Rejected logs", "Counted", "Cap", "Crossed"], rows)
    gpt_oss = agents["gpt-oss"]
    dirs = report["spend"]["dirs"]
    lines += [
        f"Rejected logs are those of each agent's model under {dirs['rejected']}.",
        f"The gpt-oss cap is PLAN's {usd(gpt_oss['cap'])} less {usd(gpt_oss['cap'] - gpt_oss['cap_left'], 4)} already spent on "
        f"OpenRouter: {usd(gpt_oss['spent_before']['pilot'], 4)} by the pilot ({dirs['pilot']}), "
        f"{usd(gpt_oss['spent_before']['pilot_rejected'], 4)} by its rejected attempts ({dirs['pilot_rejected']}) and "
        f"{usd(gpt_oss['spent_before']['smoke'], 4)} by the live smoke test ({dirs['smoke']}).",
        "The Haiku and Sonnet caps are PLAN's Budget hard caps for the full runs alone.",
        "",
        "### Spend that Inspect did not keep",
        "",
        "A log that never finished (status started) counts only the samples it wrote, and a log that replaced a failed "
        "attempt may leave that attempt's calls in no log, so the spend counted from either is a lower bound.",
        "A lower bound below its cap crosses nothing here; the provider's own usage records have the full figure.",
        "",
    ]
    records = {record["path"]: record for s in agents.values() for record in s["lower_bound"]}
    lines += [pilot_report.incomplete_line(record) for record in records.values()] or [
        "Every log counted finished, and none shows a replaced attempt."
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--logs", type=Path, default=DEFAULT_LOGS,
        help=f"the full runs' log root, holding one directory per part (default {DEFAULT_LOGS.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--rejected-logs", type=Path, default=DEFAULT_REJECTED_LOGS,
        help="directory of full-run logs moved aside after a harness change, whose spend counts against the cap of "
        f"the agent whose model they ran (default {DEFAULT_REJECTED_LOGS.relative_to(ROOT)}; none if it does not exist)",
    )
    parser.add_argument(
        "--pilot-logs", type=Path, default=pilot_report.DEFAULT_LOGS,
        help=f"the pilot's logs, whose OpenRouter spend comes off the gpt-oss cap (default {pilot_report.DEFAULT_LOGS.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--earlier-logs", type=Path, default=pilot_report.DEFAULT_EARLIER_LOGS,
        help="the rejected pilot attempts' logs, whose OpenRouter spend comes off the gpt-oss cap "
        f"(default {pilot_report.DEFAULT_EARLIER_LOGS.relative_to(ROOT)}; none if it does not exist)",
    )
    parser.add_argument(
        "--smoke-logs", type=Path, default=pilot_report.DEFAULT_SMOKE_LOGS,
        help="the live smoke test's logs, whose OpenRouter spend comes off the gpt-oss cap "
        f"(default {pilot_report.DEFAULT_SMOKE_LOGS.relative_to(ROOT)}; none if it does not exist)",
    )
    parser.add_argument("--out", type=Path, help="directory to write report.md and report.json to (default: none)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_report(args)
    except BadInput as e:
        print("Bad input, so no report was written:", *e.problems, sep="\n  ", file=sys.stderr)
        return 2
    markdown = render_markdown(report)
    if args.out is not None:
        try:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            (args.out / "report.md").write_text(markdown, encoding="utf-8")
        except OSError as e:
            print(f"Cannot write the report to {args.out}: {e}", file=sys.stderr)
            return 2
    print(markdown, end="")
    return report["exit_status"]


if __name__ == "__main__":
    sys.exit(main())
