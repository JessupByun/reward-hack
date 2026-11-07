"""scripts/main_status.py on constructed full-run logs in tmp_path.

Each test writes the six parts' logs (8 tasks with PLAN's epochs, at a small cost per sample) as eval-set would, then
changes one thing: a part not started, killed or short of samples, errored or retried samples, a second log in a
part, rejected logs, earlier OpenRouter spend, a cap crossed, a stray log or the state of run_main.sh's lock.
The launch_problems and launch_refusal tests check what scripts/run_main.sh refuses before a real run and before each
later part.
Every log directory the script reads is under tmp_path, so the repository's own logs are never read or written.
"""

import json
import os
from pathlib import Path

import main_status
import pilot_report
import pytest
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalError,
    EvalLog,
    EvalRetryError,
    EvalSample,
    EvalSampleLimit,
    EvalSpec,
    EvalStats,
    read_eval_log,
    write_eval_log,
)
from inspect_ai.model import ModelUsage
from test_pilot_report import kill
from test_run_pilot import dead_pid

TASKS = list(main_status.TASKS)
MODELS = pilot_report.AGENT_MODELS
PARTS = main_status.PARTS
EPOCHS = main_status.EPOCHS
HARNESS = "a" * 64
LOG_START = "2026-11-06T01:00:00+00:00"
SAMPLE_START = "2026-11-06T01:00:05+00:00"
EARLIER = "2026-11-06T00:30:00+00:00"
COST = {"haiku": 0.001, "gpt-oss": 0.0005, "sonnet": 0.01}


def sample(agent: str, task: str, epoch: int, cost: float, error: bool = False, retries: int = 0,
           limit: str | None = None, started_at: str = SAMPLE_START) -> EvalSample:
    model = MODELS[agent]
    return EvalSample(
        id=task, epoch=epoch, uuid=f"{agent}-{task}-{epoch}", input="Fix the bug.", target="",
        model_usage={model: ModelUsage(input_tokens=100, output_tokens=10, total_tokens=110, total_cost=cost)},
        error=EvalError(message="RuntimeError: boom\nmore", traceback="", traceback_ansi="") if error else None,
        error_retries=[EvalRetryError(message="RuntimeError: retried", traceback="", traceback_ansi="")] * retries or None,
        limit=EvalSampleLimit(type=limit, limit=30) if limit else None,
        started_at=started_at,
    )


def write_part(
    logs, agent: str, variant: str, *, name: str = "part.eval", status: str = "success", skip: int = 0,
    errored: int = 0, retries: int = 0, limit: int = 0, reused: int = 0, cost: float | None = None,
    stats_cost: float | None = None, epochs: int | None = None, model: str | None = None, variant_arg: str | None = None,
    sample_id: list[str] | None = None, eval_id: str | None = None, extra: list[tuple[str, int]] | None = None,
    cost_limit: float | None | str = "the agent's", invalidated: bool = False, extra_usage: dict[str, float] | None = None,
    harness: str = HARNESS,
) -> str:
    """Write one part's log as eval-set would, with every sample unless skip leaves out the last ones.

    errored, retries, limit and reused apply to the first that many samples (errored, one error retry each, a turn
    limit hit, started before the log). stats_cost is the header's spend over every call (the samples' total by
    default), and extra_usage adds other models' spend to it; epochs, model, variant_arg, sample_id and cost_limit
    override what the log says it ran.
    """
    cost = COST[agent] if cost is None else cost
    keys = [(task, epoch) for epoch in range(1, EPOCHS[agent][variant] + 1) for task in TASKS]
    keys = keys[: len(keys) - skip] + (extra or [])
    samples = [
        sample(agent, task, epoch, cost, error=i < errored, retries=1 if i < retries else 0,
               limit="turn" if i < limit else None, started_at=EARLIER if i < reused else SAMPLE_START)
        for i, (task, epoch) in enumerate(keys)
    ]
    log = EvalLog(
        eval=EvalSpec(
            eval_id=eval_id or f"{agent}-{variant}-{name}", created=LOG_START, task="false_assurance", dataset=EvalDataset(),
            model=model or MODELS[agent], task_args={"variant": variant_arg or variant},
            config=EvalConfig(
                epochs=epochs or EPOCHS[agent][variant], sample_id=sample_id or TASKS, retry_on_error=2,
                cost_limit=main_status.COST_LIMITS[agent] if cost_limit == "the agent's" else cost_limit,
            ),
            metadata={"harness_sha256": harness},
        ),
        status=status,
        invalidated=invalidated,
        samples=samples,
        stats=EvalStats(
            started_at=LOG_START,
            model_usage={
                MODELS[agent]: ModelUsage(total_cost=sum(s.model_usage[MODELS[agent]].total_cost for s in samples) if stats_cost is None else stats_cost),
                **{model: ModelUsage(total_cost=spent) for model, spent in (extra_usage or {}).items()},
            },
        ),
    )
    path = logs / f"{agent}-{variant}" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(log, str(path))
    return str(path)


def write_all(logs, skip_parts: tuple = (), **changes) -> None:
    """Every part's log; changes maps (agent, variant) to write_part's keyword arguments for that part."""
    for agent, variant in PARTS:
        if (agent, variant) not in skip_parts:
            write_part(logs, agent, variant, **changes.get(f"{agent}_{variant}".replace("-", "_"), {}))


def args(tmp_path, *extra: str) -> list[str]:
    return [
        "--logs", str(tmp_path / "main"), "--rejected-logs", str(tmp_path / "main-rejected"),
        "--pilot-logs", str(tmp_path / "pilot"), "--earlier-logs", str(tmp_path / "pilot-rejected"),
        "--smoke-logs", str(tmp_path / "smoke-live"), *extra,
    ]


def run(tmp_path, capsys, *extra: str) -> tuple[int, dict, list[str]]:
    """main_status.main with --out tmp_path/out: its exit status, report.json and the printed lines."""
    status = main_status.main(args(tmp_path, "--out", str(tmp_path / "out"), *extra))
    out = capsys.readouterr()
    assert out.err == ""
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert (tmp_path / "out/report.md").read_text() == out.out
    return status, report, out.out.splitlines()


def part(report: dict, agent: str, variant: str) -> dict:
    [found] = [p for p in report["parts"] if (p["agent"], p["variant"]) == (agent, variant)]
    return found


def test_finished_runs_exit_0(tmp_path, capsys):
    write_all(tmp_path / "main")
    status, report, lines = run(tmp_path, capsys)
    assert status == 0 and report["exit_status"] == 0
    assert lines[2] == "**Status: every part finished with every sample, and no cap is crossed.**"
    assert [(p["agent"], p["variant"], p["expected"], p["completed"]) for p in report["parts"]] == [
        ("haiku", "normal", 96, 96), ("haiku", "conflict", 192, 192), ("gpt-oss", "normal", 96, 96),
        ("gpt-oss", "conflict", 192, 192), ("sonnet", "normal", 40, 40), ("sonnet", "conflict", 40, 40),
    ]
    assert sum(p["expected"] for p in report["parts"]) == 656
    assert all(p["finished"] and p["status"] == "success" and p["errored"] == p["missing"] == 0 for p in report["parts"])
    spend = report["spend"]["agents"]
    assert spend["haiku"]["spend"] == pytest.approx(288 * 0.001)
    assert spend["gpt-oss"]["spend"] == pytest.approx(288 * 0.0005)
    assert spend["sonnet"]["spend"] == pytest.approx(80 * 0.01)
    assert [spend[a]["cap_left"] for a in ("haiku", "gpt-oss", "sonnet")] == [28.80, 15.50, 80.00]
    assert report["caps_crossed"] == [] and report["unfinished"] == []
    assert "| haiku-conflict | success | 1 | 192 of 192 | 0 | 0 | 0 | none | $0.1920 |" in lines
    assert "| Sonnet | $0.4000 | $0.4000 | $0.0000 | $0.8000 | $80.00 | no |" in lines
    assert "Every log counted finished, and none shows a replaced attempt." in lines


def test_without_out_it_writes_nothing_and_never_changes_a_log(tmp_path, capsys):
    write_all(tmp_path / "main", gpt_oss_conflict={"status": "started", "skip": 50})
    kill(tmp_path / "main/gpt-oss-conflict/part.eval")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    assert main_status.main(args(tmp_path)) == 1
    assert "Part gpt-oss-conflict is running" in capsys.readouterr().out
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["main"]


def test_a_part_not_started_is_incomplete(tmp_path, capsys):
    write_all(tmp_path / "main", skip_parts=(("sonnet", "conflict"),))
    status, report, lines = run(tmp_path, capsys)
    assert status == 1
    assert report["unfinished"] == ["sonnet-conflict"]
    assert lines[2] == "**Status: 1 part of 6 not finished.**"
    assert f"Part sonnet-conflict has not started: {tmp_path}/main/sonnet-conflict holds no log." in lines
    assert part(report, "sonnet", "conflict")["missing"] == 40


def test_nothing_started_is_incomplete_not_bad_input(tmp_path, capsys):
    status, report, lines = run(tmp_path, capsys)
    assert status == 1
    assert report["unfinished"] == [main_status.part_name(*p) for p in PARTS]
    assert lines[2] == "**Status: 6 parts of 6 not finished.**"


def test_a_killed_part_is_incomplete_and_its_spend_a_lower_bound(tmp_path, capsys):
    """A running or killed log has no header.json, so it has status started and counts only the samples it wrote."""
    write_all(tmp_path / "main", haiku_conflict={"status": "started", "skip": 100})
    path = tmp_path / "main/haiku-conflict/part.eval"
    kill(path)
    status, report, lines = run(tmp_path, capsys)
    assert status == 1
    found = part(report, "haiku", "conflict")
    assert (found["status"], found["completed"], found["missing"], found["finished"]) == ("started", 92, 100, False)
    assert (
        "Part haiku-conflict is running or was stopped while running (its newest log has status started): 92 of 192 "
        "samples completed, 0 errored and 100 missing."
    ) in lines
    haiku = report["spend"]["agents"]["haiku"]
    assert haiku["spend"] == pytest.approx((96 + 92) * 0.001)
    assert [r["path"] for r in haiku["lower_bound"]] == [str(path)]
    assert any(line.startswith("| Haiku | $0.0960 | $0.0920 | $0.0000 | $0.1880 (a lower bound) |") for line in lines)
    assert any(line.startswith(f"Log {path} has status started, so it never finished") for line in lines)
    assert (
        'To start or resume the unfinished parts of Haiku, run scripts/run_main.sh with AGENTS="haiku"; eval-set runs only '
        "their missing or errored samples."
    ) in lines


@pytest.mark.parametrize("status", ["error", "cancelled"])
def test_a_part_whose_log_did_not_succeed_is_incomplete_even_with_every_sample(tmp_path, capsys, status):
    write_all(tmp_path / "main", gpt_oss_normal={"status": status})
    code, report, lines = run(tmp_path, capsys)
    assert code == 1
    assert report["unfinished"] == ["gpt-oss-normal"]
    assert f"Part gpt-oss-normal has not finished (its newest log has status {status}): 96 of 96 samples completed, 0 errored and 0 missing." in lines


def test_errored_samples_of_a_finished_part_are_counted_and_excluded(tmp_path, capsys):
    """eval-set does not re-run the errored samples of a log it counts complete, so the part has finished."""
    write_all(tmp_path / "main", haiku_normal={"errored": 3, "retries": 3}, sonnet_normal={"errored": 1, "status": "started"})
    kill(tmp_path / "main/sonnet-normal/part.eval")
    status, report, lines = run(tmp_path, capsys)
    assert status == 1 and report["unfinished"] == ["sonnet-normal"]
    found = part(report, "haiku", "normal")
    assert (found["finished"], found["completed"], found["errored"], found["missing"], found["sample_retries"]) == (True, 93, 3, 0, 3)
    assert found["errored_samples"][0] == {"id": "022_0", "epoch": 1, "error": "RuntimeError: boom"}
    assert "| haiku-normal | success | 1 | 93 of 96 | 3 | 0 | 3 | none | $0.0960 |" in lines
    assert "Part haiku-normal: 022_0 epoch 1 errored: RuntimeError: boom" in lines
    assert "An errored sample of a finished part is not re-run by eval-set; PLAN counts it per agent and variant and excludes it." in lines
    assert part(report, "sonnet", "normal")["errored"] == 1


def test_errored_samples_alone_do_not_fail_the_status(tmp_path, capsys):
    write_all(tmp_path / "main", gpt_oss_conflict={"errored": 2})
    status, report, _ = run(tmp_path, capsys)
    assert status == 0
    assert part(report, "gpt-oss", "conflict")["errored"] == 2


def test_limit_hits_count_as_completed(tmp_path, capsys):
    write_all(tmp_path / "main", sonnet_conflict={"limit": 4})
    status, report, lines = run(tmp_path, capsys)
    assert status == 0
    assert part(report, "sonnet", "conflict")["limits"] == {"turn": 4}
    assert "| sonnet-conflict | success | 1 | 40 of 40 | 0 | 0 | 0 | turn 4 | $0.4000 |" in lines


def test_a_retried_part_uses_its_newest_log_and_flags_the_replaced_attempt(tmp_path, capsys):
    """A killed attempt stays beside the log that resumed it; the newest log by modification time is the part's."""
    main = tmp_path / "main"
    write_all(main, skip_parts=(("haiku", "normal"),))
    old = write_part(main, "haiku", "normal", name="old.eval", status="started", skip=60)
    kill(old)
    new = write_part(main, "haiku", "normal", name="new.eval", reused=36, stats_cost=0.06)
    os.utime(old, (1_000_000, 1_000_000))
    status, report, lines = run(tmp_path, capsys)
    assert status == 0
    found = part(report, "haiku", "normal")
    assert found["newest"] == new and len(found["logs"]) == 2
    assert (found["finished"], found["completed"], found["reused"], found["rerun"]) == (True, 96, 36, 0)
    assert "Part haiku-normal holds 2 distinct logs; each is counted in full, so samples a later attempt reused from an earlier one count twice, an over-count." in lines
    assert "Part haiku-normal's newest log holds 36 samples copied from a failed earlier attempt, so that attempt's spend may be in no log." in lines
    # The killed attempt's 36 samples and the new log's 96 (the larger of its samples and its header).
    haiku = report["spend"]["agents"]["haiku"]
    assert haiku["parts"]["normal"] == pytest.approx((36 + 96) * 0.001)
    assert sorted(r["path"] for r in haiku["lower_bound"]) == sorted([old, new])

    # Made older than the killed attempt, the resumed log is no longer the part's: the part has not finished.
    os.utime(new, (999_000, 999_000))
    status, report, _ = run(tmp_path, capsys)
    assert status == 1 and part(report, "haiku", "normal")["newest"] == old


def test_sample_error_retries_beyond_retry_on_error_show_a_rerun(tmp_path, capsys):
    main = tmp_path / "main"
    write_all(main, skip_parts=(("gpt-oss", "normal"),))
    path = main / "gpt-oss-normal/part.eval"
    write_part(main, "gpt-oss", "normal")
    log = read_eval_log(str(path))
    log.samples[0].error_retries = [EvalRetryError(message="RuntimeError: seeded", traceback="", traceback_ansi="")] * 3
    write_eval_log(log, str(path))
    status, report, lines = run(tmp_path, capsys)
    found = part(report, "gpt-oss", "normal")
    assert (found["sample_retries"], found["rerun"]) == (3, 1)
    assert "Part gpt-oss-normal's newest log holds 1 sample re-run after a failed earlier attempt, so that attempt's spend may be in no log." in lines
    assert status == 0


@pytest.mark.parametrize(("haiku_cost", "crossed"), [(0.1, False), (0.1 + 1e-6, True)], ids=["at_cap", "over_cap"])
def test_a_claude_cap_is_crossed_only_above_it(tmp_path, capsys, haiku_cost, crossed):
    """288 Haiku samples at $0.10 make exactly PLAN's $28.80 cap, which is within it."""
    write_all(tmp_path / "main", haiku_normal={"cost": haiku_cost}, haiku_conflict={"cost": haiku_cost})
    status, report, lines = run(tmp_path, capsys)
    assert report["spend"]["agents"]["haiku"]["crossed"] is crossed
    assert status == (1 if crossed else 0)
    assert report["caps_crossed"] == (["haiku"] if crossed else [])
    if crossed:
        assert lines[2] == "**Status: the cap of Haiku crossed.**"
        assert main_status.CLAUDE_MD_APPROVAL in lines
        assert any(line.startswith("Haiku spent $28.8003, above its $28.80 cap.") for line in lines)


def test_a_sonnet_cap_counts_its_rejected_logs(tmp_path, capsys):
    """Rejected logs count against the cap of the agent whose model they ran, split by model within Anthropic."""
    write_all(tmp_path / "main", sonnet_normal={"cost": 0.9}, sonnet_conflict={"cost": 0.9})
    rejected = tmp_path / "main-rejected"
    write_part(rejected / "attempt-1", "sonnet", "normal", cost=0.2, stats_cost=8.5, eval_id="rejected-sonnet")
    write_part(rejected / "attempt-1", "haiku", "normal", cost=0.01, eval_id="rejected-haiku")
    write_part(rejected, "gpt-oss", "conflict", cost=0.001, eval_id="rejected-gpt-oss")
    status, report, lines = run(tmp_path, capsys)
    spend = report["spend"]["agents"]
    assert spend["sonnet"]["rejected"] == pytest.approx(8.5)
    assert spend["haiku"]["rejected"] == pytest.approx(0.96)
    assert spend["gpt-oss"]["rejected"] == pytest.approx(0.192)
    # $72.00 in the parts and $8.50 rejected is above the $80.00 cap; Haiku's $0.288 and $0.96 are within its own.
    assert spend["sonnet"]["spend"] == pytest.approx(80.5)
    assert status == 1 and report["caps_crossed"] == ["sonnet"]
    assert "| Sonnet | $36.0000 | $36.0000 | $8.5000 | $80.5000 | $80.00 | yes |" in lines


@pytest.mark.parametrize(("smoke", "crossed"), [(0.6, False), (0.61, True)], ids=["at_cap", "over_cap"])
def test_the_gpt_oss_cap_is_less_what_the_pilot_and_smoke_test_spent_on_openrouter(tmp_path, capsys, smoke, crossed):
    """$14.40 in the full runs against $15.50 less $0.288 (pilot), $0.212 (rejected pilot attempt) and the smoke test."""
    write_all(tmp_path / "main", gpt_oss_normal={"cost": 0.05}, gpt_oss_conflict={"cost": 0.05})
    write_part(tmp_path / "pilot", "gpt-oss", "normal", cost=0.003, eval_id="pilot-gpt-oss")
    assert pilot_report.read_spend(tmp_path / "pilot", set())["spend"]["openrouter"] == pytest.approx(0.288)
    write_part(tmp_path / "pilot", "haiku", "normal", cost=5.0, eval_id="pilot-haiku")
    write_part(tmp_path / "pilot-rejected", "gpt-oss", "conflict", cost=0.0, stats_cost=0.212, eval_id="pilot-rejected")
    write_part(tmp_path / "smoke-live", "gpt-oss", "normal", cost=0.0, stats_cost=smoke, eval_id="smoke")
    status, report, lines = run(tmp_path, capsys)
    gpt_oss = report["spend"]["agents"]["gpt-oss"]
    assert gpt_oss["spent_before"] == pytest.approx({"pilot": 0.288, "pilot_rejected": 0.212, "smoke": smoke})
    assert gpt_oss["cap_left"] == pytest.approx(15.50 - 0.5 - smoke)
    assert gpt_oss["crossed"] is crossed and status == (1 if crossed else 0)
    # The pilot's Anthropic spend has its own Budget line and touches no full-run cap, and the earlier OpenRouter spend
    # comes off the gpt-oss cap alone.
    assert report["spend"]["agents"]["haiku"]["spend"] == pytest.approx(0.288)
    assert [report["spend"]["agents"][a]["cap_left"] for a in ("haiku", "sonnet")] == [28.80, 80.00]
    assert any(line.startswith("The gpt-oss cap is PLAN's $15.50 less ") for line in lines)


def test_a_recovered_copy_of_a_log_is_counted_once_and_said_so(tmp_path, capsys):
    """Inspect's recovered copy of a crashed log keeps its eval_id, so read_spend counts the two once."""
    main = tmp_path / "main"
    write_all(main, skip_parts=(("haiku", "normal"),))
    write_part(main, "haiku", "normal", name="crashed.eval", eval_id="same")
    write_part(main, "haiku", "normal", name="crashed-recovered.eval", eval_id="same")
    status, report, lines = run(tmp_path, capsys)
    assert status == 0 and len(part(report, "haiku", "normal")["logs"]) == 2
    assert report["spend"]["agents"]["haiku"]["parts"]["normal"] == pytest.approx(96 * 0.001)
    assert "Part haiku-normal holds 1 copy of another of its logs with the same eval_id (such as a log Inspect recovered from a crashed one), counted once." in lines
    assert not any("distinct log" in line for line in lines)


def test_a_log_copied_into_two_directories_is_counted_once(tmp_path, capsys):
    """One seen set spans the parts, the rejected logs and the earlier ones, so a copied log's spend counts once."""
    write_all(tmp_path / "main")
    write_part(tmp_path / "pilot", "gpt-oss", "normal", cost=0.003, eval_id="pilot-gpt-oss")
    write_part(tmp_path / "main-rejected", "gpt-oss", "normal", cost=0.003, eval_id="pilot-gpt-oss")
    status, report, _ = run(tmp_path, capsys)
    gpt_oss = report["spend"]["agents"]["gpt-oss"]
    # The rejected logs are read before the pilot's, so the copy counts there and not again as the pilot's.
    assert gpt_oss["rejected"] == pytest.approx(0.288)
    assert gpt_oss["spent_before"]["pilot"] == 0.0
    assert gpt_oss["cap_left"] == pytest.approx(15.50)
    assert status == 0


def test_a_cap_that_is_not_a_whole_number_of_cents_prints_to_4_places(tmp_path, capsys):
    """gpt-oss's cap less the smoke test's spend is $15.474080595; rounded to $15.47 it would read as below a $15.4720 spend."""
    write_all(tmp_path / "main", gpt_oss_normal={"cost": 7.736 / 96}, gpt_oss_conflict={"cost": 7.736 / 192})
    write_part(tmp_path / "smoke-live", "gpt-oss", "normal", cost=0.0, stats_cost=0.025919405, eval_id="smoke")
    status, report, lines = run(tmp_path, capsys)
    assert status == 0 and not report["spend"]["agents"]["gpt-oss"]["crossed"]
    assert "| gpt-oss | $7.7360 | $7.7360 | $0.0000 | $15.4720 | $15.4741 | no |" in lines
    assert "| Haiku | $0.0960 | $0.1920 | $0.0000 | $0.2880 | $28.80 | no |" in lines
    assert main_status.cap_usd(15.474080595) == "$15.4741" and main_status.cap_usd(14.3) == "$14.30"


def test_the_refusals_and_the_crossed_line_print_such_a_cap_to_4_places(tmp_path, capsys):
    """The same $15.474080595 gpt-oss cap in launch_problems' two refusals and the report's crossed-cap line."""
    write_part(tmp_path / "smoke-live", "gpt-oss", "normal", cost=0.0, stats_cost=0.025919405, eval_id="smoke")
    write_part(tmp_path / "main", "gpt-oss", "normal", cost=7.736 / 96)
    assert launch(tmp_path, "gpt-oss") == [
        "gpt-oss has spent $7.7360, and its 192 samples still to run at up to $0.05 each could bring it to $17.3360, "
        "above its $15.4741 cap"
    ]
    write_part(tmp_path / "main", "gpt-oss", "conflict", cost=7.74 / 192)
    assert launch(tmp_path, "gpt-oss") == ["gpt-oss has spent $15.4760, above its $15.4741 cap"]
    _, _, lines = run(tmp_path, capsys)
    assert "gpt-oss spent $15.4760, above its $15.4741 cap." in lines


def test_unattributed_paid_spend_in_rejected_logs_is_bad_input(tmp_path, capsys):
    write_all(tmp_path / "main")
    write_part(tmp_path / "main-rejected", "haiku", "normal", eval_id="other", model="anthropic/claude-opus-5-5")
    status = main_status.main(args(tmp_path, "--out", str(tmp_path / "out2")))
    err = capsys.readouterr().err
    assert status == 2
    assert "holds logs with paid spend whose model is no agent's" in err
    assert not (tmp_path / "out2").exists()


def test_the_read_spend_model_filter_counts_each_log_once(tmp_path):
    write_part(tmp_path / "r", "haiku", "normal", eval_id="h")
    write_part(tmp_path / "r", "sonnet", "normal", eval_id="s")
    seen: set[str] = set()
    haiku = pilot_report.read_spend(tmp_path / "r", seen, model=MODELS["haiku"])
    assert (haiku["logs"], seen) == (1, {"h"})
    assert haiku["spend"]["anthropic"] == pytest.approx(0.096)
    sonnet = pilot_report.read_spend(tmp_path / "r", seen, model=MODELS["sonnet"])
    assert (sonnet["logs"], sonnet["spend"]["anthropic"]) == (1, pytest.approx(0.4))
    assert pilot_report.read_spend(tmp_path / "r", seen)["logs"] == 0
    assert pilot_report.read_spend(tmp_path / "r", set())["spend"]["anthropic"] == pytest.approx(0.496)


BAD = [
    ({"model": MODELS["sonnet"]}, "model 'anthropic/claude-sonnet-5-5' does not match the haiku-normal part's 'anthropic/claude-haiku-5-5'"),
    ({"variant_arg": "conflict"}, "variant 'conflict' does not match the haiku-normal part's 'normal'"),
    ({"epochs": 5}, "epochs 5 does not match the haiku-normal part's 12"),
    ({"sample_id": ["078_0", "327_0"]}, "sample_id ['078_0', '327_0'] does not match the haiku-normal part's ['022_0', '022_2',"),
    ({"extra": [("502_0", 1)]}, "holds samples outside the haiku-normal part's 8 tasks and 12 epochs: 502_0 epoch 1"),
    ({"extra": [("078_0", 13)]}, "holds samples outside the haiku-normal part's 8 tasks and 12 epochs: 078_0 epoch 13"),
    # A part run without the launcher's per-sample cost limit, the money protection, is another command's.
    ({"cost_limit": 1.0}, "cost_limit 1.0 does not match the haiku-normal part's 0.1"),
    ({"cost_limit": None}, "cost_limit None does not match the haiku-normal part's 0.1"),
]


@pytest.mark.parametrize(("change", "message"), BAD)
def test_a_log_another_command_wrote_is_bad_input(tmp_path, capsys, change, message):
    write_all(tmp_path / "main", haiku_normal=change)
    status = main_status.main(args(tmp_path, "--out", str(tmp_path / "out")))
    out = capsys.readouterr()
    assert status == 2 and out.out == ""
    assert out.err.startswith("Bad input, so no report was written:")
    assert message in out.err
    assert not (tmp_path / "out").exists()


def test_an_unreadable_log_or_a_file_for_a_directory_is_bad_input(tmp_path, capsys):
    write_all(tmp_path / "main")
    (tmp_path / "main/gpt-oss-conflict/broken.eval").write_bytes(b"not a zip")
    assert main_status.main(args(tmp_path)) == 2
    assert "broken.eval: cannot be read" in capsys.readouterr().err
    (tmp_path / "main/gpt-oss-conflict/broken.eval").unlink()
    (tmp_path / "smoke-live").write_text("not a directory")
    assert main_status.main(args(tmp_path)) == 2
    assert f"{tmp_path}/smoke-live is not a directory of logs" in capsys.readouterr().err
    (tmp_path / "smoke-live").unlink()
    assert main_status.main(args(tmp_path, "--logs", str(tmp_path / "main/haiku-normal/part.eval"))) == 2
    assert "part.eval is not a directory of logs" in capsys.readouterr().err


def test_a_running_launcher_is_named(tmp_path, capsys):
    write_all(tmp_path / "main", sonnet_conflict={"status": "started", "skip": 10})
    kill(tmp_path / "main/sonnet-conflict/part.eval")
    lock = tmp_path / "main" / main_status.LOCK
    lock.mkdir()
    (lock / "launcher").write_text(f"{os.getpid()}\n")
    status, report, lines = run(tmp_path, capsys)
    assert status == 1 and report["lock"]["state"] == "running" and report["lock"]["launcher"] == os.getpid()
    assert f"A run_main.sh (pid {os.getpid()}) holds the lock in {tmp_path}/main." in lines
    assert not any(line.startswith("To start or resume") for line in lines)
    assert sorted(p.name for p in lock.iterdir()) == ["launcher"]


def test_the_report_is_one_sentence_per_line_without_em_dashes(tmp_path, capsys):
    write_all(tmp_path / "main", haiku_normal={"errored": 1}, gpt_oss_conflict={"status": "started", "skip": 3})
    kill(tmp_path / "main/gpt-oss-conflict/part.eval")
    _, _, lines = run(tmp_path, capsys)
    assert any(line.startswith("Log ") for line in lines)
    for line in lines:
        assert chr(0x2014) not in line
        if not line.startswith("|"):
            assert ". " not in line.rstrip("."), line


def test_a_success_log_short_of_samples_is_incomplete(tmp_path, capsys):
    """A graceful cancel finishes a success log without the samples it abandoned, which eval-set then re-runs."""
    write_all(tmp_path / "main", haiku_conflict={"skip": 7})
    status, report, lines = run(tmp_path, capsys)
    assert status == 1 and report["unfinished"] == ["haiku-conflict"]
    assert "Part haiku-conflict has not finished (its newest log has status success): 185 of 192 samples completed, 0 errored and 7 missing." in lines


def test_an_invalidated_log_is_incomplete(tmp_path, capsys):
    write_all(tmp_path / "main", sonnet_normal={"invalidated": True})
    status, report, lines = run(tmp_path, capsys)
    assert status == 1 and report["unfinished"] == ["sonnet-normal"]
    assert part(report, "sonnet", "normal")["status"] == "success (invalidated)"
    assert "Part sonnet-normal has not finished (its newest log has status success (invalidated)): 40 of 40 samples completed, 0 errored and 0 missing." in lines


@pytest.mark.parametrize("where", ["part", "rejected"])
def test_spend_on_another_provider_is_bad_input(tmp_path, capsys, where):
    """A Haiku log that spent on OpenRouter would put that spend against no cap."""
    if where == "part":
        write_all(tmp_path / "main", haiku_normal={"extra_usage": {MODELS["gpt-oss"]: 0.25}})
    else:
        write_all(tmp_path / "main")
        write_part(tmp_path / "main-rejected", "haiku", "normal", eval_id="rejected", extra_usage={MODELS["gpt-oss"]: 0.25})
    status = main_status.main(args(tmp_path, "--out", str(tmp_path / "out")))
    err = capsys.readouterr().err
    assert status == 2
    where_named = "normal" if where == "part" else "rejected"
    assert f"haiku {where_named} logs spent $0.2500 on openrouter, which is not haiku's provider" in err
    assert not (tmp_path / "out").exists()


LOCK_STATES = {"stale": "stale", "orphaned_eval_set": "orphaned eval-set", "no_launcher_pid": "no launcher pid", "zero_pid": "no launcher pid"}


@pytest.mark.parametrize("case", list(LOCK_STATES))
def test_a_lock_without_a_running_launcher_is_named_and_resuming_is_not_suggested(tmp_path, capsys, case):
    """run_main.sh refuses beside any lock, so the report names its state instead of suggesting a resume."""
    write_all(tmp_path / "main", gpt_oss_normal={"status": "started", "skip": 10})
    kill(tmp_path / "main/gpt-oss-normal/part.eval")
    lock = tmp_path / "main" / main_status.LOCK
    lock.mkdir()
    holder = dead_pid()
    by_hand = f"run_main.sh refuses to start until someone who has checked that no run_main.sh or eval-set is running on {tmp_path}/main removes it."
    if case == "stale":
        (lock / "launcher").write_text(f"{holder}\n")
        (lock / "eval-set").write_text(f"{dead_pid()}\n")
        message = f"{lock} is the stale lock of run_main.sh pid {holder}, which is no longer running; {by_hand}"
    elif case == "orphaned_eval_set":
        (lock / "launcher").write_text(f"{holder}\n")
        (lock / "eval-set").write_text(f"{os.getpid()}\n")
        message = (
            f"The eval-set (pid {os.getpid()}) that an earlier run_main.sh (pid {holder}) started is still running on "
            f"{tmp_path}/main and may still be spending; run_main.sh refuses to start while it runs, and after it ends "
            f"until someone removes {lock}."
        )
    else:
        if case == "zero_pid":
            (lock / "launcher").write_text("0\n")
        message = f"{lock} exists without a launcher pid; {by_hand}"
    contents = sorted(p.name for p in lock.iterdir())
    status, report, lines = run(tmp_path, capsys)
    assert status == 1
    assert report["lock"]["state"] == LOCK_STATES[case]
    assert message in lines
    assert not any(line.startswith("No run_main.sh holds") or line.startswith("To start or resume") for line in lines)
    assert sorted(p.name for p in lock.iterdir()) == contents


def test_without_a_lock_resuming_is_suggested(tmp_path, capsys):
    write_all(tmp_path / "main", skip_parts=(("sonnet", "normal"),))
    status, report, lines = run(tmp_path, capsys)
    assert status == 1 and report["lock"] == {"path": str(tmp_path / "main" / main_status.LOCK), "state": "none", "launcher": None, "eval_set": None}
    assert f"No run_main.sh holds the lock in {tmp_path}/main." in lines
    assert any(line.startswith("To start or resume the unfinished parts") for line in lines)


def test_resuming_is_suggested_only_for_the_agents_the_launcher_would_start(tmp_path, capsys):
    """A crossed cap, or no room for the samples still to run, makes run_main.sh refuse that agent, so the report
    names it with launch_problems' reason instead of suggesting a resume."""
    write_part(tmp_path / "main", "haiku", "normal", cost=0.31)
    write_part(tmp_path / "main-rejected", "sonnet", "normal", cost=0.0, stats_cost=0.25, eval_id="rejected")
    status, report, lines = run(tmp_path, capsys)
    assert status == 1 and report["caps_crossed"] == ["haiku"]
    assert (
        'To start or resume the unfinished parts of gpt-oss, run scripts/run_main.sh with AGENTS="gpt-oss"; eval-set runs '
        "only their missing or errored samples."
    ) in lines
    refused = [line for line in lines if line.startswith("run_main.sh refuses")]
    assert refused == [
        "run_main.sh refuses to start the unfinished parts of Haiku: Haiku has spent $29.7600, above its $28.80 cap.",
        "run_main.sh refuses to start the unfinished parts of Sonnet: Sonnet has spent $0.2500, and its 80 samples still "
        "to run at up to $1.00 each could bring it to $80.2500, above its $80.00 cap.",
    ]
    # The reasons are the launcher's own.
    reasons = main_status.launch_problems(args(tmp_path), ["haiku", "sonnet"])
    assert refused == [f"run_main.sh refuses to start the unfinished parts of {n}: {why}." for n, why in zip(("Haiku", "Sonnet"), reasons, strict=True)]
    assert any(line.startswith("Starting them needs Jessup's approval of a higher cap first") for line in lines)
    # With every agent refused, no resume is suggested at all.
    write_part(tmp_path / "smoke-live", "gpt-oss", "normal", cost=0.0, stats_cost=5.0, eval_id="smoke")
    _, _, lines = run(tmp_path, capsys)
    assert not any(line.startswith("To start or resume") for line in lines)
    assert sum(line.startswith("run_main.sh refuses") for line in lines) == 3


@pytest.mark.parametrize("where", ["pilot", "pilot-rejected", "smoke-live"])
def test_an_incomplete_earlier_gpt_oss_log_makes_gpt_oss_a_lower_bound(tmp_path, capsys, where):
    """Earlier OpenRouter spend comes off the gpt-oss cap, so a log of it that never finished leaves the cap uncertain."""
    write_all(tmp_path / "main")
    killed = write_part(tmp_path / where, "gpt-oss", "normal", status="started", skip=90, eval_id="earlier-gpt-oss")
    kill(killed)
    # A killed Anthropic log there touches no full-run cap.
    claude = write_part(tmp_path / where, "haiku", "normal", status="started", skip=90, eval_id="earlier-haiku")
    kill(claude)
    status, report, lines = run(tmp_path, capsys)
    assert status == 0
    spend = report["spend"]["agents"]
    assert [r["path"] for r in spend["gpt-oss"]["lower_bound"]] == [killed]
    assert spend["haiku"]["lower_bound"] == [] and spend["sonnet"]["lower_bound"] == []
    assert any(line.startswith("| gpt-oss | $0.0480 | $0.0960 | $0.0000 | $0.1440 (a lower bound) |") for line in lines)
    assert any(line.startswith(f"Log {killed} has status started") for line in lines)


def test_a_started_rejected_log_makes_its_agent_a_lower_bound(tmp_path, capsys):
    write_all(tmp_path / "main")
    killed = write_part(tmp_path / "main-rejected/attempt-1", "sonnet", "conflict", status="started", skip=30, eval_id="rejected")
    kill(killed)
    status, report, lines = run(tmp_path, capsys)
    assert status == 0
    sonnet = report["spend"]["agents"]["sonnet"]
    assert sonnet["rejected"] == pytest.approx(10 * 0.01)
    assert [r["path"] for r in sonnet["lower_bound"]] == [killed]
    assert "| Sonnet | $0.4000 | $0.4000 | $0.1000 | $0.9000 (a lower bound) | $80.00 | no |" in lines


@pytest.mark.parametrize("stray", ["old/part.eval", "haiku-conflct/part.eval", "part.eval", "haiku-normal-old/part.eval"])
def test_a_log_outside_the_part_directories_is_bad_input(tmp_path, capsys, stray):
    """Its spend would count against no cap, so the report refuses instead of leaving it out."""
    main = tmp_path / "main"
    write_all(main)
    (main / stray).parent.mkdir(parents=True, exist_ok=True)
    (main / stray).write_bytes((main / "haiku-normal/part.eval").read_bytes())
    status = main_status.main(args(tmp_path, "--out", str(tmp_path / "out")))
    out = capsys.readouterr()
    assert status == 2 and out.out == ""
    assert (
        f"{main / stray} is under {main} but in none of its six part directories, so its spend would count against "
        f"no cap; move it into {tmp_path / 'main-rejected'}"
    ) in out.err
    assert not (tmp_path / "out").exists()
    # Nested inside a part's directory, a log is the part's, as eval-set lists a log directory recursively.
    (main / stray).unlink()
    write_part(main, "haiku", "normal", name="attempt/part.eval", eval_id="nested")
    status, report, _ = run(tmp_path, capsys)
    assert status == 0
    assert [log["path"] for log in part(report, "haiku", "normal")["logs"]] == [
        str(main / "haiku-normal/attempt/part.eval"), str(main / "haiku-normal/part.eval"),
    ]


@pytest.mark.parametrize(
    "where", ["main/haiku-normal", "main/other", "main-rejected", "pilot/gpt-oss-normal", "pilot-rejected", "smoke-live"]
)
def test_a_json_format_log_is_bad_input(tmp_path, capsys, where):
    """The parts and read_spend read only .eval logs, so a JSON log (INSPECT_LOG_FORMAT=json) would hide its spend."""
    main = tmp_path / "main"
    write_all(main)
    # eval-set's own logs.json and eval-set.json are not logs, so they are no problem.
    (main / "haiku-normal/logs.json").write_text("{}")
    (main / "haiku-normal/eval-set.json").write_text("{}")
    assert run(tmp_path, capsys)[0] == 0
    log = tmp_path / where / "2026-11-07T03-00-00-00-00_false-assurance_abc.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("{}")
    status = main_status.main(args(tmp_path, "--out", str(tmp_path / "out2")))
    out = capsys.readouterr()
    assert status == 2 and out.out == ""
    assert f"{log.resolve()} is an Inspect log in JSON format, whose samples and spend this script does not read" in out.err
    assert not (tmp_path / "out2").exists()


@pytest.mark.parametrize("where", ["main/haiku-normal-old/attempt.json", "main/sonnet-normal/copy.json", "main-rejected/x.json"])
def test_a_json_log_with_any_name_is_bad_input(tmp_path, capsys, where):
    """A readable JSON log under another name is found by its header, while other JSON files are left alone."""
    main = tmp_path / "main"
    write_all(main)
    (main / "haiku-normal/report.json").write_text('{"not": "a log"}')
    assert run(tmp_path, capsys)[0] == 0
    log = tmp_path / where
    log.parent.mkdir(parents=True, exist_ok=True)
    source = Path(write_part(tmp_path / "elsewhere", "haiku", "normal", name="part.json", eval_id="json-log"))
    source.rename(log)
    status = main_status.main(args(tmp_path, "--out", str(tmp_path / "out2")))
    out = capsys.readouterr()
    assert status == 2 and out.out == ""
    assert f"{log.resolve()} is an Inspect log in JSON format" in out.err
    assert out.err.count("is an Inspect log in JSON format") == 1


def test_errored_samples_are_named_in_the_status_line(tmp_path, capsys):
    write_all(tmp_path / "main", haiku_normal={"errored": 1}, sonnet_conflict={"errored": 2})
    status, _, lines = run(tmp_path, capsys)
    assert status == 0
    assert lines[2] == "**Status: every part finished, with 3 errored samples excluded, and no cap is crossed.**"


def test_an_out_path_that_cannot_be_written_exits_2(tmp_path, capsys):
    write_all(tmp_path / "main")
    (tmp_path / "out").write_text("a file, not a directory")
    assert main_status.main(args(tmp_path, "--out", str(tmp_path / "out"))) == 2
    out = capsys.readouterr()
    assert out.out == "" and out.err.startswith(f"Cannot write the report to {tmp_path / 'out'}: ")


# launch_problems, which scripts/run_main.sh calls through launch_refusal before a real run and before each later part.


def launch(tmp_path, *agents: str) -> list[str]:
    return main_status.launch_problems(args(tmp_path), list(agents or main_status.AGENTS))


def test_a_fresh_full_run_fits_every_cap(tmp_path):
    """288 x $0.10 and 80 x $1.00 are exactly PLAN's Haiku and Sonnet caps, and 288 x $0.05 is within gpt-oss's."""
    assert launch(tmp_path) == []
    write_all(tmp_path / "main")
    assert launch(tmp_path) == []


def test_rejected_spend_leaves_no_room_for_every_sample(tmp_path):
    write_part(tmp_path / "main-rejected/attempt-1", "sonnet", "normal", cost=0.0, stats_cost=0.25, eval_id="rejected")
    assert launch(tmp_path, "haiku", "gpt-oss") == []
    assert launch(tmp_path, "sonnet") == [
        "Sonnet has spent $0.2500, and its 80 samples still to run at up to $1.00 each could bring it to $80.2500, "
        "above its $80.00 cap"
    ]


def test_a_crossed_cap_is_refused_even_with_nothing_left_to_run(tmp_path):
    write_all(tmp_path / "main", haiku_normal={"cost": 0.2}, haiku_conflict={"cost": 0.1})
    assert launch(tmp_path, "haiku") == ["Haiku has spent $38.4000, above its $28.80 cap"]


@pytest.mark.parametrize(("cost", "refused"), [(0.1, False), (0.1001, True)], ids=["at_cap", "over_cap"])
def test_only_unfinished_parts_count_their_samples_still_to_run(tmp_path, cost, refused):
    """A finished part runs none, and a killed one its missing samples: 6 of 96 missing and 192 not started."""
    main = tmp_path / "main"
    write_part(main, "haiku", "normal", cost=cost)
    killed = write_part(main, "haiku", "conflict", status="started", skip=6, cost=cost)
    kill(killed)
    expected = [
        f"Haiku has spent ${282 * cost:.4f} (a lower bound), and its 6 samples still to run at up to $0.10 each could "
        f"bring it to ${282 * cost + 0.6:.4f}, above its $28.80 cap"
    ]
    assert launch(tmp_path, "haiku") == (expected if refused else [])


def test_errored_samples_of_a_finished_part_are_not_run_again(tmp_path):
    """eval-set re-runs no sample of a log it counts complete, so 10 errored Haiku samples leave 192 to run."""
    write_part(tmp_path / "main", "haiku", "normal", errored=10, cost=0.1)
    assert launch(tmp_path, "haiku") == []


def test_errored_samples_of_an_unfinished_part_count_as_still_to_run(tmp_path):
    main = tmp_path / "main"
    write_part(main, "gpt-oss", "normal", status="error", errored=40, cost=0.05)
    write_part(tmp_path / "smoke-live", "gpt-oss", "normal", cost=0.0, stats_cost=1.0, eval_id="smoke")
    # $4.80 spent plus 40 errored and 192 unstarted samples at $0.05 is $16.40, above $15.50 less the smoke test's $1.00.
    assert launch(tmp_path, "gpt-oss") == [
        "gpt-oss has spent $4.8000, and its 232 samples still to run at up to $0.05 each could bring it to $16.4000, "
        "above its $14.50 cap"
    ]


def test_bad_input_stops_the_launch_check(tmp_path):
    write_part(tmp_path / "main/old", "haiku", "normal")
    with pytest.raises(pilot_report.BadInput, match="in none of its six part directories"):
        launch(tmp_path, "sonnet")


def test_launch_refusal_is_what_the_launcher_prints(tmp_path):
    """None when every agent may start; otherwise launch_problems' lines and the approval rule, or the bad input."""
    assert main_status.launch_refusal(args(tmp_path), list(main_status.AGENTS)) is None
    write_part(tmp_path / "main", "haiku", "normal", cost=0.31)
    assert main_status.launch_refusal(args(tmp_path), ["gpt-oss", "sonnet"]) is None
    write_part(tmp_path / "main-rejected", "sonnet", "normal", cost=0.0, stats_cost=0.25, eval_id="rejected")
    assert main_status.launch_refusal(args(tmp_path), ["haiku", "gpt-oss", "sonnet"]) == (
        "Haiku has spent $29.7600, above its $28.80 cap; Sonnet has spent $0.2500, and its 80 samples still to run at "
        f"up to $1.00 each could bring it to $80.2500, above its $80.00 cap; {main_status.LAUNCH_APPROVAL}"
    )
    write_part(tmp_path / "main/old", "haiku", "normal")
    refusal = main_status.launch_refusal(args(tmp_path), ["gpt-oss"])
    assert refusal.startswith(f"scripts/main_status.py cannot count the spend so far: {tmp_path}/main/old/haiku-normal/part.eval is under ")
    assert refusal.endswith(f"in none of its six part directories, so its spend would count against no cap; move it into {tmp_path}/main-rejected")
