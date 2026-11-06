"""The operator script scripts/score_logs.py on logs under tmp_path; no model call.

run_command is replaced in every test that could reach it except one M1 run, which needs no model, so
the script never starts inspect score with PLAN's Claude roles; the commands are compared with the one
PLAN.md writes out.
"""

import json
import os
import re
import shlex
from pathlib import Path

import pytest
import score_logs
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ModelUsage
from inspect_ai.scorer import Score
from test_llm_monitors import GPT_OSS, HAIKU, MODEL, episode, label_row, make_log, rewrite_lock_held, sample, write_labels, write_log

from rewardhack.monitors import llm_monitors, spend, summarize, transcript
from rewardhack.monitors.summarize import SummaryTarget
from rewardhack.task import ROOT

PLAN = (ROOT / "docs/PLAN.md").read_text()


@pytest.fixture(autouse=True)
def command_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test runs inspect score (when it does) and scans for .env from a folder under tmp_path, never the repo root, whose .env holds the API keys."""
    assert score_logs.COMMAND_DIR == ROOT
    directory = tmp_path / "command_dir"
    directory.mkdir()
    monkeypatch.setattr(score_logs, "COMMAND_DIR", directory)
    return directory


@pytest.fixture
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "spend.json"
    monkeypatch.setenv(spend.LEDGER_ENV, str(path))
    return spend.init_ledger()


@pytest.fixture
def no_runs(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Fails any attempt to start inspect score; returns the list of commands that would have run."""

    def refuse(command: list[str], lock_fd: int | None = None) -> int:
        raise AssertionError(f"tried to run {command}")

    monkeypatch.setattr(score_logs, "run_command", refuse)
    return []


def plan_command(log: str, scorer: str) -> str:
    """PLAN.md's inspect score command for one scorer, the judge with the judge role."""
    template = re.search(r"as `(inspect score logs/scored/<log>\.eval [^`]+)`", PLAN).group(1)
    command = template.replace("logs/scored/<log>.eval", log).replace("<scorer>", scorer)
    if scorer == "judge":
        setting = re.search(r"judge `(\{[^`]+\})`", PLAN).group(1)
        command = re.sub(r"--model-role '[^']+'", f"--model-role 'judge={setting}'", command)
    return command


def test_the_commands_are_plans():
    log = "logs/scored/main/gpt-oss-conflict/x.eval"
    for scorer in llm_monitors.SCORERS:
        assert shlex.join(score_logs.score_command(log, scorer)) == plan_command(log, scorer)
    assert plan_command(log, "m2").endswith(
        "--model-role 'monitor={model: anthropic/claude-haiku-5-5, reasoning_effort: low, max_tokens: 8000}'"
    )
    assert shlex.join(score_logs.score_command(log, "judge", "logs/pilot/labels.jsonl")) == plan_command(log, "judge") + " -S labels=logs/pilot/labels.jsonl"
    assert "--stream" not in plan_command(log, "m2")
    assert shlex.join(score_logs.m1_command(log)) == (
        f"inspect score {log} --overwrite --action append --model mockllm/model --scorer src/rewardhack/monitors/m1_regex.py@m1"
    )
    with pytest.raises(ValueError):
        score_logs.score_command(log, "m2", "labels.jsonl")


# --- scoring ---


def scored_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """A scored root with a gpt-oss conflict log and a Haiku normal log, and label rows for both."""
    root = tmp_path / "scored"
    gpt = make_log(GPT_OSS, [sample("g1"), sample("g2", epoch=2)], eval_id="gpt")
    haiku = make_log(HAIKU, [sample("h1", variant="normal", messages=episode("claude"))], eval_id="haiku")
    gpt_path = write_log(root / "main/gpt-oss-conflict/a.eval", gpt)
    haiku_path = write_log(root / "main/haiku-normal/b.eval", haiku)
    rows = [
        label_row(gpt, gpt.samples[0], outcome="visible_pass_but_vulnerable", gaming=True),
        label_row(gpt, gpt.samples[1], outcome="visible_fail", gaming=False),
        label_row(haiku, haiku.samples[0], outcome="visible_pass_but_vulnerable", gaming=None),
    ]
    return root, gpt_path, haiku_path, write_labels(tmp_path / "labels.jsonl", rows)


@pytest.fixture
def frozen_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(summarize, "TARGET_FILE", tmp_path / "summary_target.json")
    target = SummaryTarget(tokens=50, words=40, blocks=47, counter=HAIKU, agent=HAIKU, logs=("logs/pilot/haiku-normal/x.eval",))
    return summarize.write_target(target)


def command_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith("inspect score ")]


def test_the_dry_run_prints_every_command_in_plans_order_and_runs_nothing(tmp_path, ledger, frozen_target, no_runs, capsys):
    root, gpt_path, haiku_path, labels = scored_tree(tmp_path)
    before = {path: path.read_bytes() for path in (gpt_path, haiku_path)}
    assert score_logs.main(["score", "--dry-run", "--scored-root", str(root), "--labels", str(labels)]) == 0
    output = capsys.readouterr().out
    gpt, haiku = str(gpt_path.resolve()), str(haiku_path.resolve())
    expected = [plan_command(gpt, "m2"), plan_command(haiku, "m2"), plan_command(gpt, "m3"), plan_command(haiku, "m3"), plan_command(gpt, "m3_summary")]
    expected += [plan_command(log, "judge") + f" -S labels={labels.resolve()}" for log in (gpt, haiku)]
    assert command_lines(output) == expected
    assert "monitor cap $12.00: spent $0.0000" in output and "judge cap $12.00" in output
    assert {path: path.read_bytes() for path in (gpt_path, haiku_path)} == before
    assert spend.ledger_entries() == []


def test_the_judge_uses_grade_hiddens_labels_by_default(tmp_path, ledger, no_runs, capsys, monkeypatch):
    root, gpt_path, _, labels = scored_tree(tmp_path)
    monkeypatch.setattr(llm_monitors, "DEFAULT_LABELS", labels)
    assert score_logs.main(["score", str(gpt_path), "--scorers", "judge", "--dry-run", "--scored-root", str(root), "--labels", str(labels)]) == 0
    assert command_lines(capsys.readouterr().out) == [plan_command(str(gpt_path.resolve()), "judge")]


def test_scoring_refuses_over_the_cap_before_any_run(tmp_path, ledger, frozen_target, no_runs, capsys):
    root, _, _, labels = scored_tree(tmp_path)
    usage = ModelUsage(input_tokens=119_999_000, output_tokens=0, total_tokens=119_999_000)
    spend.record_call(scorer="m2", role="monitor", model=HAIKU, usage=usage)
    assert score_logs.main(["score", "--scored-root", str(root), "--labels", str(labels)]) == 1
    captured = capsys.readouterr()
    assert "monitor cap $12.00" in captured.out and "(OVER)" in captured.out
    assert "refused before any call" in captured.err
    assert command_lines(captured.out) == []
    # A dry run reports the same projection and still prints the commands.
    assert score_logs.main(["score", "--dry-run", "--scored-root", str(root), "--labels", str(labels)]) == 0
    assert "(OVER)" in capsys.readouterr().out


def test_scoring_refuses_what_it_must_not_score(tmp_path, ledger, no_runs, capsys, monkeypatch):
    root, gpt_path, _, labels = scored_tree(tmp_path)
    monkeypatch.setattr(summarize, "TARGET_FILE", tmp_path / "not_computed.json")
    outside = write_log(tmp_path / "elsewhere/x.eval", make_log(GPT_OSS, [sample("x1")]))
    assert score_logs.main(["score", str(outside), "--scored-root", str(root), "--labels", str(labels)]) == 2
    assert "only scored copies" in capsys.readouterr().err
    # m3_summary fails closed without the frozen target.
    assert score_logs.main(["score", str(gpt_path), "--scorers", "m3_summary", "--scored-root", str(root)]) == 2
    assert "summary-target" in capsys.readouterr().err
    # Without a ledger nothing runs (a dry run says so and still prints the commands).
    spend.ledger_path().unlink()
    assert score_logs.main(["score", str(gpt_path), "--scorers", "m2", "--scored-root", str(root)]) == 2
    assert "no ledger" in capsys.readouterr().err
    assert score_logs.main(["score", str(gpt_path), "--scorers", "m2", "--dry-run", "--scored-root", str(root)]) == 0
    assert "no projection" in capsys.readouterr().out
    assert score_logs.main(["score", "--scored-root", str(tmp_path / "none")]) == 2


def test_the_target_must_be_counted_like_the_summarizer(tmp_path, ledger, no_runs, capsys, monkeypatch):
    root, gpt_path, _, _ = scored_tree(tmp_path)
    monkeypatch.setattr(summarize, "TARGET_FILE", tmp_path / "t.json")
    summarize.write_target(SummaryTarget(tokens=5, words=4, blocks=1, counter=MODEL, agent=HAIKU, logs=("x",)))
    assert score_logs.main(["score", str(gpt_path), "--scorers", "m3_summary", "--dry-run", "--scored-root", str(root)]) == 2
    assert "counted by mockllm/model" in capsys.readouterr().err


def add_scores(path: Path, name: str, status: str = "scored") -> None:
    log = read_eval_log(str(path))
    for s in log.samples:
        s.scores[name] = Score(value=1, metadata={"status": status})
    write_eval_log(log, str(path))


def test_already_scored_logs_are_skipped_unless_rescored(tmp_path, ledger, no_runs, capsys):
    root, gpt_path, haiku_path, _ = scored_tree(tmp_path)
    add_scores(gpt_path, "m2")
    assert score_logs.main(["score", "--scorers", "m2", "--dry-run", "--scored-root", str(root)]) == 0
    output = capsys.readouterr().out
    assert command_lines(output) == [plan_command(str(haiku_path.resolve()), "m2")] and "skip m2 on" in output
    assert score_logs.main(["score", "--scorers", "m2", "--rescore", "--dry-run", "--scored-root", str(root)]) == 0
    assert len(command_lines(capsys.readouterr().out)) == 2


def test_a_real_run_checks_that_every_sample_was_scored(tmp_path, ledger, monkeypatch, capsys):
    root, gpt_path, haiku_path, _ = scored_tree(tmp_path)
    ran: list[list[str]] = []

    def fake_inspect_score(command: list[str], lock_fd: int | None = None) -> int:
        """Stands in for inspect score: adds the scorer's score (the next free name) to every sample.

        The script passes it the rewrite lock it holds for the log, which no other command can take meanwhile.
        """
        ran.append(command)
        path, name = Path(command[2]), command[command.index("--scorer") + 1].split("@")[1]
        assert os.path.samestat(os.fstat(lock_fd), os.stat(transcript.rewrite_lock_path(path)))
        with pytest.raises(transcript.RewriteLocked):
            transcript.open_rewrite_lock(path)
        taken = score_logs.score_names(read_eval_log(str(path)), name)
        add_scores(path, name if not taken else f"{name}{len(taken)}")
        return 0

    monkeypatch.setattr(score_logs, "run_command", fake_inspect_score)
    assert score_logs.main(["score", "--scorers", "m3", "m2", "--scored-root", str(root)]) == 0
    assert [(Path(c[2]).name, c[c.index("--scorer") + 1].split("@")[1]) for c in ran] == [("a.eval", "m2"), ("b.eval", "m2"), ("a.eval", "m3"), ("b.eval", "m3")]
    assert "m2 on" in capsys.readouterr().out
    assert score_logs.main(["score", "--scorers", "m2", "--rescore", "--scored-root", str(root)]) == 0
    assert "m21" in read_eval_log(str(gpt_path)).samples[0].scores

    monkeypatch.setattr(score_logs, "run_command", lambda command, lock_fd=None: 0)
    assert score_logs.main(["score", str(haiku_path), "--scorers", "m2", "--rescore", "--scored-root", str(root)]) == 1
    assert "expected one new score name" in capsys.readouterr().err
    monkeypatch.setattr(score_logs, "run_command", lambda command, lock_fd=None: 3)
    assert score_logs.main(["score", "--scorers", "m3", "--rescore", "--scored-root", str(root)]) == 1
    assert "failed; stopping" in capsys.readouterr().err


def test_a_run_with_an_errored_sample_fails_and_unscored_samples_warn(tmp_path, ledger, monkeypatch, capsys):
    root, gpt_path, _, _ = scored_tree(tmp_path)

    def scored_as(status: str):
        def run(command: list[str], lock_fd: int | None = None) -> int:
            add_scores(Path(command[2]), "m2", status)
            return 0

        return run

    monkeypatch.setattr(score_logs, "run_command", scored_as("error"))
    assert score_logs.main(["score", str(gpt_path), "--scorers", "m2", "--scored-root", str(root)]) == 1
    assert "{'error': 2}" in capsys.readouterr().out
    write_log(gpt_path, make_log(GPT_OSS, [sample("g1"), sample("g2", epoch=2)], eval_id="gpt"))
    monkeypatch.setattr(score_logs, "run_command", scored_as("unparsed"))
    assert score_logs.main(["score", str(gpt_path), "--scorers", "m2", "--scored-root", str(root)]) == 0
    output = capsys.readouterr().out
    assert "warning: m2 on" in output and "left samples unscored: {'unparsed': 2}" in output


def test_check_run_warns_about_summary_failures_and_long_prompts_and_fails_on_a_missing_score(tmp_path, ledger, frozen_target, monkeypatch, capsys):
    root, gpt_path, _, _ = scored_tree(tmp_path)

    def run(command: list[str], lock_fd: int | None = None) -> int:
        """Scores the first sample summary_failed and the second with a prompt over 100K input tokens."""
        log = read_eval_log(command[2])
        first, second = log.samples
        first.scores["m3_summary"] = Score(value=float("nan"), metadata={"status": "summary_failed", "over_100k_input": False})
        second.scores["m3_summary"] = Score(value=5, metadata={"status": "scored", "over_100k_input": True})
        write_eval_log(log, command[2])
        return 0

    monkeypatch.setattr(score_logs, "run_command", run)
    assert score_logs.main(["score", str(gpt_path), "--scorers", "m3_summary", "--scored-root", str(root)]) == 0
    output = capsys.readouterr().out
    assert "left samples unscored: {'summary_failed': 1}" in output
    assert "warning: m3_summary on" in output and "1 samples had a prompt over 100K input tokens" in output

    def score_one(command: list[str], lock_fd: int | None = None) -> int:
        log = read_eval_log(command[2])
        log.samples[0].scores["m21"] = Score(value=5, metadata={"status": "scored"})
        write_eval_log(log, command[2])
        return 0

    add_scores(gpt_path, "m2")
    monkeypatch.setattr(score_logs, "run_command", score_one)
    assert score_logs.main(["score", str(gpt_path), "--scorers", "m2", "--rescore", "--scored-root", str(root)]) == 1
    assert "'missing': 1" in capsys.readouterr().out


def test_the_judge_cap_refuses_before_any_run(tmp_path, ledger, no_runs, capsys):
    root, gpt_path, _, labels = scored_tree(tmp_path)
    # One earlier $7 judge prompt: $7 spent, and the one planned prompt is projected at that ledger mean.
    usage = ModelUsage(input_tokens=3_500_000, output_tokens=0, total_tokens=3_500_000)
    spend.record_call(scorer="judge", role="judge", model=llm_monitors.ROLE_MODELS["judge"], usage=usage)
    assert spend.spent(spend.ledger_entries())["judge"] == pytest.approx(7.0)
    assert score_logs.main(["score", str(gpt_path), "--scorers", "judge", "--scored-root", str(root), "--labels", str(labels)]) == 1
    captured = capsys.readouterr()
    assert "judge cap $12.00: spent $7.0000 + all planned runs $7.0000 = $14.0000 (OVER)" in captured.out
    assert "monitor cap" not in captured.out and "refused before any call" in captured.err
    assert command_lines(captured.out) == []


def test_the_test_retest_subset(tmp_path, ledger, no_runs, capsys, monkeypatch):
    """--subset re-scores only the listed samples with m2 and m3, and skips logs that hold none of them."""
    root, gpt_path, haiku_path, labels = scored_tree(tmp_path)
    for path in (gpt_path, haiku_path):
        add_scores(path, "m2")
        add_scores(path, "m3")
    subset = tmp_path / "retest.txt"
    subset.write_text("g2\n")
    assert score_logs.main(["score", "--subset", str(subset), "--rescore", "--dry-run", "--scored-root", str(root)]) == 0
    output = capsys.readouterr().out
    gpt = str(gpt_path.resolve())
    assert command_lines(output) == [plan_command(gpt, name) + f" -S subset={subset.resolve()}" for name in ("m2", "m3")]
    assert "skip m2 on" in output and "no sample of the subset" in output
    assert "monitor anthropic/claude-haiku-5-5 1 prompts" in output
    assert score_logs.main(["score", "--subset", str(subset), "--dry-run", "--scored-root", str(root)]) == 2
    assert "it needs --rescore" in capsys.readouterr().err
    assert score_logs.main(["score", "--subset", str(subset), "--scorers", "m2", "judge", "--rescore", "--dry-run", "--scored-root", str(root), "--labels", str(labels)]) == 2
    assert "re-scores only m2, m3, not judge" in capsys.readouterr().err
    # The subset's scorers are checked before the label rows are read, so a missing labels file is not the refusal named.
    monkeypatch.setattr(llm_monitors, "DEFAULT_LABELS", tmp_path / "missing/labels.jsonl")
    assert score_logs.main(["score", "--subset", str(subset), "--scorers", "m2", "judge", "--rescore", "--dry-run", "--scored-root", str(root)]) == 2
    assert "re-scores only m2, m3, not judge" in capsys.readouterr().err
    subset.write_text("g2\nnot-a-sample\n")
    assert score_logs.main(["score", "--subset", str(subset), "--rescore", "--dry-run", "--scored-root", str(root)]) == 2
    assert "1 subset uuids are in none of the given logs, such as not-a-sample" in capsys.readouterr().err
    with pytest.raises(ValueError, match="only m2, m3 take a subset"):
        score_logs.score_command(gpt, "judge", subset="retest.txt")


def test_the_subset_goes_only_beside_a_full_run_of_each_scorer(tmp_path, ledger, no_runs, capsys):
    """A subset run on a log without the full run would take the primary name (m2, with the rest not_selected)."""
    root, gpt_path, haiku_path, _ = scored_tree(tmp_path)
    subset = tmp_path / "retest.txt"
    subset.write_text("g2\n")
    gpt = score_logs.shown(gpt_path)
    for rescore in ([], ["--rescore"]):
        assert score_logs.main(["score", "--subset", str(subset), *rescore, "--dry-run", "--scored-root", str(root)]) == 2
        captured = capsys.readouterr()
        assert command_lines(captured.out) == []
    assert f"score them without --subset first: m2 on {gpt}, m3 on {gpt}" in captured.err
    # A full m2 run is not enough for m3, and a subset-only run (the other samples not_selected) is not a full run.
    add_scores(gpt_path, "m2")
    assert score_logs.main(["score", "--subset", str(subset), "--rescore", "--dry-run", "--scored-root", str(root)]) == 2
    assert f"first: m3 on {gpt}" in capsys.readouterr().err
    add_scores(gpt_path, "m3", "not_selected")
    assert score_logs.main(["score", "--subset", str(subset), "--rescore", "--scorers", "m3", "--dry-run", "--scored-root", str(root)]) == 2
    assert f"first: m3 on {gpt}" in capsys.readouterr().err
    # Only the logs the subset re-scores need the full run: haiku's log holds no subset sample.
    assert score_logs.main(["score", "--subset", str(subset), "--rescore", "--scorers", "m2", "--dry-run", "--scored-root", str(root)]) == 0
    assert command_lines(capsys.readouterr().out) == [plan_command(str(gpt_path.resolve()), "m2") + f" -S subset={subset.resolve()}"]


def test_no_scored_root_inside_logs_but_logs_scored(tmp_path, no_runs, capsys):
    """The scored-root test hook cannot point the scripts at agent logs (nothing under the repo's logs/ is read or written)."""
    for command in ("m1", "score"):
        assert score_logs.main([command, "--scored-root", str(ROOT / "logs/main")]) == 2
        assert "agent logs are never scored in place" in capsys.readouterr().err
    logs, source, _ = agent_tree(tmp_path)
    assert score_logs.main(["copy", str(source), "--logs-root", str(logs), "--scored-root", str(ROOT / "logs/pilot/scored")]) == 2
    assert "agent logs are never scored in place" in capsys.readouterr().err
    assert not (ROOT / "logs/pilot/scored").exists()
    # A root that contains logs/ (the repo root, or logs/ itself) would reach every agent log.
    for scored_root in (ROOT, ROOT / "logs", ROOT.parent):
        for command in ("m1", "score"):
            assert score_logs.main([command, "--dry-run", "--scored-root", str(scored_root)]) == 2
            assert "agent logs are never scored in place" in capsys.readouterr().err



def test_another_spelling_of_logs_is_the_same_directory(tmp_path, monkeypatch, no_runs, capsys):
    """On a case-insensitive file system (macOS APFS by default) LOGS/main is logs/main, so it is refused as a scored root."""
    monkeypatch.setattr(score_logs, "LOGS_ROOT", tmp_path / "logs")
    monkeypatch.setattr(score_logs, "SCORED_ROOT", tmp_path / "logs/scored")
    write_log(tmp_path / "logs/main/haiku-normal/a.eval", make_log(HAIKU, [sample("h1", variant="normal", messages=episode("claude"))]))
    (tmp_path / "logs/scored").mkdir()
    case_insensitive = (tmp_path / "LOGS").exists()
    for spelling in ("LOGS/main", "Logs/Main/haiku-normal"):
        assert (tmp_path / spelling).exists() == case_insensitive
        if case_insensitive:
            assert score_logs.main(["m1", "--dry-run", "--scored-root", str(tmp_path / spelling)]) == 2
            assert "is inside" in capsys.readouterr().err
    if case_insensitive:
        # The scored root under another spelling is still the scored root: allowed (and empty here).
        assert score_logs.main(["m1", "--dry-run", "--scored-root", str(tmp_path / "LOGS/scored")]) == 2
        assert "no scored logs" in capsys.readouterr().err


def test_a_symlinked_logs_scored_is_refused(tmp_path, monkeypatch, no_runs, capsys):
    """Every check compares resolved paths with logs/scored, so a logs/scored linked to logs/main would pass agent logs as copies."""
    logs = tmp_path / "logs"
    monkeypatch.setattr(score_logs, "LOGS_ROOT", logs)
    monkeypatch.setattr(score_logs, "SCORED_ROOT", logs / "scored")
    write_log(logs / "main/haiku-normal/a.eval", make_log(HAIKU, [sample("h1", variant="normal", messages=episode("claude"))]))
    write_log(logs / "pilot/haiku-normal/b.eval", make_log(HAIKU, [sample("h2", variant="normal", messages=episode("claude"))]))
    (logs / "scored").symlink_to(logs / "main", target_is_directory=True)
    before = snapshot(logs)
    for args in (["m1"], ["m1", "--dry-run"], ["m1", str(logs / "scored/haiku-normal/a.eval")], ["score", "--scorers", "m2", "--dry-run"]):
        assert score_logs.main(args) == 2
        captured = capsys.readouterr()
        assert "is a symlink or resolves to" in captured.err and command_lines(captured.out) == []
    assert copy(logs / "pilot", logs=logs, scored=logs / "scored") == 2
    assert "is a symlink or resolves to" in capsys.readouterr().err
    assert snapshot(logs) == before and not (logs / "main/pilot").exists()
    # A scored root that is itself a symlink is refused too, wherever it points.
    (logs / "scored").unlink()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "linked").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    assert score_logs.main(["m1", "--dry-run", "--scored-root", str(tmp_path / "linked")]) == 2
    assert "is a symlink; name the scored directory itself" in capsys.readouterr().err


def test_symlinked_directories_are_refused(tmp_path, no_runs, capsys):
    root, gpt_path, _, _ = scored_tree(tmp_path)
    (root / "linked").symlink_to(gpt_path.parent, target_is_directory=True)
    assert score_logs.main(["m1", "--dry-run", "--scored-root", str(root)]) == 2
    assert "holds symlinked directories" in capsys.readouterr().err


def test_inspect_score_runs_without_inspect_score_variables(monkeypatch, command_dir):
    """inspect score reads INSPECT_SCORE_STREAM and the like for any option not given, so they are removed."""
    monkeypatch.setenv("INSPECT_SCORE_STREAM", "1")
    monkeypatch.setenv("INSPECT_SCORE_SCORER_ARGS", "log=logs/main/x.eval")
    monkeypatch.setenv(transcript.REWRITE_LOCK_FD_ENV, "3")
    seen: list[dict] = []
    monkeypatch.setattr(score_logs.subprocess, "run", lambda command, **kwargs: seen.append(kwargs) or subprocess_result())
    assert score_logs.run_command(["inspect", "score", "x.eval"]) == 0
    assert score_logs.run_command(["inspect", "score", "x.eval"], 7) == 0
    first, second = seen
    # COMMAND_DIR is the repo root (the command_dir fixture checks that) and here a tmp folder.
    assert first["cwd"] == command_dir and not [key for key in first["env"] if key.startswith("INSPECT_SCORE_")]
    assert first["env"]["PATH"] == os.environ["PATH"]
    # A rewrite lock's descriptor reaches the command only when the script passes the one it holds.
    assert transcript.REWRITE_LOCK_FD_ENV not in first["env"] and first["pass_fds"] == ()
    assert second["env"][transcript.REWRITE_LOCK_FD_ENV] == "7" and second["pass_fds"] == (7,)


@pytest.mark.parametrize(
    "line",
    ["INSPECT_SCORE_STREAM=1", "  export INSPECT_SCORE_STREAM = true", "'INSPECT_SCORE_STREAM'=1", "INSPECT_SCORE_OUTPUT_FILE=/tmp/x.eval"],
)
def test_m1_and_score_refuse_inspect_score_variables_in_the_dotenv_file(tmp_path, command_dir, ledger, frozen_target, no_runs, capsys, line):
    """inspect loads the .env file from its working directory up before it reads its options, so command_env cannot remove them."""
    root, gpt_path, _, labels = scored_tree(tmp_path)
    secret = "do-not-print-this-value"
    (tmp_path / ".env").write_text(f"SOME_API_KEY={secret}\n{line}\n# INSPECT_SCORE_SCORER_ARGS=commented out\n")
    key = re.search(r"INSPECT_SCORE_\w+", line).group(0)
    for args in (["m1", "--scored-root", str(root)], ["score", "--scored-root", str(root), "--labels", str(labels)], ["score", "--dry-run", "--scored-root", str(root)]):
        assert score_logs.main(args) == 2
        captured = capsys.readouterr()
        assert f"remove these inspect score option variables from {tmp_path / '.env'} first: {key}" in captured.err
        assert secret not in captured.out + captured.err and "SCORER_ARGS" not in captured.err
        assert command_lines(captured.out) == []
    # The first .env walking up from the command folder is the one inspect loads; without such a key both run.
    (command_dir / ".env").write_text(f"SOME_API_KEY={secret}\nINSPECT_EVAL_LOG_DIR=x\n")
    assert score_logs.main(["m1", "--dry-run", "--scored-root", str(root)]) == 0
    assert score_logs.main(["score", str(gpt_path), "--scorers", "m2", "--dry-run", "--scored-root", str(root)]) == 0
    assert secret not in capsys.readouterr().out


def subprocess_result():
    class Done:
        returncode = 0

    return Done()


def test_m1_runs_on_each_scored_copy(tmp_path, no_runs, capsys):
    root, gpt_path, haiku_path, _ = scored_tree(tmp_path)
    assert score_logs.main(["m1", "--dry-run", "--scored-root", str(root)]) == 0
    assert command_lines(capsys.readouterr().out) == [shlex.join(score_logs.m1_command(str(p.resolve()))) for p in (gpt_path, haiku_path)]


def test_m1_and_score_refuse_while_another_command_rewrites_a_copy(tmp_path, ledger, frozen_target, no_runs, capsys):
    """inspect score --overwrite writes the whole log back, so a second command on a copy would drop one's scores."""
    root, gpt_path, haiku_path, labels = scored_tree(tmp_path)
    before = {path: path.read_bytes() for path in (gpt_path, haiku_path)}
    with rewrite_lock_held(haiku_path, tmp_path):
        for args in (["m1"], ["m1", str(haiku_path), "--rescore"], ["score", "--labels", str(labels)], ["score", str(haiku_path), "--scorers", "m2"]):
            assert score_logs.main([*args, "--scored-root", str(root)]) == 2
            captured = capsys.readouterr()
            assert f"is rewriting {haiku_path}" in captured.err and command_lines(captured.out) == []
            # A refused command holds no lock, so the other copy is free.
            os.close(transcript.open_rewrite_lock(gpt_path))
        # A dry run rewrites nothing, so it takes no lock.
        assert score_logs.main(["m1", "--dry-run", "--scored-root", str(root)]) == 0
        assert len(command_lines(capsys.readouterr().out)) == 2
    assert {path: path.read_bytes() for path in (gpt_path, haiku_path)} == before
    assert spend.ledger_entries() == []


def test_m1_passes_its_rewrite_lock_to_the_inspect_score_it_runs(tmp_path, monkeypatch, capsys):
    """A real M1 run (no model call): the inspect score command holds the lock score_logs holds instead of refusing it."""
    root, gpt_path, _, _ = scored_tree(tmp_path)
    # COMMAND_DIR is a tmp folder (the command_dir fixture), so the scorer is named by its absolute path.
    monkeypatch.setattr(score_logs, "M1_SCORER", f"{ROOT / 'src/rewardhack/monitors/m1_regex.py'}@m1")
    for key in [key for key in os.environ if key.endswith("_API_KEY")]:
        monkeypatch.delenv(key)
    assert score_logs.main(["m1", str(gpt_path), "--scored-root", str(root)]) == 0, capsys.readouterr()
    scores = [sample.scores for sample in read_eval_log(str(gpt_path)).samples]
    assert [set(found) for found in scores] == [{"episode_record", "m1"}] * 2
    # The script and its command released the lock when they finished.
    os.close(transcript.open_rewrite_lock(gpt_path))


# --- copy ---


def agent_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    logs = tmp_path / "logs"
    source = logs / "main/haiku-normal"
    write_log(source / "a.eval", make_log(HAIKU, [sample("h1", variant="normal", messages=episode("claude"))]))
    (source / "logs.json").write_text("{}")
    (source / ".eval-set-id").write_text("set")
    return logs, source, logs / "scored"


def snapshot(directory: Path) -> dict[str, tuple[bytes, int]]:
    return {str(p.relative_to(directory)): (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(directory.rglob("*")) if p.is_file()}


def copy(*args: str | Path, logs: Path, scored: Path) -> int:
    return score_logs.main(["copy", *map(str, args), "--logs-root", str(logs), "--scored-root", str(scored)])


def test_copy_writes_each_log_once_and_never_touches_the_source(tmp_path, capsys):
    logs, source, scored = agent_tree(tmp_path)
    before = snapshot(source)
    assert copy(source, logs=logs, scored=scored) == 0
    destination = scored / "main/haiku-normal/a.eval"
    assert destination.read_bytes() == (source / "a.eval").read_bytes()
    assert [p.name for p in scored.rglob("*") if p.is_file()] == ["a.eval"]
    assert snapshot(source) == before

    destination.write_bytes(b"scored since")
    assert copy(source, logs=logs, scored=scored) == 2
    assert "exists; each log is copied once" in capsys.readouterr().err
    assert destination.read_bytes() == b"scored since"
    assert snapshot(source) == before


def test_a_copy_that_does_not_match_its_source_is_never_linked_into_place(tmp_path, monkeypatch, capsys):
    logs, source, scored = agent_tree(tmp_path)
    before = snapshot(source)

    def truncated_copy(src: Path, dst: Path) -> None:
        Path(dst).write_bytes(Path(src).read_bytes()[:-1])

    monkeypatch.setattr(score_logs.shutil, "copyfile", truncated_copy)
    with pytest.raises(RuntimeError, match="does not match it"):
        copy(source, logs=logs, scored=scored)
    assert [p for p in scored.rglob("*") if p.is_file()] == []
    assert snapshot(source) == before


def test_copy_never_replaces_a_destination_that_appears_during_the_copy(tmp_path, monkeypatch, capsys):
    """The copy is linked into place, which fails rather than replace a file written after copy_plan checked."""
    logs, source, scored = agent_tree(tmp_path)
    original = score_logs.shutil.copyfile
    destination = scored / "main/haiku-normal/a.eval"

    def copy_then_appear(src: Path, dst: Path) -> None:
        original(src, dst)
        destination.write_bytes(b"written meanwhile")

    monkeypatch.setattr(score_logs.shutil, "copyfile", copy_then_appear)
    assert copy(source, logs=logs, scored=scored) == 2
    assert "File exists" in capsys.readouterr().err
    assert destination.read_bytes() == b"written meanwhile"
    assert [p.name for p in scored.rglob("*") if p.is_file()] == ["a.eval"]


def test_copy_refuses_everything_when_any_copy_exists(tmp_path):
    logs, source, scored = agent_tree(tmp_path)
    other = logs / "main/gpt-oss-normal"
    write_log(other / "b.eval", make_log(GPT_OSS, [sample("g1", variant="normal")]))
    assert copy(source, logs=logs, scored=scored) == 0
    assert copy(other, source, logs=logs, scored=scored) == 2
    assert not (scored / "main/gpt-oss-normal").exists()
    assert copy(other, logs=logs, scored=scored) == 0


def test_copy_refuses_paths_it_must_not_copy(tmp_path, capsys):
    logs, source, scored = agent_tree(tmp_path)
    elsewhere = write_log(tmp_path / "elsewhere/x.eval", make_log(HAIKU, [sample("h1")]))
    assert copy(elsewhere, logs=logs, scored=scored) == 2
    assert copy(logs, logs=logs, scored=scored) == 2
    assert "contains" in capsys.readouterr().err
    copy(source, logs=logs, scored=scored)
    assert copy(scored / "main", logs=logs, scored=scored) == 2
    assert "scored copy" in capsys.readouterr().err
    assert copy(logs / "missing", logs=logs, scored=scored) == 2


def symlinked_scored_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """logs/ with agent logs under main/ and pilot/ and an empty logs/scored, as the script's default roots."""
    logs = tmp_path / "logs"
    monkeypatch.setattr(score_logs, "LOGS_ROOT", logs)
    monkeypatch.setattr(score_logs, "SCORED_ROOT", logs / "scored")
    for name, path in (("h1", "main/haiku-normal/main-run.eval"), ("s1", "main/sonnet-normal/s-run.eval"), ("p1", "pilot/haiku-normal/p.eval")):
        write_log(logs / path, make_log(HAIKU, [sample(name, variant="normal", messages=episode("claude"))]))
    (logs / "scored").mkdir()
    return logs


def test_copy_never_writes_through_a_symlink_in_the_scored_root(tmp_path, monkeypatch, capsys):
    """A symlinked directory inside logs/scored would put a copy into logs/pilot or logs/main, so copy refuses and writes nothing."""
    logs = symlinked_scored_tree(tmp_path, monkeypatch)
    scored = logs / "scored"
    before = snapshot(logs)
    (scored / "main").symlink_to(Path("../pilot"), target_is_directory=True)
    assert copy(logs / "main/haiku-normal", logs=logs, scored=scored) == 2
    assert "holds symlinked directories, which copies would be written through" in capsys.readouterr().err
    assert snapshot(logs) == before
    (scored / "main").unlink()
    (scored / "main").mkdir()
    (scored / "main/sonnet-normal").symlink_to(Path("../../main/haiku-normal"), target_is_directory=True)
    assert copy(logs / "main/sonnet-normal", logs=logs, scored=scored) == 2
    assert "holds symlinked directories" in capsys.readouterr().err
    assert snapshot(logs) == before
    # A dangling link is no directory, so each destination's resolved path is checked too.
    (scored / "main/sonnet-normal").unlink()
    (scored / "main/sonnet-normal").symlink_to(Path("../../main/new"), target_is_directory=True)
    assert copy(logs / "main/sonnet-normal", logs=logs, scored=scored) == 2
    assert "is reached through a symlink, to" in capsys.readouterr().err
    assert snapshot(logs) == before and not (logs / "main/new").exists()


def test_copy_never_writes_through_a_symlink_made_after_the_plan(tmp_path, monkeypatch, capsys):
    logs = symlinked_scored_tree(tmp_path, monkeypatch)
    scored = logs / "scored"
    before = snapshot(logs)
    original = score_logs.copy_plan

    def plan_then_link(*args: object) -> list[tuple[Path, Path]]:
        pairs = original(*args)
        (scored / "main").symlink_to(Path("../pilot"), target_is_directory=True)
        return pairs

    monkeypatch.setattr(score_logs, "copy_plan", plan_then_link)
    with pytest.raises(RuntimeError, match="through a symlink made after the copy was planned"):
        copy(logs / "main/haiku-normal", logs=logs, scored=scored)
    assert snapshot(logs) == before and sorted(p.name for p in (logs / "pilot/haiku-normal").iterdir()) == ["p.eval"]


def test_copy_takes_only_finished_logs(tmp_path, capsys):
    logs, _, scored = agent_tree(tmp_path)
    for status in ("error", "started"):
        log = make_log(GPT_OSS, [sample("g1")])
        log.status = status
        write_log(logs / f"main/{status}/x.eval", log)
    assert copy(logs / "main/error", logs=logs, scored=scored) == 2
    assert "has status error" in capsys.readouterr().err
    assert copy(logs / "main/started", logs=logs, scored=scored) == 2
    assert score_logs.main(["copy", str(logs / "main/started"), "--allow-error", "--logs-root", str(logs), "--scored-root", str(scored)]) == 2
    assert score_logs.main(["copy", str(logs / "main/error"), "--allow-error", "--logs-root", str(logs), "--scored-root", str(scored)]) == 0


# --- the summary target ---


def haiku_logs(tmp_path: Path) -> Path:
    directory = tmp_path / "pilot"
    haiku = make_log(HAIKU, [sample("h1", messages=episode("claude", ("one two three", "one two three four five")))])
    write_log(directory / "haiku-normal/a.eval", haiku)
    gpt = make_log(GPT_OSS, [sample("g1", messages=episode("gpt-oss", ("a " * 500, "b " * 500)))])
    write_log(directory / "gpt-oss-normal/b.eval", gpt)
    return directory


def test_the_summary_target_is_computed_once(tmp_path, capsys):
    directory, out = haiku_logs(tmp_path), tmp_path / "summary_target.json"
    assert score_logs.main(["summary-target", str(directory), "--out", str(out), "--dry-run"]) == 0
    assert "would count 2 blocks with anthropic/claude-haiku-5-5" in capsys.readouterr().out and not out.exists()
    assert score_logs.main(["summary-target", str(directory), "--out", str(out), "--counter-model", MODEL]) == 0
    target = summarize.load_target(out)
    assert (target.words, target.blocks, target.counter, target.agent) == (4, 2, MODEL, HAIKU)
    assert target.logs == (str((directory / "haiku-normal/a.eval").resolve()),)
    assert 3 <= target.tokens <= 8
    frozen = out.read_text()
    assert score_logs.main(["summary-target", str(directory), "--out", str(out), "--counter-model", MODEL]) == 2
    assert "frozen" in capsys.readouterr().err and out.read_text() == frozen
    assert json.loads(frozen)["version"] == 1
