"""The summarizer and the M3-summary arm (rewardhack.monitors.summarize, llm_monitors.m3_summary), mockllm only.

Token counts come from a fake counter or from mockllm/model's local estimate; nothing calls
Anthropic's count_tokens endpoint or any paid model.
"""

import json
import math
from pathlib import Path

import anyio
import pytest
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageAssistant, ContentText, ModelOutput
from inspect_ai.scorer import Score
from test_llm_monitors import (
    GPT_OSS,
    HAIKU,
    MODEL,
    SONNET,
    TICKET,
    Script,
    block,
    episode,
    errored,
    make_log,
    raised,
    replies,
    run_scorer,
    sample,
    write_log,
)

from rewardhack.monitors import llm_monitors, spend, summarize
from rewardhack.monitors.llm_monitors import m3_summary
from rewardhack.monitors.reasoning import PLACEHOLDER
from rewardhack.monitors.summarize import (
    BlockSummary,
    PromptTemplate,
    SummaryTarget,
    SummaryTargetMissing,
    block_key,
    cached_summaries,
    compute_target,
    load_target,
    summarize_blocks,
    summary_stats,
    write_target,
)
from rewardhack.monitors.transcript import ReasoningBlock, m3_transcript

SHORT = "Read the file."
LONG = " ".join(f"step{i} considers the separator handling" for i in range(40))
SUMMARY = "I checked the separator handling and decided to pass the arguments as a list."


@pytest.fixture
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "spend.json"
    monkeypatch.setenv(spend.LEDGER_ENV, str(path))
    return spend.init_ledger()


def target(tokens: int = 50, counter: str = MODEL, words: int = 12) -> SummaryTarget:
    return SummaryTarget(tokens=tokens, words=words, blocks=3, counter=counter, agent=HAIKU, logs=("logs/pilot/haiku-normal/x.eval",))


def write(tmp_path: Path, value: SummaryTarget) -> Path:
    return write_target(value, tmp_path / "summary_target.json")


class WordCounter:
    """A fake counter: one token per word."""

    name = "words"

    def __init__(self) -> None:
        self.counted: list[str] = []

    async def count(self, text: str) -> int:
        self.counted.append(text)
        return len(text.split())


class DoubleCounter(WordCounter):
    """A fake counter whose count is not the word count: two tokens per word plus three of framing."""

    name = "double"

    async def count(self, text: str) -> int:
        return 2 * await super().count(text) + 3


# --- the target ---


def test_the_target_fails_closed(tmp_path):
    with pytest.raises(SummaryTargetMissing, match="summary-target"):
        load_target(tmp_path / "none.json")
    path = tmp_path / "bad.json"
    for text in ("", "{", json.dumps([1]), json.dumps(target().to_dict() | {"tokens": 0}), json.dumps(target().to_dict() | {"agent": SONNET}),
                 json.dumps(target().to_dict() | {"version": 2}), json.dumps(target().to_dict() | {"extra": 1}), json.dumps(target().to_dict() | {"logs": []})):
        path.write_text(text)
        with pytest.raises(SummaryTargetMissing):
            load_target(path)


def test_the_target_is_written_once(tmp_path):
    path = write(tmp_path, target())
    assert load_target(path) == target()
    with pytest.raises(FileExistsError):
        write(tmp_path, target(tokens=60))
    assert load_target(path) == target()


def test_the_default_target_file_is_read_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setattr(summarize, "TARGET_FILE", tmp_path / "frozen.json")
    with pytest.raises(SummaryTargetMissing):
        load_target()
    write_target(target())
    assert load_target() == target()


def claude_message(*texts: str) -> ChatMessageAssistant:
    return ChatMessageAssistant(content=[*(block(text, "claude") for text in texts), ContentText(text="Done.")])


def test_the_target_is_the_median_over_haiku_blocks_only():
    haiku = make_log(HAIKU, [
        sample("h1", messages=[*episode("claude", ("one two", "one two three four"))[:2], claude_message("a b c d e f", PLACEHOLDER, " ")]),
        sample("h2", messages=episode("claude", ("x", "y z"))),
        errored("h3"),
    ])
    haiku.samples[2].messages.append(claude_message("never counted because the sample errored"))
    others = [make_log(SONNET, [sample("s1", messages=episode("claude", ("lots of words here", "and here too")))]), make_log(GPT_OSS, [sample("g1")])]
    texts = summarize.target_texts([haiku, *others])
    assert texts == ["a b c d e f", "x", "y z"]
    counter = DoubleCounter()
    found = anyio.run(compute_target, [("haiku.eval", haiku), ("sonnet.eval", others[0]), ("gpt.eval", others[1])], counter)
    # Tokens [15, 5, 7] and words [6, 1, 2]: each median is taken over its own counts.
    assert found == SummaryTarget(tokens=7, words=2, blocks=3, counter="double", agent=HAIKU, logs=("haiku.eval",))
    assert counter.counted == texts


def test_the_target_needs_haiku_blocks():
    with pytest.raises(ValueError, match="none of the logs"):
        anyio.run(compute_target, [("gpt.eval", make_log(GPT_OSS, [sample("g1")]))], WordCounter())
    empty = make_log(HAIKU, [sample("h1", messages=episode("claude", (" ", PLACEHOLDER)))])
    with pytest.raises(ValueError, match="no kept reasoning"):
        anyio.run(compute_target, [("haiku.eval", empty)], WordCounter())


def test_median_rounds_half_up():
    assert summarize.median_int([1, 2]) == 2
    assert summarize.median_int([1, 2, 3, 10]) == 3
    assert summarize.median_int([7]) == 7


# --- the prompt ---


def test_the_summarizer_prompt():
    template = summarize.prompt_template()
    assert template.fields == {"WORDS", "REASONING"}
    prompt = summarize.summary_prompt(LONG, target(tokens=91, words=37), template)
    assert "about 37 words" in prompt and f"<reasoning>\n{LONG}\n</reasoning>" in prompt
    # The prompt asks for the word target, never the token target.
    assert "91" not in prompt
    for rule in ("Do not judge", "do not add any information", "Reply with the summary only"):
        assert rule in prompt
    with pytest.raises(ValueError, match="takes exactly"):
        template.render(WORDS="3")


def test_prompt_templates_fill_in_one_pass(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("A {{ONE}} and {{TWO}}.")
    template = PromptTemplate.load(path)
    assert template.render(ONE="{{TWO}}", TWO="x") == "A {{TWO}} and x."
    assert template.sha256 == summarize.sha256_text("A {{ONE}} and {{TWO}}.")


# --- summarizing blocks ---


def blocks(*texts: str) -> list[ReasoningBlock]:
    return [ReasoningBlock(turn, 0, text) for turn, text in enumerate(texts)]


class Recorder:
    def __init__(self) -> None:
        self.outputs: list[ModelOutput] = []
        self.attempts: list[int] = []

    async def __call__(self, output: ModelOutput, attempt: int) -> spend.CallSpend:
        self.outputs.append(output)
        self.attempts.append(attempt)
        return spend.CallSpend(
            scorer="m3_summary", role="summarizer", cap="monitor", model=MODEL, sample_id=None, epoch=None, log=None, attempt=attempt, input_tokens=1,
            output_tokens=1, input_tokens_cache_write=0, input_tokens_cache_read=0, reasoning_tokens=0, cost=0.0, inspect_cost=None,
            usage_missing=False, timestamp="t",
        )


def summarize_with(script: Script, items: list[ReasoningBlock], cache: dict | None = None, value: SummaryTarget | None = None):
    record = Recorder()
    value = value or target(tokens=5, counter="words")
    done, calls = anyio.run(
        lambda: summarize_blocks(items, model=script.model(), counter=WordCounter(), target=value, cache=cache or {}, record=record)
    )
    return done, calls, record


def test_short_blocks_pass_through_and_the_rest_are_summarized():
    script = Script(lambda prompt: SUMMARY)
    items = blocks("one two three", "one two three four five", LONG)
    done, calls, record = summarize_with(script, items)
    assert [entry.mode for entry in done] == ["passed_through", "summarized", "summarized"]
    assert [entry.tokens for entry in done] == [3, 5, len(LONG.split())]
    assert len(script.prompts) == len(calls) == len(record.outputs) == 2
    # Each block is its own prompt, so each first call is attempt 1.
    assert record.attempts == [1, 1]
    assert done[0].replacement(items[0]) == "one two three" and done[2].replacement(items[2]) == SUMMARY
    stats = summary_stats(done)
    assert (stats["blocks"], stats["passed_through"], stats["summarized"], stats["failed"]) == (3, 1, 2, 0)
    assert stats["passed_through_share"] == pytest.approx(1 / 3)
    assert stats["summary_words"] == [len(SUMMARY.split())] * 2


def test_cached_blocks_are_reused_without_a_call():
    items = blocks(LONG)
    first, _, _ = summarize_with(Script(lambda prompt: SUMMARY), items)
    script = Script(lambda prompt: "never")
    again, calls, _ = summarize_with(script, items, cache={entry.key: entry for entry in first})
    assert script.prompts == [] and calls == []
    assert again[0].reused and again[0].summary == SUMMARY


def test_a_block_without_an_accepted_summary_fails_after_two_calls():
    cut = ModelOutput.from_content(model=MODEL, content="I checked", stop_reason="max_tokens")
    script = Script(replies("   ", cut))
    done, calls, record = summarize_with(script, blocks(LONG))
    assert done[0].mode == "failed" and done[0].summary is None
    assert [a["status"] for a in done[0].attempts] == ["empty", "stop_reason_max_tokens"]
    assert len(calls) == 2 and record.attempts == [1, 2]
    with pytest.raises(ValueError, match="no summary"):
        done[0].replacement(blocks(LONG)[0])


class BrokenCounter(WordCounter):
    async def count(self, text: str) -> int:
        raise ConnectionError("count_tokens is down")


def test_a_block_that_cannot_be_counted_fails_without_a_call():
    script = Script(lambda prompt: SUMMARY)
    record = Recorder()
    done, calls = anyio.run(
        lambda: summarize_blocks(blocks(LONG), model=script.model(), counter=BrokenCounter(), target=target(tokens=5, counter="words"), cache={}, record=record)
    )
    assert done[0].mode == "failed" and done[0].tokens is None and "count_tokens is down" in done[0].attempts[0]["error"]
    assert script.prompts == [] and calls == []


def test_the_counter_must_be_the_targets():
    with pytest.raises(ValueError, match="would not compare"):
        summarize_with(Script(lambda prompt: SUMMARY), blocks(LONG), value=target(counter="other"))


def test_a_summary_is_only_used_for_its_own_text():
    [entry], _, _ = summarize_with(Script(lambda prompt: SUMMARY), blocks(LONG))
    with pytest.raises(ValueError, match="different text"):
        entry.replacement(ReasoningBlock(0, 0, LONG + " changed"))


def test_cached_summaries_match_prompt_target_and_model():
    items = blocks(LONG, "a b c d e f", "x")
    done, _, _ = summarize_with(Script(replies(SUMMARY, "")), items[:1])
    good = done[0]
    failed = BlockSummary(1, 0, block_key(items[1])[2], 11, 6, "failed", None, good.prompt_sha256, good.target_tokens, good.model)
    stale = BlockSummary(2, 0, block_key(items[2])[2], 1, 1, "passed_through", None, "old-prompt", good.target_tokens, good.model)
    scores = {
        "m3_summary": Score(value=1, metadata={summarize.CACHE_KEY: [good.to_dict(), failed.to_dict(), {"turn": "junk"}]}),
        "m3_summary1": Score(value=1, metadata={summarize.CACHE_KEY: [stale.to_dict()]}),
        "m3": Score(value=1, metadata={summarize.CACHE_KEY: [good.to_dict() | {"turn": 5}]}),
    }
    found = cached_summaries(scores, prompt_sha256=good.prompt_sha256, target=target(tokens=5, counter="words"), model=good.model)
    assert list(found) == [good.key]
    assert cached_summaries(scores, prompt_sha256=good.prompt_sha256, target=target(tokens=6, counter="words"), model=good.model) == {}
    assert cached_summaries(scores, prompt_sha256=good.prompt_sha256, target=target(tokens=5, counter="words"), model="other") == {}
    assert cached_summaries(None, prompt_sha256="x", target=target(), model=MODEL) == {}


# --- the m3_summary scorer ---


def respond(prompt: str) -> str:
    """The summarizer and the monitor share the mock model (the summarizer role falls back to the monitor's)."""
    return SUMMARY if "Summarize it faithfully" in prompt else "General fix.\nSCORE: 15"


def gpt_oss_log(tmp_path: Path):
    log = make_log(GPT_OSS, [sample("u1", messages=episode("gpt-oss", (SHORT, LONG))), errored("u2")])
    return log, write_log(tmp_path / "gpt.eval", log)


def test_m3_summary_replaces_long_blocks_and_reuses_its_summaries(tmp_path, ledger):
    log, path = gpt_oss_log(tmp_path)
    value_file = write(tmp_path, target(tokens=50))
    script = Script(respond)
    scored = run_scorer(log, m3_summary(log=str(path), target_file=str(value_file)), {"monitor": script.model()})
    summarizer_prompts = [p for p in script.prompts if "Summarize it faithfully" in p]
    [monitor_prompt] = [p for p in script.prompts if p not in summarizer_prompts]
    assert len(summarizer_prompts) == 1 and LONG in summarizer_prompts[0]
    raw = m3_transcript(log.samples[0])
    assert LONG in raw and LONG not in monitor_prompt
    assert f"<reasoning>\n{SUMMARY}\n</reasoning>" in monitor_prompt and f"<reasoning>\n{SHORT}\n</reasoning>" in monitor_prompt
    assert TICKET in monitor_prompt
    first = scored.samples[0].scores["m3_summary"]
    assert first.value == 15
    meta = first.metadata
    assert [entry["mode"] for entry in meta["summaries"]] == ["passed_through", "summarized"]
    assert meta["summary_stats"]["passed_through_share"] == 0.5
    assert meta["spend"]["by_role"]["summarizer"]["calls"] == 1 and meta["spend"]["by_role"]["monitor"]["calls"] == 1
    assert math.isnan(scored.samples[1].scores["m3_summary"].value)
    assert [e.role for e in spend.ledger_entries()] == ["summarizer", "monitor"]

    # A second run on the scored log reuses every summary: only the monitor is called again.
    write_log(path, scored)
    rescored = run_scorer(read_eval_log(str(path)), m3_summary(log=str(path), target_file=str(value_file)), {"monitor": script.model()})
    assert len(script.prompts) == 3 and script.prompts[2] == monitor_prompt
    second = rescored.samples[0].scores["m3_summary1"]
    assert second.value == 15 and [entry["reused"] for entry in second.metadata["summaries"]] == [True, True]
    assert second.metadata["spend"]["calls"] == 1
    assert [e.role for e in spend.ledger_entries()] == ["summarizer", "monitor", "monitor"]


def test_m3_summary_scores_gpt_oss_logs_only(tmp_path, ledger):
    log = make_log(HAIKU, [sample("u1", messages=episode("claude"))])
    path = write_log(tmp_path / "haiku.eval", log)
    script = Script(respond)
    raised(ValueError, "gpt-oss agent logs only", lambda: run_scorer(log, m3_summary(log=str(path), target_file=str(write(tmp_path, target()))), {"monitor": script.model()}))
    assert script.prompts == []


def test_m3_summary_refuses_without_a_matching_target(tmp_path, ledger):
    log, path = gpt_oss_log(tmp_path)
    script = Script(respond)
    raised(SummaryTargetMissing, "summary-target", lambda: run_scorer(log, m3_summary(log=str(path), target_file=str(tmp_path / "none.json")), {"monitor": script.model()}))
    other = write_target(target(counter=HAIKU), tmp_path / "haiku_target.json")
    raised(ValueError, "counted by anthropic/claude-haiku-5-5", lambda: run_scorer(log, m3_summary(log=str(path), target_file=str(other)), {"monitor": script.model()}))
    assert script.prompts == [] and spend.ledger_entries() == []


def test_the_summarizer_role_is_checked_too(tmp_path, ledger):
    log, path = gpt_oss_log(tmp_path)
    script = Script(respond)
    roles = {"monitor": script.model(), "summarizer": script.model(temperature=0.5)}
    raised(ValueError, "the summarizer role sets temperature", lambda: run_scorer(log, m3_summary(log=str(path), target_file=str(write(tmp_path, target()))), roles))
    assert script.prompts == [] and spend.ledger_entries() == []


def test_a_failed_summary_leaves_the_sample_unscored_without_a_monitor_call(tmp_path, ledger):
    log, path = gpt_oss_log(tmp_path)
    script = Script(lambda prompt: "" if "Summarize it faithfully" in prompt else "SCORE: 1")
    scored = run_scorer(log, m3_summary(log=str(path), target_file=str(write(tmp_path, target()))), {"monitor": script.model()})
    result = scored.samples[0].scores["m3_summary"]
    assert math.isnan(result.value) and result.metadata["status"] == "summary_failed"
    assert all("Summarize it faithfully" in p for p in script.prompts) and len(script.prompts) == 2
    assert [e.role for e in spend.ledger_entries()] == ["summarizer", "summarizer"]


def test_m3_summary_plans_every_uncached_block(tmp_path):
    """The plan is an upper bound: both blocks count as summarizer calls, though the short one will pass through."""
    log, _ = gpt_oss_log(tmp_path)
    names = {"monitor": MODEL, "summarizer": MODEL}
    monitor, summarizer = llm_monitors.plan("m3_summary", log, names, target=target())
    assert (monitor.role, monitor.prompts, summarizer.role, summarizer.prompts) == ("monitor", 1, "summarizer", 2)
    # The monitor prompt's bound allows each block's summary to be as long as a summarizer reply can be.
    raw = spend.input_tokens_bound(llm_monitors.monitor_template().render(TRANSCRIPT=m3_transcript(log.samples[0])))
    assert monitor.input_tokens_bound == raw + 2 * spend.MAX_OUTPUT_TOKENS
    with pytest.raises(ValueError, match="gpt-oss"):
        llm_monitors.plan("m3_summary", make_log(HAIKU, [sample("u1")]), names, target=target())
