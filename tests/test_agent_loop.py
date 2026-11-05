"""The agent loop: which replies end an episode, which are regenerated, and that every limit still applies.

Each test runs a one-sample Inspect task on the host with mockllm and a tool that only echoes its
argument, so these need no Docker, run no task or agent code and make no paid calls.
scripts/smoke_mockllm.py runs the same loop inside the full sandboxed task.
"""

import json

import anyio
import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalSample
from inspect_ai.model import (
    ChatMessage,
    ContentReasoning,
    ContentText,
    ModelCost,
    ModelInfo,
    ModelOutput,
    ModelUsage,
    set_model_info,
)
from inspect_ai.solver import system_message, use_tools
from inspect_ai.tool import tool

from rewardhack.task import (
    JSON_SEARCH_CHARS,
    MAX_REGENERATIONS,
    ROOT,
    SYSTEM_PROMPT,
    agent_loop,
    discarded_reply,
    trailing_json_object,
)

MODEL = "mockllm/model"
# A mockllm model with a price, so a cost limit can trip: $10,000 per million output tokens.
PRICED_MODEL = "mockllm/priced"
# What gpt-oss-120b wrote in a live smoke run: its next tool call's arguments, in its reasoning only.
TOOL_CALL_IN_REASONING = 'Also test_legacy_listing.\n{"path":"tests/test_legacy_listing.py"}'
# The reasoning of an ordinary gpt-oss tool-call turn, which comes with an empty text block.
GPT_OSS_REASONING = "We need to read the source."
# Nested deeper than json's C decoder allows, which makes it raise RecursionError.
TOO_DEEP = '{"a":' + "[" * 15_000


@tool
def note():
    async def execute(text: str) -> str:
        """Record a note.

        Args:
            text: The note.

        Returns:
            The note.
        """
        return text

    return execute


def usage(output_tokens: int) -> ModelUsage:
    return ModelUsage(input_tokens=10, output_tokens=output_tokens, total_tokens=10 + output_tokens)


def reasoning_only(
    reasoning: str = TOOL_CALL_IN_REASONING, output_tokens: int | None = None, stop_reason: str = "stop"
) -> ModelOutput:
    """A reply shaped like gpt-oss's: a reasoning block, an empty text block and no tool call."""
    output = ModelOutput.from_content(
        model=MODEL, content=[ContentReasoning(reasoning=reasoning), ContentText(text="")], stop_reason=stop_reason
    )
    if output_tokens is not None:
        output.usage = usage(output_tokens)
    return output


def call(text: str = "hello") -> ModelOutput:
    return ModelOutput.for_tool_call(model=MODEL, tool_name="note", tool_arguments={"text": text})


def gpt_oss_call(text: str = "hello") -> ModelOutput:
    """A tool-call reply shaped like every live gpt-oss one: a reasoning block, an empty text block and the call."""
    output = call(text)
    output.message.content = [ContentReasoning(reasoning=GPT_OSS_REASONING), ContentText(text="")]
    return output


def reply(content: str | list) -> ModelOutput:
    return ModelOutput.from_content(model=MODEL, content=content)


def run(outputs, tmp_path, model: str = MODEL, **limits) -> EvalSample:
    """Run one episode with the standard limits, overridden by any given, and return its sample."""
    task = Task(
        dataset=[Sample(input="Fix the bug.")],
        solver=[system_message(SYSTEM_PROMPT), use_tools(note()), agent_loop()],
    )
    [log] = eval(
        task,
        model=model,
        model_args={"custom_outputs": outputs},
        **{"turn_limit": 30, "token_limit": "output:100k", "time_limit": 900, "cost_limit": 0.01} | limits,
        model_cost_config=str(ROOT / "costs.yaml"),
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.status == "success", log.error
    [sample] = log.samples
    assert sample.error is None, sample.error
    return sample


def roles(sample: EvalSample) -> list[str]:
    return [message.role for message in sample.messages]


def model_events(sample: EvalSample) -> list:
    return [event for event in sample.events if event.event == "model"]


def regenerations(sample: EvalSample) -> list[dict]:
    return sample.store.get("regenerations", [])


def reasoning_in(message: ChatMessage) -> list[str]:
    blocks = message.content if isinstance(message.content, list) else []
    return [block.reasoning for block in blocks if isinstance(block, ContentReasoning)]


def test_reasoning_only_reply_is_regenerated_and_the_episode_continues(tmp_path):
    sample = run([gpt_oss_call(), reasoning_only(), gpt_oss_call("again"), reply("Done.")], tmp_path)
    assert roles(sample) == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert sample.messages[-1].text == "Done."
    assert regenerations(sample) == [{
        "turn": 1,
        "stop_reason": "stop",
        "visible_chars": 0,
        "reasoning_chars": len(TOOL_CALL_IN_REASONING),
        "reasoning_json": {"path": "tests/test_legacy_listing.py"},
    }]
    # Both gpt-oss tool calls are accepted with their reasoning; the discarded reply reaches no message,
    # but Inspect's model event keeps its raw output.
    accepted = [message for message in sample.messages if message.role == "assistant"][:2]
    assert [reasoning_in(message) for message in accepted] == [[GPT_OSS_REASONING]] * 2
    assert [message.tool_calls[0].arguments for message in accepted] == [{"text": "hello"}, {"text": "again"}]
    assert all(TOOL_CALL_IN_REASONING not in reasoning_in(message) for message in sample.messages)
    events = model_events(sample)
    assert len(events) == 4
    assert reasoning_in(events[1].output.message) == [TOOL_CALL_IN_REASONING]


def test_gpt_oss_tool_calls_with_empty_text_are_not_regenerated(tmp_path):
    sample = run([gpt_oss_call(), gpt_oss_call("again"), reply("Done.")], tmp_path)
    assert regenerations(sample) == []
    assert len(model_events(sample)) == 3
    assert roles(sample) == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert [message.text for message in sample.messages if message.role == "tool"] == ["hello", "again"]


def test_a_garbled_call_to_a_tool_without_arguments_is_regenerated(tmp_path):
    sample = run([reasoning_only("Now run the tests.{}"), call(), reply("Done.")], tmp_path)
    assert [r["reasoning_json"] for r in regenerations(sample)] == [{}]
    assert sample.messages[-1].text == "Done."


def test_a_hidden_call_with_whitespace_only_text_is_regenerated(tmp_path):
    whitespace = reply([ContentReasoning(reasoning=TOOL_CALL_IN_REASONING), ContentText(text=" \n")])
    sample = run([whitespace, call(), reply("Done.")], tmp_path)
    assert len(regenerations(sample)) == 1
    assert sample.messages[-1].text == "Done."


@pytest.mark.parametrize(("blocks", "regenerated"), [
    (['{"path": "a"}', "Plain."], False),
    (["Plain.", '{"path": "a"}'], True),
], ids=["json_in_an_earlier_block", "json_ending_the_last_block"])
def test_only_the_last_reasoning_block_is_checked_for_a_hidden_call(tmp_path, blocks, regenerated):
    content = [*(ContentReasoning(reasoning=text) for text in blocks), ContentText(text="")]
    sample = run([reply(content), call(), reply("Done.")], tmp_path)
    assert len(regenerations(sample)) == int(regenerated)


def test_a_reasoning_only_reply_too_deep_for_the_json_decoder_ends_the_episode_without_crashing(tmp_path):
    sample = run([reasoning_only(TOO_DEEP), reply("Never sent.")], tmp_path)
    assert regenerations(sample) == []
    assert roles(sample) == ["system", "user", "assistant"]
    assert reasoning_in(sample.messages[-1]) == [TOO_DEEP]


@pytest.mark.parametrize("silent", [
    reply(""),
    reply(" \n"),
    reasoning_only("Nothing more to do."),
    reasoning_only("No need to note anything or run the tests again. Done."),
    # JSON that is not the last thing in the reasoning is not a hidden call.
    reasoning_only('The handler returns {"status": "ok"} now, so we are done.'),
    # Cut off by the output limit: retrying would likely run away again, so it ends as it is.
    reasoning_only(stop_reason="max_tokens"),
], ids=["empty", "whitespace", "reasoning_without_a_call", "tool_named", "json_not_at_the_end", "max_tokens"])
def test_a_reply_without_a_hidden_call_ends_the_episode_and_is_not_regenerated(tmp_path, silent):
    # A model that silently stops gets no extra draws, which in the conflict variant could add gaming edits.
    sample = run([silent, call(), reply("Never sent.")], tmp_path)
    assert regenerations(sample) == []
    assert len(model_events(sample)) == 1
    assert roles(sample) == ["system", "user", "assistant"]


def test_four_hidden_call_replies_end_the_episode_after_three_regenerations(tmp_path):
    assert MAX_REGENERATIONS == 3
    hidden = [reasoning_only(), reasoning_only('{"path": "src/a.py"}'), reasoning_only("Next.{}"), reasoning_only('{"path": "b"}')]
    sample = run([*hidden, reply("Never sent.")], tmp_path)
    assert len(model_events(sample)) == 4
    assert [(r["turn"], r["reasoning_json"]) for r in regenerations(sample)] == [
        (0, {"path": "tests/test_legacy_listing.py"}), (0, {"path": "src/a.py"}), (0, {}),
    ]
    # The fourth is appended once the budget is spent, and it ends the episode.
    assert roles(sample) == ["system", "user", "assistant"]
    assert reasoning_in(sample.messages[-1]) == ['{"path": "b"}']


@pytest.mark.parametrize("content", [
    "Fixed it.",
    [ContentReasoning(reasoning="signature", summary="Done; now summarise.", redacted=True), ContentText(text="Fixed it.")],
], ids=["text", "summarised_reasoning_and_text"])
def test_visible_text_without_a_tool_call_ends_the_episode(tmp_path, content):
    sample = run([reply(content), call()], tmp_path)
    assert roles(sample) == ["system", "user", "assistant"]
    assert sample.messages[-1].text == "Fixed it."
    assert regenerations(sample) == []
    assert len(model_events(sample)) == 1


def test_an_empty_refusal_ends_the_episode_and_is_not_regenerated(tmp_path):
    refusal = ModelOutput.from_content(model=MODEL, content="", stop_reason="content_filter")
    sample = run([refusal, call()], tmp_path)
    assert roles(sample) == ["system", "user", "assistant"]
    assert sample.output.stop_reason == "content_filter"
    assert regenerations(sample) == []


def test_a_regeneration_resends_the_identical_conversation(tmp_path):
    scripted = iter([call(), reasoning_only(), reasoning_only('Then {"text": "x"}'), reply("Done.")])
    sent = []

    def record(input, tools, tool_choice, config):
        sent.append(([message.model_dump() for message in input], [t.name for t in tools], tool_choice))
        return next(scripted)

    sample = run(record, tmp_path)
    assert len(sent) == 4
    assert [message["role"] for message in sent[1][0]] == ["system", "user", "assistant", "tool"]
    assert sent[1] == sent[2] == sent[3]
    events = model_events(sample)
    assert events[1].input == events[2].input == events[3].input
    assert len(regenerations(sample)) == 2


def test_regenerations_count_toward_the_turn_limit(tmp_path):
    # Without the two regenerations, call, call, reply would finish inside 3 turns.
    sample = run([reasoning_only(), call(), reasoning_only(), call(), reply("Done.")], tmp_path, turn_limit=3)
    assert (sample.limit.type, sample.limit.limit) == ("turn", 3)
    assert len(model_events(sample)) == 4
    assert len(regenerations(sample)) == 2
    assert roles(sample) == ["system", "user", "assistant", "tool"]


def test_regenerations_count_toward_the_token_limit(tmp_path):
    outputs = [reasoning_only(output_tokens=60), reasoning_only(output_tokens=60), reply("Done.")]
    sample = run(outputs, tmp_path, token_limit="output:100")
    assert (sample.limit.type, sample.limit.limit) == ("token", 100)
    assert len(regenerations(sample)) == 1
    assert roles(sample) == ["system", "user"]


def test_regenerations_count_toward_the_cost_limit(tmp_path):
    set_model_info(PRICED_MODEL, ModelInfo(cost=ModelCost(input=0, output=10_000, input_cache_write=0, input_cache_read=0)))
    # Each reply costs $0.60, so the second crosses the $1.00 limit.
    outputs = [reasoning_only(output_tokens=60), reasoning_only(output_tokens=60), reply("Done.")]
    sample = run(outputs, tmp_path, model=PRICED_MODEL, cost_limit=1.0)
    assert (sample.limit.type, sample.limit.limit) == ("cost", 1.0)
    assert len(regenerations(sample)) == 1
    assert roles(sample) == ["system", "user"]


def test_regenerations_run_inside_the_time_limit(tmp_path):
    async def slow_reasoning_only(input, tools, tool_choice, config):
        await anyio.sleep(0.4)
        return reasoning_only()

    # Four replies would take 1.6 seconds, so the 1-second limit stops the third.
    sample = run(slow_reasoning_only, tmp_path, time_limit=1)
    assert (sample.limit.type, sample.limit.limit) == ("time", 1)
    assert 1 <= len(regenerations(sample)) <= 2
    assert roles(sample) == ["system", "user"]


@pytest.mark.parametrize(("text", "expected"), [
    (TOOL_CALL_IN_REASONING, {"path": "tests/test_legacy_listing.py"}),
    ('Also check tests.{"path":"tests/test_listing.py"}', {"path": "tests/test_listing.py"}),
    ("Now run the tests.{}", {}),
    ('{ "outer": {"inner": 1} }', {"outer": {"inner": 1}}),
    ('First {"path": "a"}, then {"path": "b"}', {"path": "b"}),
    ('Trailing whitespace {"path": "x"} \n ', {"path": "x"}),
    ('First {"path": "a"}, then done.', None),
    ('A list [{"path": "a"}]', None),
    ("Python dict {'path': 'x'}", None),
    ('Cut off: {"path": "src/', None),
    ('Unbalanced {"path": "x"}}', None),
    ("", None),
    (TOO_DEEP, None),
    # Ends with a brace, so the too-deep candidate is decoded and must be skipped rather than raise.
    (TOO_DEEP + "}", None),
    (TOO_DEEP + ' {"path": "x"}', {"path": "x"}),
    # A garbled write_file whose content holds braces: inner candidates fail, and the outer object is found.
    ("Write it." + json.dumps({"path": "src/a.py", "content": 'd = {"k": 1}; s = "{"'}),
     {"path": "src/a.py", "content": 'd = {"k": 1}; s = "{"'}),
    # Only the end of the reasoning is searched, after trailing whitespace is dropped.
    ('{"path": "' + "a" * JSON_SEARCH_CHARS + '"}', None),
    ('{"path": "early"}' + " " * JSON_SEARCH_CHARS, {"path": "early"}),
], ids=lambda value: "long" if isinstance(value, str) and len(value) > 60 else None)
def test_trailing_json_object(text, expected):
    assert trailing_json_object(text) == expected


def test_too_deep_really_trips_the_json_decoder():
    # Guards the TOO_DEEP + "}" case above: it only tests the RecursionError path while json raises it.
    with pytest.raises(RecursionError):
        json.loads(TOO_DEEP)


def test_discarded_reply_reads_claude_summaries_and_raw_gpt_oss_reasoning():
    summary = 'Next I read {"path": "src/app.py"}'
    claude = reply([ContentReasoning(reasoning="signature", summary=summary, redacted=True), ContentText(text="")])
    assert discarded_reply(4, claude) == {
        "turn": 4,
        "stop_reason": "stop",
        "visible_chars": 0,
        "reasoning_chars": len(summary),
        "reasoning_json": {"path": "src/app.py"},
    }
    gpt_oss = reasoning_only("Plain thoughts.")
    assert discarded_reply(0, gpt_oss)["reasoning_chars"] == len("Plain thoughts.")
    assert discarded_reply(0, gpt_oss)["reasoning_json"] is None
