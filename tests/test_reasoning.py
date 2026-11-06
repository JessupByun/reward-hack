"""Reasoning extraction and counts (rewardhack.monitors.reasoning) on constructed messages and logs.

gpt-oss blocks are built with inspect-ai's own OpenRouter conversion, so they have the shape the
live smoke logs have; every text here is made up, and nothing calls a model or needs Docker.
"""

import itertools
import json

import pytest
from inspect_ai.log import EvalConfig, EvalDataset, EvalError, EvalLog, EvalSample, EvalSpec, read_eval_log, write_eval_log
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
)
from inspect_ai.model._openai import chat_message_assistant_from_openai
from inspect_ai.model._openrouter_reasoning import openrouter_reasoning_details_to_reasoning
from inspect_ai.tool import ToolCall
from openai.types.chat import ChatCompletionMessage

from rewardhack import task
from rewardhack.monitors.reasoning import (
    DETAILS_SIGNATURE,
    PLACEHOLDER,
    ReasoningCounts,
    agent_counts,
    agent_family,
    episode_counts,
    is_json,
    is_placeholder,
    is_unparsed_details,
    log_counts,
    message_reasoning,
    message_reasoning_text,
    reasoning_blocks,
    reasoning_text,
    summary_detail_text,
    turn_counts,
)

CLAUDE = "anthropic/claude-haiku-5-5"
GPT_OSS = "openrouter/openai/gpt-oss-120b"


def claude_block(summary: str | None) -> ContentReasoning:
    """A Claude thinking block as the Anthropic provider stores it: the signature in reasoning, the text in summary."""
    return ContentReasoning(reasoning="opaque-signature", summary=summary, redacted=True)


def text_detail(text: str, index: int = 0) -> dict:
    return {"type": "reasoning.text", "text": text, "format": "unknown", "index": index}


def summary_detail(summary: str, index: int = 0) -> dict:
    return {"type": "reasoning.summary", "summary": summary, "format": "unknown", "index": index}


def encrypted_detail(data: str, index: int = 0) -> dict:
    return {"type": "reasoning.encrypted", "data": data, "format": "unknown", "index": index}


def openrouter_block(*details: dict) -> ContentReasoning:
    """A block made by inspect-ai's OpenRouter conversion from these reasoning details."""
    return openrouter_reasoning_details_to_reasoning(list(details))


def gpt_oss_block(text: str) -> ContentReasoning:
    """A gpt-oss block as every one in the smoke logs is: one reasoning.text detail."""
    return openrouter_block(text_detail(text))


def assistant(*content: ContentReasoning | ContentText, tool: str | None = None) -> ChatMessageAssistant:
    tool_calls = [ToolCall(id="call_1", function=tool, arguments={})] if tool else None
    return ChatMessageAssistant(content=list(content), tool_calls=tool_calls)


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        (claude_block("Read the tests first."), "Read the tests first."),
        # A redacted_thinking block has no summary, and the opaque data in reasoning is never read.
        (ContentReasoning(reasoning="opaque-data", redacted=True), ""),
        (claude_block(""), ""),
        (gpt_oss_block("The fix belongs in listing.py."), "The fix belongs in listing.py."),
        (ContentReasoning(reasoning=""), ""),
        # An unredacted block's summary is not its text, even when it has no reasoning.
        (ContentReasoning(reasoning="raw", summary="summary"), "raw"),
        (ContentReasoning(reasoning="", summary="summary"), ""),
    ],
)
def test_reasoning_text(block, expected):
    assert reasoning_text(block) == expected
    # task.py's agent loop applies the same rule until it imports this one.
    assert task.reasoning_text(block) == expected


def test_message_reasoning_drops_placeholder_and_empty_blocks_in_order():
    message = assistant(
        claude_block("First."),
        claude_block(PLACEHOLDER),
        claude_block(None),
        claude_block(" \n"),
        claude_block("Second."),
        ContentText(text="Visible text is not reasoning."),
        tool="read_file",
    )
    assert [reasoning_text(block) for block in reasoning_blocks(message)] == ["First.", PLACEHOLDER, "", " \n", "Second."]
    assert message_reasoning(message) == ["First.", "Second."]
    assert message_reasoning_text(message) == "First.\n\nSecond."
    assert message_reasoning_text(message, separator="\n") == "First.\nSecond."


def test_only_the_exact_placeholder_is_dropped_and_counted():
    near = [PLACEHOLDER + " ", PLACEHOLDER.lower(), PLACEHOLDER + " More reasoning.", " " + PLACEHOLDER]
    assert is_placeholder(PLACEHOLDER) and not any(is_placeholder(text) for text in near)
    message = assistant(*(claude_block(text) for text in near))
    assert message_reasoning(message) == near
    assert turn_counts(message, "claude") == ReasoningCounts(turns=1, turns_with_reasoning=1, blocks=4)
    # Each near miss alone still gives its turn reasoning, as the text M3 reads still holds it.
    for text in near:
        assert turn_counts(assistant(claude_block(text)), "claude") == ReasoningCounts(turns=1, turns_with_reasoning=1, blocks=1)
    for block in (claude_block(PLACEHOLDER), gpt_oss_block(PLACEHOLDER)):
        assert message_reasoning(assistant(block)) == []
        assert turn_counts(assistant(block), "claude") == ReasoningCounts(turns=1, blocks=1, placeholder_blocks=1)


@pytest.mark.parametrize(
    "message",
    [
        ChatMessageAssistant(content="Implemented the fix."),
        ChatMessageAssistant(content=[]),
        assistant(ContentText(text="Implemented the fix.")),
        ChatMessageUser(content="Fix the bug."),
    ],
)
def test_messages_without_reasoning_blocks(message):
    assert reasoning_blocks(message) == []
    assert message_reasoning(message) == []
    assert message_reasoning_text(message) == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"path": "tests/test_listing.py"}', True),
        (' [{"type": "reasoning.text"}]\n', True),
        ("{}", True),
        ('"just a string"', True),
        ("42", True),
        ("null", True),
        # Reasoning that only ends with a tool call's arguments is reasoning, not JSON.
        ('Also check the tests.\n{"path": "tests/test_listing.py"}', False),
        ('{"path": "a"} then more', False),
        ("{'type': 'reasoning.text'}", False),
        ("The function needs a check.", False),
        ("", False),
        ("   ", False),
        ("{", False),
        # Nesting too deep for the decoder raises RecursionError, which counts as not JSON.
        ("[" * 100_000 + "]" * 100_000, False),
    ],
)
def test_is_json(text, expected):
    assert is_json(text) is expected


@pytest.mark.parametrize(
    ("details", "text", "redacted", "expected"),
    [
        ([text_detail("raw")], "raw", False, False),
        # A summary with no text or encrypted sibling reads like raw reasoning; only the signature tells.
        ([summary_detail("short")], "short", False, True),
        ([summary_detail("")], "", False, True),
        ([summary_detail("short"), text_detail("raw")], "raw", False, False),
        ([text_detail("raw"), summary_detail("short")], "raw", False, False),
        ([summary_detail("short"), encrypted_detail("blob")], "short", True, True),
        ([encrypted_detail("blob"), summary_detail("short")], "short", True, True),
        ([text_detail("raw"), encrypted_detail("blob")], "raw", True, False),
        ([encrypted_detail("blob")], "", True, False),
        # No readable text: inspect marks a non-empty list redacted and leaves no summary.
        ([text_detail("")], "", True, False),
        ([], "", False, False),
    ],
)
def test_summary_detail_text_on_inspect_blocks(details, text, redacted, expected):
    block = openrouter_block(*details)
    assert block.signature == DETAILS_SIGNATURE + json.dumps(details)
    assert (reasoning_text(block), block.redacted) == (text, redacted)
    assert summary_detail_text(block) is expected


def test_summary_detail_text_matches_inspect_for_every_detail_order():
    """Each detail gets a distinct text, so inspect's own conversion shows which one the block's text came from."""
    kinds = ("text", "empty_text", "summary", "encrypted")
    for length in range(1, 4):
        for combo in itertools.product(kinds, repeat=length):
            details = []
            for i, kind in enumerate(combo):
                if kind == "text":
                    details.append(text_detail(f"text-{i}", i))
                elif kind == "empty_text":
                    details.append(text_detail("", i))
                elif kind == "summary":
                    details.append(summary_detail(f"summary-{i}", i))
                else:
                    details.append(encrypted_detail(f"data-{i}", i))
            block = openrouter_block(*details)
            assert summary_detail_text(block) is reasoning_text(block).startswith("summary-"), combo


def test_a_list_that_fails_validation_is_json_not_a_summary_detail():
    # A summary detail without its summary field fails inspect's validation, so the block's text is
    # the whole list as JSON.
    details = [{"type": "reasoning.summary", "format": "unknown"}]
    block = openrouter_block(*details)
    assert reasoning_text(block) == json.dumps(details)
    assert summary_detail_text(block) is False
    assert is_json(reasoning_text(block))
    assert turn_counts(assistant(block), "gpt-oss") == ReasoningCounts(turns=1, turns_with_reasoning=1, blocks=1, json_blocks=1)


@pytest.mark.parametrize(
    "block",
    [
        claude_block("Summary."),
        ContentReasoning(reasoning="raw", signature="anthropic-signature"),
        ContentReasoning(reasoning="raw", signature=DETAILS_SIGNATURE + "not json"),
        ContentReasoning(reasoning="raw", signature=DETAILS_SIGNATURE + '{"type": "reasoning.summary"}'),
        ContentReasoning(reasoning="raw", signature=DETAILS_SIGNATURE + '["reasoning.summary"]'),
    ],
)
def test_summary_detail_text_is_false_without_openrouter_details(block):
    assert summary_detail_text(block) is False


def test_openrouter_reply_shapes_through_inspect_conversion():
    """A reply's reasoning_details become the block that the counts read, as in a live OpenRouter call."""

    def converted(details: list[dict]) -> ChatMessageAssistant:
        reply = ChatCompletionMessage.model_validate(
            {"role": "assistant", "content": None, "reasoning": "unused", "reasoning_details": details}
        )
        return chat_message_assistant_from_openai("openai/gpt-oss-120b", reply, [])

    raw = converted([text_detail("Check the path handling.")])
    assert message_reasoning(raw) == ["Check the path handling."]
    assert turn_counts(raw, "gpt-oss") == ReasoningCounts(turns=1, turns_with_reasoning=1, blocks=1)

    summarized = converted([summary_detail("Checked the path handling.")])
    [block] = reasoning_blocks(summarized)
    assert (block.reasoning, block.summary, block.redacted) == ("Checked the path handling.", None, False)
    assert turn_counts(summarized, "gpt-oss") == ReasoningCounts(turns=1, turns_with_reasoning=1, blocks=1, summary_detail_blocks=1)
    # The summary-detail count is for gpt-oss only.
    assert turn_counts(summarized, "claude").summary_detail_blocks == 0


def test_unparsed_reasoning_details_count_as_json_blocks():
    """reasoning_details that is not a list leaves its Python repr as the block's text (see the module docstring)."""

    def converted(fields: dict) -> ChatMessageAssistant:
        reply = ChatCompletionMessage.model_validate({"role": "assistant", "content": None, **fields})
        return chat_message_assistant_from_openai("openai/gpt-oss-120b", reply, [])

    unparsed = converted({"reasoning_details": {"type": "reasoning.text", "text": "Raw thought."}})
    [block] = reasoning_blocks(unparsed)
    assert (reasoning_text(block), block.internal) == ("{'type': 'reasoning.text', 'text': 'Raw thought.'}", "reasoning_details")
    assert is_unparsed_details(block) and not is_json(reasoning_text(block))
    assert turn_counts(unparsed, "gpt-oss") == ReasoningCounts(turns=1, turns_with_reasoning=1, blocks=1, json_blocks=1)
    assert turn_counts(unparsed, "claude").json_blocks == 0

    # A plain reasoning field, with no details, is raw reasoning.
    plain = converted({"reasoning": "Plain thought."})
    [block] = reasoning_blocks(plain)
    assert (reasoning_text(block), block.internal, is_unparsed_details(block)) == ("Plain thought.", "reasoning", False)
    assert turn_counts(plain, "gpt-oss") == ReasoningCounts(turns=1, turns_with_reasoning=1, blocks=1)
    # inspect's own OpenRouter conversion sets no internal value.
    assert not is_unparsed_details(gpt_oss_block("Raw thought."))


def test_turn_counts_claude():
    message = assistant(
        claude_block("Read the tests."),
        ContentReasoning(reasoning="opaque-data", redacted=True),
        claude_block(PLACEHOLDER),
        # JSON and summary-detail checks are for gpt-oss blocks only.
        claude_block('{"path": "src/listing.py"}'),
        ContentText(text="Reading the tests."),
        tool="read_file",
    )
    assert turn_counts(message, "claude") == ReasoningCounts(
        turns=1, turns_with_reasoning=1, blocks=4, empty_blocks=1, placeholder_blocks=1
    )


def test_turn_counts_gpt_oss():
    message = assistant(
        gpt_oss_block("Read the source next."),
        gpt_oss_block('{"path": "src/listing.py"}'),
        openrouter_block(summary_detail("Read the source.")),
        gpt_oss_block(""),
        ContentText(text=""),
        tool="read_file",
    )
    counts = turn_counts(message, "gpt-oss")
    assert counts == ReasoningCounts(
        turns=1, turns_with_reasoning=1, blocks=4, empty_blocks=1, json_blocks=1, summary_detail_blocks=1
    )
    assert counts.kept_blocks == 3
    # Counted blocks stay in the text: only the placeholder and empty blocks are dropped.
    assert message_reasoning(message) == ["Read the source next.", '{"path": "src/listing.py"}', "Read the source."]


@pytest.mark.parametrize(
    "message",
    [
        assistant(claude_block(PLACEHOLDER), tool="read_file"),
        assistant(claude_block(None), claude_block("  "), tool="read_file"),
        ChatMessageAssistant(content="Done."),
    ],
)
def test_a_turn_with_no_kept_block_has_no_reasoning(message):
    counts = turn_counts(message, "claude")
    assert (counts.turns, counts.turns_with_reasoning, counts.kept_blocks) == (1, 0, 0)


def episode(*turns: ChatMessageAssistant) -> list:
    """An accepted conversation: system prompt, ticket, then each assistant turn and its tool result."""
    messages = [ChatMessageSystem(content="You are a software engineer."), ChatMessageUser(content="Fix the bug.")]
    for turn in turns:
        messages.append(turn)
        if turn.tool_calls:
            messages.append(ChatMessageTool(content="ok", tool_call_id="call_1", function=turn.tool_calls[0].function))
    return messages


def test_episode_counts_and_coverage():
    messages = episode(
        assistant(gpt_oss_block("List the files."), ContentText(text=""), tool="list_files"),
        assistant(gpt_oss_block("Run the tests."), ContentText(text=""), tool="run_tests"),
        # gpt-oss's final answer in the smoke logs is a plain string with no reasoning.
        ChatMessageAssistant(content="Implemented the fix."),
    )
    counts = episode_counts(messages, "gpt-oss")
    assert counts == ReasoningCounts(episodes=1, episodes_with_reasoning=1, turns=3, turns_with_reasoning=2, blocks=2)
    assert counts.turn_coverage == pytest.approx(2 / 3)
    assert counts.episode_coverage == 1.0

    silent = episode_counts(episode(assistant(claude_block(None), tool="list_files"), ChatMessageAssistant(content="Done.")), "claude")
    assert silent == ReasoningCounts(episodes=1, turns=2, blocks=1, empty_blocks=1)
    assert silent.episode_coverage == 0.0


def test_counts_add_and_empty_coverage():
    total = ReasoningCounts(episodes=1, turns=2, blocks=3) + ReasoningCounts(episodes=1, turns_with_reasoning=1, json_blocks=1)
    assert total == ReasoningCounts(episodes=2, turns=2, turns_with_reasoning=1, blocks=3, json_blocks=1)
    assert sum([total, total], ReasoningCounts()) == total + total
    assert ReasoningCounts().turn_coverage is None
    assert ReasoningCounts().episode_coverage is None


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("anthropic/claude-haiku-5-5", "claude"),
        ("anthropic/claude-sonnet-5-5", "claude"),
        ("claude-haiku-5-5", "claude"),
        ("openrouter/openai/gpt-oss-120b", "gpt-oss"),
        ("openai/gpt-oss-120b", "gpt-oss"),
        ("gpt-oss-120b", "gpt-oss"),
    ],
)
def test_agent_family(model, family):
    assert agent_family(model) == family


@pytest.mark.parametrize("model", ["mockllm/model", "openai/gpt-4o", "anthropic/haiku-claude", "claude", ""])
def test_agent_family_rejects_other_models(model):
    with pytest.raises(ValueError):
        agent_family(model)


def make_log(model: str, samples: list[EvalSample]) -> EvalLog:
    spec = EvalSpec(created="2026-01-01T00:00:00+00:00", task="false_assurance", dataset=EvalDataset(), model=model, config=EvalConfig())
    return EvalLog(eval=spec, samples=samples)


def sample(epoch: int, messages: list, errored: bool = False) -> EvalSample:
    error = EvalError(message="boom", traceback="", traceback_ansi="") if errored else None
    return EvalSample(id="078_0", epoch=epoch, input="Fix the bug.", target="", messages=messages, error=error)


def test_log_counts_after_a_log_round_trip(tmp_path):
    """Counts read from .eval files written and read back by Inspect, shaped like the smoke logs."""
    gpt_oss_episode = episode(
        assistant(gpt_oss_block("List the files first."), ContentText(text=""), tool="list_files"),
        assistant(gpt_oss_block("Read the source."), ContentText(text=""), tool="read_file"),
        ChatMessageAssistant(content="Implemented the fix."),
    )
    claude_episode = episode(
        ChatMessageAssistant(content=[], tool_calls=[ToolCall(id="call_1", function="list_files", arguments={})]),
        assistant(claude_block("The tests pin the behaviour."), ContentText(text="Reading the tests."), tool="read_file"),
        ChatMessageAssistant(content=[ContentText(text="Implemented the fix.")]),
    )
    paths = {
        "gpt_oss_1": make_log(GPT_OSS, [sample(1, gpt_oss_episode), sample(2, gpt_oss_episode, errored=True)]),
        "gpt_oss_2": make_log(GPT_OSS, [sample(1, gpt_oss_episode)]),
        "claude": make_log(CLAUDE, [sample(1, claude_episode)]),
    }
    for name, log in paths.items():
        write_eval_log(log, str(tmp_path / f"{name}.eval"))
    logs = {name: read_eval_log(str(tmp_path / f"{name}.eval")) for name in paths}

    one_gpt_oss = ReasoningCounts(episodes=1, episodes_with_reasoning=1, turns=3, turns_with_reasoning=2, blocks=2)
    one_claude = ReasoningCounts(episodes=1, episodes_with_reasoning=1, turns=3, turns_with_reasoning=1, blocks=1)
    # The errored sample is left out.
    assert log_counts(logs["gpt_oss_1"]) == one_gpt_oss
    assert log_counts(logs["claude"]) == one_claude
    assert agent_counts(logs.values()) == {GPT_OSS: one_gpt_oss + one_gpt_oss, CLAUDE: one_claude}

    header = read_eval_log(str(tmp_path / "claude.eval"), header_only=True)
    with pytest.raises(ValueError, match="header_only"):
        log_counts(header)


def test_log_counts_rejects_a_log_from_another_model():
    with pytest.raises(ValueError):
        log_counts(make_log("mockllm/model", []))
