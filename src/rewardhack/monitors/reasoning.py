"""Reasoning extraction shared by M3, the summarizer, the judge and the pilot report (PLAN.md, "Monitors").

Each ContentReasoning block's text is the provider summary for a redacted block and the raw
reasoning otherwise, and blocks whose text is exactly PLACEHOLDER are dropped.
Only accepted assistant messages are read; regenerated replies stay in Inspect's model events.

How inspect-ai 0.3.282 stores each agent's reasoning (line numbers are in .venv/lib/python3.12/site-packages/inspect_ai/):
- Claude, through the Anthropic provider (model/_providers/anthropic.py): a thinking block becomes
  ContentReasoning(summary=thinking, reasoning=signature, redacted=True) (lines 5031-5039), so the
  text is the summary; a redacted_thinking block has no summary (lines 5048-5052), so its text is
  empty, and the hidden reasoning is never recovered.
- gpt-oss, through OpenRouter: parse_reasoning_content (model/_openai.py lines 1129-1148) takes the
  reply's reasoning_details before its plain reasoning field, and the provider's extractor
  (model/_providers/openrouter.py lines 202-216) passes the list to
  openrouter_reasoning_details_to_reasoning (model/_openrouter_reasoning.py lines 55-103), which keeps
  the whole list as JSON in the block's signature after DETAILS_SIGNATURE (lines 58-59).
  A reasoning.text detail becomes the block's reasoning with redacted=False (lines 77-81), which is
  every block in the smoke logs.
  A reasoning.summary detail with no text or encrypted sibling also lands in reasoning with
  redacted=False (lines 75-76 and 88-91), so it reads like raw reasoning; summary_detail_text finds
  that case from the signature.
  A list that fails validation becomes ContentReasoning(reasoning=<the list as JSON>) (lines 61-68),
  which is_json finds.
  reasoning_details that is not a list makes the provider's extractor return None (openrouter.py
  lines 205-214), and the fallback (model/_openai.py lines 1104-1107 and 1068-1081) stores its Python
  repr as ContentReasoning(reasoning=str(details), internal="reasoning_details"); is_unparsed_details
  finds that block, and the counts add it to the JSON blocks, since it is raw details, not reasoning.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from typing import Literal

from inspect_ai.log import EvalLog, EvalSample
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ContentReasoning

# Anthropic's text for a summarized thinking block that was cut off; it carries no reasoning.
PLACEHOLDER = "This part of the response was interrupted before it finished."
# inspect_ai/model/_openrouter_reasoning.py line 11: the prefix of an OpenRouter block's signature.
DETAILS_SIGNATURE = "reasoning-details://"

Family = Literal["claude", "gpt-oss"]


def agent_family(model: str) -> Family:
    """claude or gpt-oss from an agent model name, with or without its provider prefix.

    That covers log names such as anthropic/claude-haiku-5-5 and openrouter/openai/gpt-oss-120b and
    message names such as claude-haiku-5-5 and openai/gpt-oss-120b; any other model raises ValueError.
    """
    name = model.rsplit("/", 1)[-1]
    if name.startswith("claude-"):
        return "claude"
    if name.startswith("gpt-oss-"):
        return "gpt-oss"
    raise ValueError(f"{model!r} is neither a claude nor a gpt-oss agent")


def reasoning_text(block: ContentReasoning) -> str:
    """A reasoning block's text: the provider summary for a redacted (Claude) block, else the raw reasoning."""
    return (block.summary if block.redacted else block.reasoning) or ""


def is_empty(text: str) -> bool:
    """Whether a block's text has nothing but whitespace."""
    return not text.strip()


def is_placeholder(text: str) -> bool:
    """Whether a block's text is exactly PLACEHOLDER; message_reasoning drops it and block_counts counts it."""
    return text == PLACEHOLDER


def is_json(text: str) -> bool:
    """Whether the whole text, without surrounding whitespace, decodes as one JSON value.

    That is what a failed OpenRouter conversion leaves (see the module docstring).
    Text that only ends with JSON, as when gpt-oss writes a tool call's arguments at the end of its
    reasoning (task.trailing_json_object), is reasoning and does not count.
    Decoding uses the json module, so a bare number, string, true, false, null, NaN or Infinity counts
    too; none of those is reasoning either.
    """
    try:
        json.loads(text.strip())
    except (ValueError, RecursionError):
        return False
    return True


def is_unparsed_details(block: ContentReasoning) -> bool:
    """Whether the block is inspect's fallback for an OpenRouter reasoning_details that was not a list.

    Its text is the Python repr of that value (see the module docstring), and a block that OpenRouter's
    own conversion made never has this internal value, since that conversion sets none.
    """
    return block.internal == "reasoning_details"


def summary_detail_text(block: ContentReasoning) -> bool:
    """Whether the block's text came from an OpenRouter reasoning.summary detail.

    The raw detail list is in the block's signature after DETAILS_SIGNATURE, and this replays
    openrouter_reasoning_details_to_reasoning (inspect_ai/model/_openrouter_reasoning.py lines 70-99)
    over it, tracking which detail type supplied the reasoning and the summary.
    A block whose text is the signature's list itself (a list that failed validation, lines 61-68) did
    not come from one detail, so it is False here and is_json counts it.
    Blocks without that signature, such as Claude's from the Anthropic provider, are False.
    """
    signature = block.signature or ""
    if not signature.startswith(DETAILS_SIGNATURE):
        return False
    payload = signature.removeprefix(DETAILS_SIGNATURE)
    if reasoning_text(block) == payload:
        return False
    try:
        details = json.loads(payload)
    except (ValueError, RecursionError):
        return False
    if not isinstance(details, list) or not all(isinstance(detail, dict) for detail in details):
        return False
    # The detail type that set inspect's reasoning and summary variables, in the same order of updates.
    reasoning_from: str | None = None
    summary_from: str | None = None
    redacted = False
    for detail in details:
        kind = detail.get("type")
        if kind == "reasoning.summary":
            summary_from = kind
        elif kind == "reasoning.text":
            if detail.get("text"):
                reasoning_from = kind
        elif kind == "reasoning.encrypted":
            if reasoning_from is not None:
                summary_from = reasoning_from
            reasoning_from, redacted = kind, True
    if reasoning_from is None:
        reasoning_from, summary_from = summary_from, None
    # reasoning_text reads the summary of a redacted block and the reasoning of any other.
    return (summary_from if redacted else reasoning_from) == "reasoning.summary"


def reasoning_blocks(message: ChatMessage) -> list[ContentReasoning]:
    """Every ContentReasoning in a message, in order; a message with string content has none."""
    content = message.content if isinstance(message.content, list) else []
    return [block for block in content if isinstance(block, ContentReasoning)]


def message_reasoning(message: ChatMessage) -> list[str]:
    """The texts of a message's reasoning blocks, in order, without placeholder or empty blocks.

    The summarizer takes these one block at a time.
    """
    texts = [reasoning_text(block) for block in reasoning_blocks(message)]
    return [text for text in texts if not is_placeholder(text) and not is_empty(text)]


def message_reasoning_text(message: ChatMessage, separator: str = "\n\n") -> str:
    """A message's reasoning as one string (see message_reasoning), or "" when it has none.

    M3 inserts this before each turn's visible text and tool calls.
    """
    return separator.join(message_reasoning(message))


@dataclass(frozen=True)
class ReasoningCounts:
    """Reasoning counts for a block, a turn, an episode, a log or an agent; counts add with +.

    Turns are accepted assistant messages, and a turn or episode has reasoning when at least one of
    its blocks is kept (neither empty nor the placeholder).
    Every block counts once in blocks, and empty_blocks and placeholder_blocks never overlap, because
    the placeholder is not empty.
    json_blocks and summary_detail_blocks count gpt-oss blocks only (they stay 0 for Claude), whether
    or not the block is also empty or the placeholder; the smoke test and pilot require both to be 0.
    json_blocks also counts the unparsed-details fallback (is_unparsed_details), whose text is not JSON.
    """

    episodes: int = 0
    episodes_with_reasoning: int = 0
    turns: int = 0
    turns_with_reasoning: int = 0
    blocks: int = 0
    empty_blocks: int = 0
    placeholder_blocks: int = 0
    json_blocks: int = 0
    summary_detail_blocks: int = 0

    def __add__(self, other: "ReasoningCounts") -> "ReasoningCounts":
        return ReasoningCounts(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    @property
    def kept_blocks(self) -> int:
        return self.blocks - self.empty_blocks - self.placeholder_blocks

    @property
    def turn_coverage(self) -> float | None:
        """The share of turns with reasoning, or None with no turns."""
        return self.turns_with_reasoning / self.turns if self.turns else None

    @property
    def episode_coverage(self) -> float | None:
        """The share of episodes with reasoning, or None with no episodes."""
        return self.episodes_with_reasoning / self.episodes if self.episodes else None


def block_counts(block: ContentReasoning, family: Family) -> ReasoningCounts:
    text = reasoning_text(block)
    gpt_oss = family == "gpt-oss"
    return ReasoningCounts(
        blocks=1,
        empty_blocks=int(is_empty(text)),
        placeholder_blocks=int(is_placeholder(text)),
        json_blocks=int(gpt_oss and (is_json(text) or is_unparsed_details(block))),
        summary_detail_blocks=int(gpt_oss and summary_detail_text(block)),
    )


def turn_counts(message: ChatMessageAssistant, family: Family) -> ReasoningCounts:
    counts = sum((block_counts(block, family) for block in reasoning_blocks(message)), ReasoningCounts())
    return counts + ReasoningCounts(turns=1, turns_with_reasoning=int(counts.kept_blocks > 0))


def episode_counts(messages: Sequence[ChatMessage], family: Family) -> ReasoningCounts:
    """Counts over one episode's accepted conversation; only assistant messages are turns."""
    turns = [message for message in messages if isinstance(message, ChatMessageAssistant)]
    counts = sum((turn_counts(message, family) for message in turns), ReasoningCounts())
    return counts + ReasoningCounts(episodes=1, episodes_with_reasoning=int(counts.turns_with_reasoning > 0))


def sample_counts(sample: EvalSample, family: Family) -> ReasoningCounts:
    return episode_counts(sample.messages, family)


def log_counts(log: EvalLog) -> ReasoningCounts:
    """Counts over a log's samples, with the family from the log's model.

    Samples that errored are left out, as PLAN.md excludes them everywhere.
    A log read with header_only=True has no samples and raises ValueError.
    """
    if log.samples is None:
        raise ValueError(f"{log.location or 'log'} has no samples; read it without header_only")
    family = agent_family(log.eval.model)
    samples = [sample for sample in log.samples if sample.error is None]
    return sum((sample_counts(sample, family) for sample in samples), ReasoningCounts())


def agent_counts(logs: Iterable[EvalLog]) -> dict[str, ReasoningCounts]:
    """log_counts summed per agent, keyed by each log's model name."""
    totals: dict[str, ReasoningCounts] = {}
    for log in logs:
        totals[log.eval.model] = totals.get(log.eval.model, ReasoningCounts()) + log_counts(log)
    return totals
