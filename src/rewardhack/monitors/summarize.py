"""Haiku summaries of gpt-oss reasoning blocks for the M3-summary arm (PLAN.md, "Monitors").

The summarizer makes one call per gpt-oss reasoning block (transcript.reasoning_blocks, which reads
the kept blocks of reasoning.message_reasoning) with the frozen prompt prompts/summarizer.txt: restate
what the model considered and decided, with no judgement of the agent and no added information.
The prompt is modeled on the CoT summarizer of Bootstrapped Monitoring (Xiao and Phuong, arXiv
2606.11998, Appendix G), which asks for a faithful summary and the summary alone, with PLAN's
no-judgement and no-added-information rules and a length target added.

Length target: the pilot's median per-block summary length from Haiku 5.5 agents only (PLAN.md), that
is, the median over every kept reasoning block of the non-errored samples of the pilot's
anthropic/claude-haiku-5-5 logs, whose text is Anthropic's summary of Haiku's thinking.
Tokens are counted by a TokenCounter; ModelTokenCounter wraps Inspect's Model.count_tokens, which for an
Anthropic model calls Anthropic's free count_tokens endpoint (inspect_ai/model/_providers/anthropic.py,
AnthropicAPI.count_tokens) and for mockllm/model makes a local estimate with no API call.
Each count is of one user message holding the text, so it includes a constant framing overhead of a
few tokens; the target and the block counts carry the same overhead, so comparing them is exact
(the median of x + c is the median of x plus c).
The target is computed once by `scripts/score_logs.py summary-target` and frozen in
prompts/summary_target.json with the median word count of the same blocks, which is the length the
prompt asks for, since a model can aim at a word count but cannot see its tokens.
Until that file exists, load_target raises SummaryTargetMissing and the m3_summary scorer refuses to run.

A block whose token count is below the target passes through unchanged and makes no call; every
other block is summarized, and summary_stats reports the share that passed through.
A reply is accepted when it stopped normally (stop_reason stop) with non-empty visible text; a
refused, cut-off or empty reply is retried once from the identical prompt, so a block takes at most
MAX_ATTEMPTS calls, each recorded in the spend ledger.
A block still without a summary is marked failed, and the m3_summary scorer then scores that sample
as unscored instead of monitoring an incomplete transcript.

Every block's outcome is cached in the m3_summary score's metadata under CACHE_KEY, keyed by its turn,
index and text hash together with the prompt hash, the target and the summarizer model, so a later
m3_summary run on the same scored log (stored as m3_summary1, m3_summary2, ...) reuses it without a
call; failed blocks are retried.

PromptTemplate, MAX_ATTEMPTS and generate_until are shared with rewardhack.monitors.llm_monitors.
"""

import hashlib
import json
import math
import re
import statistics
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessageUser, Model, ModelOutput
from inspect_ai.scorer import Score

from rewardhack.monitors.spend import MAX_ATTEMPTS, CallSpend
from rewardhack.monitors.transcript import ReasoningBlock, reasoning_blocks

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_FILE = PROMPTS_DIR / "summarizer.txt"
TARGET_FILE = PROMPTS_DIR / "summary_target.json"
TARGET_VERSION = 1
# The agent whose provider summaries set the length target (PLAN.md: Haiku 5.5 only, because Sonnet 5.5
# mixes progress updates into its summaries).
TARGET_AGENT = "anthropic/claude-haiku-5-5"
# The scorer whose Score.metadata caches the summaries, and the metadata key that holds them.
SCORER = "m3_summary"
CACHE_KEY = "summaries"
# MAX_ATTEMPTS (spend.MAX_ATTEMPTS, imported above) is the calls per prompt: the first, and one retry from
# the identical prompt when the reply is not accepted; spend projects every prompt at that many calls
# until the ledger has history.

Mode = Literal["passed_through", "summarized", "failed"]


# --- prompts and calls (shared with llm_monitors) ---

PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")


@dataclass(frozen=True)
class PromptTemplate:
    """A frozen prompt file with {{NAME}} placeholders, filled in one pass.

    One pass means a value that itself contains {{NAME}} text (an agent can write anything) is never
    filled again.
    """

    path: Path
    text: str

    @classmethod
    def load(cls, path: Path) -> "PromptTemplate":
        return cls(path, path.read_text(encoding="utf-8"))

    @property
    def fields(self) -> frozenset[str]:
        return frozenset(PLACEHOLDER.findall(self.text))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def render(self, **values: str) -> str:
        if set(values) != self.fields:
            raise ValueError(f"{self.path.name} takes exactly {sorted(self.fields)}, got {sorted(values)}")
        return PLACEHOLDER.sub(lambda match: values[match.group(1)], self.text)


# Records one finished call, with its attempt number for its prompt (from 1), in the spend ledger and
# returns its entry (llm_monitors binds the scorer, role, sample and log).
Record = Callable[[ModelOutput, int], Awaitable[CallSpend]]
# Whether a reply is accepted, and the status kept for it ("ok" when accepted, else why not).
Check = Callable[[ModelOutput], tuple[bool, str]]


async def generate_until(
    model: Model, prompt: str, check: Check, record: Record, attempts: int = MAX_ATTEMPTS
) -> tuple[ModelOutput | None, list[dict[str, Any]], list[CallSpend]]:
    """Call the model with the prompt as one user message until check accepts a reply, at most attempts times.

    Returns the accepted output (None if none was), one dict per attempt (status, stop_reason, the
    visible completion and any error), and the ledger entry of every call.
    Each call is recorded right after it returns, accepted or not, with its attempt number.
    A call that raises (after Inspect's own retries inside generate) returned no usage, so it is not
    recorded; it is kept as an attempt with status error, and no further attempt is made.
    The model's role config is used as it is: no temperature, top_p or top_k is ever passed.
    """
    replies: list[dict[str, Any]] = []
    calls: list[CallSpend] = []
    for attempt in range(1, attempts + 1):
        try:
            output = await model.generate([ChatMessageUser(content=prompt)])
        except Exception as e:
            replies.append({"status": "error", "stop_reason": None, "completion": None, "error": f"{type(e).__name__}: {e}"})
            return None, replies, calls
        calls.append(await record(output, attempt))
        accepted, status = check(output)
        replies.append({"status": status, "stop_reason": output.stop_reason, "completion": output.completion, "error": None})
        if accepted:
            return output, replies, calls
    return None, replies, calls


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- token counting ---


class TokenCounter(Protocol):
    """Counts the tokens of a text; name is recorded with the target, which only that counter's counts match."""

    @property
    def name(self) -> str: ...

    async def count(self, text: str) -> int: ...


class ModelTokenCounter:
    """Model.count_tokens of one model: Anthropic's free count_tokens endpoint for an Anthropic model.

    Inspect retries 429s and timeouts and bounds the concurrency of these calls in their own pool
    (inspect_ai/model/_model.py, Model.count_tokens); for mockllm/model the count is a local estimate.
    """

    def __init__(self, model: Model):
        self.model = model

    @property
    def name(self) -> str:
        return str(self.model)

    async def count(self, text: str) -> int:
        return await self.model.count_tokens(text)


# --- the length target ---


class SummaryTargetMissing(RuntimeError):
    """No valid frozen summary target; m3_summary refuses to run until score_logs.py summary-target writes one."""


@dataclass(frozen=True)
class SummaryTarget:
    """The frozen length target: medians over the Haiku 5.5 agents' per-block summaries in the pilot.

    tokens is the median count by the counter named in counter, the unit that pass-through compares,
    and words is the median whitespace-separated word count, the length the prompt asks for.
    """

    tokens: int
    words: int
    blocks: int
    counter: str
    agent: str
    logs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"version": TARGET_VERSION, **asdict(self), "logs": list(self.logs)}

    @classmethod
    def from_dict(cls, data: object) -> "SummaryTarget":
        names = {"version", "tokens", "words", "blocks", "counter", "agent", "logs"}
        if not isinstance(data, dict) or set(data) != names or data["version"] != TARGET_VERSION:
            raise ValueError(f"a summary target must have exactly the fields {sorted(names)} and version {TARGET_VERSION}")
        for name in ("tokens", "words", "blocks"):
            if type(data[name]) is not int or data[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        if data["agent"] != TARGET_AGENT or not isinstance(data["counter"], str) or not data["counter"]:
            raise ValueError(f"the target must come from {TARGET_AGENT} and name its counter")
        if not isinstance(data["logs"], list) or not data["logs"] or not all(isinstance(log, str) for log in data["logs"]):
            raise ValueError("logs must be a non-empty list of paths")
        return cls(data["tokens"], data["words"], data["blocks"], data["counter"], data["agent"], tuple(data["logs"]))


def load_target(path: str | Path | None = None) -> SummaryTarget:
    """The frozen target (TARGET_FILE by default); a missing, unreadable or invalid file raises SummaryTargetMissing."""
    path = Path(path or TARGET_FILE)
    try:
        return SummaryTarget.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as e:
        raise SummaryTargetMissing(
            f"no summary target at {path}; compute it once with `uv run --env-file .env python scripts/score_logs.py summary-target`"
        ) from e
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise SummaryTargetMissing(f"{path} is not a valid summary target ({e}); m3_summary refuses to run") from e


def write_target(target: SummaryTarget, path: str | Path | None = None) -> Path:
    """Write the frozen target once (TARGET_FILE by default); raises FileExistsError if a target is already there."""
    path = Path(path or TARGET_FILE)
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(target.to_dict(), indent=2) + "\n")
    return path


def target_texts(logs: Iterable[EvalLog]) -> list[str]:
    """Every kept reasoning block of the non-errored samples of the TARGET_AGENT logs, in order.

    Other agents' logs are skipped; a block is kept as reasoning.message_reasoning keeps it (not empty and
    not the interrupted-thinking placeholder), over accepted assistant messages only.
    """
    texts: list[str] = []
    for log in logs:
        if log.eval.model != TARGET_AGENT:
            continue
        if log.samples is None:
            raise ValueError(f"{log.location or 'log'} has no samples; read it without header_only")
        for sample in log.samples:
            if sample.error is None:
                texts += [block.text for block in reasoning_blocks(sample.messages)]
    return texts


def median_int(values: Sequence[int]) -> int:
    """The median, rounded half up to an integer."""
    return math.floor(statistics.median(values) + 0.5)


def word_count(text: str) -> int:
    return len(text.split())


async def compute_target(logs: Sequence[tuple[str, EvalLog]], counter: TokenCounter) -> SummaryTarget:
    """The target from (name, log) pairs; only the TARGET_AGENT logs count, and there must be some blocks."""
    used = [(name, log) for name, log in logs if log.eval.model == TARGET_AGENT]
    if not used:
        raise ValueError(f"none of the logs is a {TARGET_AGENT} agent log")
    texts = target_texts(log for _, log in used)
    if not texts:
        raise ValueError(f"the {TARGET_AGENT} logs have no kept reasoning block")
    tokens = [await counter.count(text) for text in texts]
    return SummaryTarget(
        tokens=median_int(tokens),
        words=median_int([word_count(text) for text in texts]),
        blocks=len(texts),
        counter=counter.name,
        agent=TARGET_AGENT,
        logs=tuple(name for name, _ in used),
    )


# --- summaries ---


def prompt_template() -> PromptTemplate:
    return PromptTemplate.load(PROMPT_FILE)


def summary_prompt(text: str, target: SummaryTarget, template: PromptTemplate | None = None) -> str:
    return (template or prompt_template()).render(WORDS=str(target.words), REASONING=text)


def check_summary(output: ModelOutput) -> tuple[bool, str]:
    """A summary is accepted when the reply stopped normally with non-empty visible text."""
    if output.stop_reason != "stop":
        return False, f"stop_reason_{output.stop_reason}"
    if not output.completion.strip():
        return False, "empty"
    return True, "ok"


@dataclass(frozen=True)
class BlockSummary:
    """One reasoning block's outcome, as cached in Score.metadata[CACHE_KEY].

    tokens is the block's count by the target's counter (None when counting failed), summary the
    accepted summary (None unless mode is summarized), and reused marks an entry taken from an earlier
    m3_summary score.
    """

    turn: int
    index: int
    sha256: str
    chars: int
    tokens: int | None
    mode: Mode
    summary: str | None
    prompt_sha256: str
    target_tokens: int
    model: str
    attempts: tuple[dict[str, Any], ...] = ()
    reused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "attempts": list(self.attempts)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BlockSummary":
        entry = cls(**{**data, "attempts": tuple(data.get("attempts") or ())})
        if entry.mode not in ("passed_through", "summarized", "failed") or (entry.mode == "summarized") != bool(entry.summary):
            raise ValueError("a cached summary has an unknown mode or a summary that does not match it")
        return entry

    @property
    def key(self) -> tuple[int, int, str]:
        return (self.turn, self.index, self.sha256)

    def replacement(self, block: ReasoningBlock) -> str:
        """The text the M3-summary transcript shows for the block: its summary, or the block itself when passed through."""
        if self.key != block_key(block):
            raise ValueError(f"the summary for turn {self.turn} block {self.index} is for a different text")
        if self.mode == "failed":
            raise ValueError(f"turn {self.turn} block {self.index} has no summary")
        return block.text if self.mode == "passed_through" else str(self.summary)


def block_key(block: ReasoningBlock) -> tuple[int, int, str]:
    return (block.turn, block.index, sha256_text(block.text))


# Inspect names a repeated scorer m3_summary1, m3_summary2, ... (and name-1 if that too were taken).
SCORE_NAME = re.compile(rf"^{SCORER}(-?\d+)?$")


def cached_summaries(
    scores: Mapping[str, Score] | None, *, prompt_sha256: str, target: SummaryTarget, model: str
) -> dict[tuple[int, int, str], BlockSummary]:
    """Reusable entries from earlier m3_summary scores (m3_summary, m3_summary1, ...), later scores winning.

    An entry is reusable when it is not failed and was made with the same prompt, target and model;
    an entry that does not parse is ignored, so its block is done again.
    """
    found: dict[tuple[int, int, str], BlockSummary] = {}
    for name, score in (scores or {}).items():
        if not SCORE_NAME.match(name):
            continue
        for raw in (score.metadata or {}).get(CACHE_KEY) or []:
            try:
                entry = BlockSummary.from_dict(raw)
            except (TypeError, ValueError):
                continue
            current = (entry.prompt_sha256, entry.target_tokens, entry.model) == (prompt_sha256, target.tokens, model)
            if entry.mode != "failed" and current:
                found[entry.key] = entry
    return found


async def summarize_blocks(
    blocks: Sequence[ReasoningBlock],
    *,
    model: Model,
    counter: TokenCounter,
    target: SummaryTarget,
    cache: Mapping[tuple[int, int, str], BlockSummary],
    record: Record,
    template: PromptTemplate | None = None,
) -> tuple[list[BlockSummary], list[CallSpend]]:
    """Each block's outcome, in order, and the ledger entries of the summarizer calls made.

    A cached block is reused; any other block is counted, passed through when its count is below
    target.tokens, and otherwise summarized by one call (two when the first reply is not accepted).
    A block whose count raises (after Inspect's retries) is failed without a call.
    Blocks are done one at a time; inspect score already runs the samples concurrently.
    """
    if counter.name != target.counter:
        raise ValueError(f"the target was counted by {target.counter}, not {counter.name}; the counts would not compare")
    template = template or prompt_template()
    done: list[BlockSummary] = []
    calls: list[CallSpend] = []
    for block in blocks:
        key = block_key(block)
        if key in cache:
            done.append(replace(cache[key], reused=True))
            continue
        try:
            tokens: int | None = await counter.count(block.text)
        except Exception as e:
            # Inspect already retried; the block fails, so its sample is left unscored rather than the whole run stopping.
            tokens, error = None, {"status": "error", "stop_reason": None, "completion": None, "error": f"count_tokens: {type(e).__name__}: {e}"}
        common = {
            "turn": block.turn,
            "index": block.index,
            "sha256": key[2],
            "chars": len(block.text),
            "tokens": tokens,
            "prompt_sha256": template.sha256,
            "target_tokens": target.tokens,
            "model": str(model),
        }
        if tokens is None:
            done.append(BlockSummary(**common, mode="failed", summary=None, attempts=(error,)))
            continue
        if tokens < target.tokens:
            done.append(BlockSummary(**common, mode="passed_through", summary=None))
            continue
        output, attempts, spent = await generate_until(model, summary_prompt(block.text, target, template), check_summary, record)
        calls += spent
        if output is None:
            done.append(BlockSummary(**common, mode="failed", summary=None, attempts=tuple(attempts)))
        else:
            done.append(BlockSummary(**common, mode="summarized", summary=output.completion.strip(), attempts=tuple(attempts)))
    return done, calls


def summary_stats(entries: Sequence[BlockSummary]) -> dict[str, Any]:
    """Counts per mode, the reused entries, and the share of blocks that passed through (None with no blocks)."""
    modes = {mode: sum(entry.mode == mode for entry in entries) for mode in ("passed_through", "summarized", "failed")}
    summaries = [entry.summary for entry in entries if entry.summary]
    return {
        "blocks": len(entries),
        **modes,
        "reused": sum(entry.reused for entry in entries),
        "passed_through_share": modes["passed_through"] / len(entries) if entries else None,
        "summary_words": [word_count(summary) for summary in summaries],
    }
