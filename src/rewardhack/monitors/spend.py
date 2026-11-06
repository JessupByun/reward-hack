"""Spend ledger for monitor, summarizer and judge calls (PLAN.md, "Monitors" and "Budget").

logs/spend.json is one ledger shared by every LLM scorer and the scoring script, and record_call (or
arecord_call, from a scorer) adds an entry after every monitor, summarizer or judge call, pilot tuning included.
Each entry holds the scorer, role, cap, model, sample id and epoch, log path, the call's attempt number for
its prompt (1, or 2 for the retry of a reply that was not accepted), token counts, cost and a UTC timestamp,
and the cost is priced from costs.yaml with Inspect's compute_model_cost.
The caps are PLAN's Budget rows: the monitor cap ($12.00) covers M2, M3, the per-block Haiku summaries and
pilot tuning, and the judge cap ($12.00) covers the Sonnet judge; ROLE_CAPS maps each role to its cap.

A scoring run calls check_budget once, before it makes any call, with the prompts it will send; it raises
BudgetExceeded when the cap's spend so far plus the run's projected cost would cross the cap.
Each prompt takes one call, plus one retry from the identical prompt when the reply is not accepted, so at
most MAX_ATTEMPTS calls.
The projection is PLAN's pilot mean times samples: each role's mean cost per prompt over the ledger's entries
for the same scorer, role and model (their total cost over their first attempts, so the history's retries
are counted), times the planned prompts.
A scorer with no entries yet (its first pilot run) is projected at MAX_ATTEMPTS times worst_case_call_cost
per prompt instead.
Nothing stops a run midway: record_call only records, so a run whose projection was too low finishes and
the next run's check sees the overspend.
Scoring runs go one at a time: each run's setup takes the run lock beside the ledger (hold_run_lock) before
its check_budget and holds it until its process exits, so a second scoring run started meanwhile refuses
with RunLocked before any call instead of checking against spend the first run has not recorded yet.

The ledger fails closed: a missing, unreadable or corrupt ledger raises LedgerError on every read, so
check_budget and record_call refuse until it is repaired by hand, and nothing ever resets it to zero.
It is created once with init_ledger (`uv run python -m rewardhack.monitors.spend init`).
Updates hold a threading lock and an exclusive flock on a sidecar `<ledger>.lock`, then write a temporary
file and os.replace it over the ledger, so concurrent async calls, threads and processes never lose an entry
and a reader never sees a partial file.
The path is the `path` argument, else $REWARDHACK_SPEND_LEDGER, else logs/spend.json, so tests use tmp_path.
"""

import argparse
import contextlib
import fcntl
import functools
import json
import math
import os
import sys
import tempfile
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import anyio
import yaml
from inspect_ai.model import (
    Model,
    ModelCost,
    ModelInfo,
    ModelUsage,
    compute_model_cost,
    set_model_cost,
    set_model_info,
)

# The package is installed editable from src/rewardhack/monitors/, so the repo root is three levels up.
ROOT = Path(__file__).resolve().parents[3]
COSTS_FILE = ROOT / "costs.yaml"
DEFAULT_LEDGER = ROOT / "logs" / "spend.json"
LEDGER_ENV = "REWARDHACK_SPEND_LEDGER"
LEDGER_VERSION = 1
# Calls per prompt: the first, and one retry from the identical prompt when the reply is not accepted.
MAX_ATTEMPTS = 2

# PLAN's Budget table, in dollars.
CAPS: dict[str, float] = {"monitor": 12.00, "judge": 12.00}
# The cap each role's calls count toward; pilot tuning uses the same roles, so it counts toward the monitor cap.
ROLE_CAPS: dict[str, str] = {"monitor": "monitor", "summarizer": "monitor", "judge": "judge"}

# PLAN's Role settings give every monitor, summarizer and judge call max_tokens 8000.
MAX_OUTPUT_TOKENS = 8000
# Stated input bound for a call with no ledger history.
# A monitor prompt is the ticket, each turn's text, reasoning and tool calls, tool results cut to their last
# 2,000 characters and the final diff, within the 30-turn limit; the largest pilot transcript built that way
# is about 35,000 UTF-8 bytes (gpt-oss, 078_0 conflict), and a byte-level BPE tokenizer makes at most one token
# per byte, so 100K tokens leaves about 3x headroom.
# PLAN does not model Haiku's higher price above 100K prompt tokens and flags any sample that exceeds it.
WORST_CASE_INPUT_TOKENS = 100_000
# Framing tokens (roles, message boundaries, the instruction) that input_tokens_bound adds to the text's bytes.
PROMPT_OVERHEAD_TOKENS = 1_000

PRICE_KEYS = ("input", "output", "input_cache_write", "input_cache_read")
# mockllm/model is never listed in costs.yaml (PLAN.md, "Setup"); it is priced at zero here, as task.py
# registers it, so end-to-end tests with mockllm/model record $0 entries.
ZERO_COST_MODELS: dict[str, ModelCost] = {
    "mockllm/model": ModelCost(input=0, output=0, input_cache_write=0, input_cache_read=0),
}

# Score.metadata key that spend_metadata's dict goes under.
METADATA_KEY = "spend"

_TOKEN_FIELDS = ("input_tokens", "output_tokens", "input_tokens_cache_write", "input_tokens_cache_read", "reasoning_tokens")
# Serializes ledger updates between threads of one process; flock serializes processes.
_THREAD_LOCK = threading.Lock()


class LedgerError(RuntimeError):
    """The ledger is missing, unreadable or corrupt; nothing may be spent until it is repaired by hand."""


class BudgetExceeded(RuntimeError):
    """A run's projected cost would take a cap past its limit; raised before the run makes any call."""


class RunLocked(RuntimeError):
    """Another process holds the scoring-run lock; raised before the run makes any call."""


def _is_amount(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass(frozen=True)
class CallSpend:
    """One monitor, summarizer or judge call: a ledger entry.

    cost is what counts toward the cap: the call priced from costs.yaml, or Inspect's own figure for the call
    (inspect_cost, ModelUsage.total_cost) when that is higher, for example when Anthropic escalated the
    sample's cache writes to the 1-hour rate.
    usage_missing marks a call whose output had no usage; its tokens are 0 and its cost is
    worst_case_call_cost, so the ledger never undercounts.
    attempt is the call's number for its prompt, from 1; project counts a prompt once, at its first attempt.
    """

    scorer: str
    role: str
    cap: str
    model: str
    sample_id: str | int | None
    epoch: int | None
    log: str | None
    attempt: int
    input_tokens: int
    output_tokens: int
    input_tokens_cache_write: int
    input_tokens_cache_read: int
    reasoning_tokens: int
    cost: float
    inspect_cost: float | None
    usage_missing: bool
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: object) -> "CallSpend":
        """An entry read back from the ledger; any field of the wrong shape raises ValueError."""
        names = {f.name for f in fields(cls)}
        if not isinstance(data, dict) or set(data) != names:
            raise ValueError(f"an entry must have exactly the fields {sorted(names)}")
        for name in ("scorer", "role", "cap", "model", "timestamp"):
            if not isinstance(data[name], str) or not data[name]:
                raise ValueError(f"entry field {name} must be a non-empty string")
        if data["cap"] not in CAPS or ROLE_CAPS.get(data["role"]) != data["cap"]:
            raise ValueError(f"entry role {data['role']!r} does not count toward cap {data['cap']!r}")
        sample_id = data["sample_id"]
        if sample_id is not None and (isinstance(sample_id, bool) or not isinstance(sample_id, str | int)):
            raise ValueError("entry sample_id must be a string, an integer or null")
        if data["epoch"] is not None and not _is_count(data["epoch"]):
            raise ValueError("entry epoch must be a non-negative integer or null")
        if data["log"] is not None and not isinstance(data["log"], str):
            raise ValueError("entry log must be a string or null")
        if not _is_count(data["attempt"]) or data["attempt"] < 1:
            raise ValueError("entry attempt must be a positive integer")
        if not all(_is_count(data[name]) for name in _TOKEN_FIELDS):
            raise ValueError("entry token counts must be non-negative integers")
        if not _is_amount(data["cost"]) or not (data["inspect_cost"] is None or _is_amount(data["inspect_cost"])):
            raise ValueError("entry costs must be finite and non-negative")
        if not isinstance(data["usage_missing"], bool):
            raise ValueError("entry usage_missing must be a boolean")
        inspect_cost = None if data["inspect_cost"] is None else float(data["inspect_cost"])
        return cls(**{**data, "cost": float(data["cost"]), "inspect_cost": inspect_cost})


# --- prices ---


def load_costs(costs_file: str | Path = COSTS_FILE) -> dict[str, ModelCost]:
    """costs.yaml as ModelCost per model, plus ZERO_COST_MODELS.

    Every model must map exactly the four ModelCost keys to non-negative prices, as --model-cost-config needs,
    and mockllm/model must not be listed (PLAN.md, "Setup"); anything else raises ValueError.
    """
    data = yaml.safe_load(Path(costs_file).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{costs_file} must map model names to prices")
    costs: dict[str, ModelCost] = {}
    for model, prices in data.items():
        if not isinstance(prices, dict) or set(prices) != set(PRICE_KEYS) or not all(_is_amount(prices[k]) for k in PRICE_KEYS):
            raise ValueError(f"{costs_file}: {model} must map exactly {', '.join(PRICE_KEYS)} to non-negative prices")
        costs[str(model)] = ModelCost(**prices)
    if listed := set(costs) & set(ZERO_COST_MODELS):
        raise ValueError(f"{costs_file} must not list {sorted(listed)}; it is priced in spend.ZERO_COST_MODELS")
    return {**costs, **ZERO_COST_MODELS}


def price_key(model: str | Model, costs: Mapping[str, ModelCost]) -> str:
    """The costs.yaml name of a model given as a Model, its full name, or a name without its provider prefix.

    get_model(role="monitor") prints as anthropic/claude-haiku-5-5 and ModelOutput.model is claude-haiku-5-5;
    both give anthropic/claude-haiku-5-5. A model with no price raises ValueError, so no call goes unpriced.
    """
    name = str(model)
    if name in costs:
        return name
    matches = [key for key in costs if key.endswith(f"/{name}")]
    if len(matches) != 1:
        raise ValueError(f"no single price for {name!r} in costs.yaml (known: {sorted(costs)})")
    return matches[0]


def call_cost(
    model: str | Model, usage: ModelUsage, costs: Mapping[str, ModelCost] | None = None, cache_ttl: str | None = None
) -> float:
    """A call's cost in dollars from costs.yaml prices, computed by Inspect's compute_model_cost.

    Cache writes are priced at the 5-minute rate unless cache_ttl is "1h".
    """
    costs = load_costs() if costs is None else costs
    return compute_model_cost(costs[price_key(model, costs)], usage, cache_ttl)


def worst_case_call_cost(
    model: str | Model,
    input_tokens_bound: int = WORST_CASE_INPUT_TOKENS,
    costs: Mapping[str, ModelCost] | None = None,
) -> float:
    """The most one call can cost: MAX_OUTPUT_TOKENS of output plus input_tokens_bound of input.

    The input is priced at the dearest input rate: uncached, or written to the cache at the 1-hour rate
    (2x the base input price for Anthropic), whichever is higher.
    """
    if not _is_count(input_tokens_bound):
        raise ValueError("input_tokens_bound must be a non-negative integer")
    costs = load_costs() if costs is None else costs
    prices = costs[price_key(model, costs)]
    uncached = ModelUsage(input_tokens=input_tokens_bound, output_tokens=MAX_OUTPUT_TOKENS)
    written = ModelUsage(input_tokens=0, output_tokens=MAX_OUTPUT_TOKENS, input_tokens_cache_write=input_tokens_bound)
    return max(compute_model_cost(prices, uncached), compute_model_cost(prices, written, cache_ttl="1h"))


def input_tokens_bound(*texts: str) -> int:
    """An input-token bound for a prompt made of these texts: their UTF-8 bytes plus PROMPT_OVERHEAD_TOKENS.

    This rests on a byte-level BPE tokenizer making at most one token per byte; pass it as
    Planned.input_tokens_bound when a run's prompts are known before its first call.
    """
    return sum(len(text.encode("utf-8")) for text in texts) + PROMPT_OVERHEAD_TOKENS


def register_costs(*models: str | Model, costs: Mapping[str, ModelCost] | None = None) -> None:
    """set_model_cost for each model with its costs.yaml prices, which each LLM scorer calls (PLAN.md, "Monitors").

    `inspect score` takes no --model-cost-config, so without this the usage Inspect logs for a scorer's calls
    has no cost; mockllm/model gets zero-cost ModelInfo instead, as in task.py, because set_model_cost
    raises for a model missing from Inspect's database.
    """
    costs = load_costs() if costs is None else costs
    for model in models:
        key = price_key(model, costs)
        if key in ZERO_COST_MODELS:
            set_model_info(key, ModelInfo(cost=costs[key]))
        else:
            set_model_cost(key, costs[key])


# --- ledger file ---


def ledger_path(path: str | Path | None = None) -> Path:
    """The ledger: path if given, else $REWARDHACK_SPEND_LEDGER, else logs/spend.json at the repo root."""
    if path is None:
        path = os.environ.get(LEDGER_ENV) or DEFAULT_LEDGER
    return Path(path).expanduser()


def _lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold the ledger's thread lock and an exclusive flock on its sidecar lock file.

    The lock file stays on disk; the flock is released when its descriptor closes, also if the process dies.
    """
    with _THREAD_LOCK:
        try:
            fd = os.open(_lock_path(path), os.O_RDWR | os.O_CREAT, 0o644)
        except FileNotFoundError as e:
            raise _missing(path) from e
        except OSError as e:
            raise LedgerError(f"cannot lock {path} ({e}); spending stops until it is repaired by hand") from e
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)


def _reject_constant(name: str) -> None:
    raise ValueError(f"{name} is not a valid amount")


def _parse(text: str, path: Path) -> list[CallSpend]:
    try:
        data = json.loads(text, parse_constant=_reject_constant)
        if not isinstance(data, dict) or set(data) != {"version", "entries"}:
            raise ValueError("the ledger must be an object with exactly version and entries")
        if type(data["version"]) is not int or data["version"] != LEDGER_VERSION:
            raise ValueError(f"the ledger version must be {LEDGER_VERSION}")
        if not isinstance(data["entries"], list):
            raise ValueError("entries must be a list")
        return [CallSpend.from_dict(entry) for entry in data["entries"]]
    except (ValueError, TypeError, RecursionError) as e:
        raise LedgerError(f"{path} is corrupt ({e}); spending stops until it is repaired by hand") from e


def _missing(path: Path) -> LedgerError:
    return LedgerError(
        f"no ledger at {path}; spending stops rather than restart at zero. "
        "Create one only if none ever existed: uv run python -m rewardhack.monitors.spend init"
    )


def _read(path: Path) -> list[CallSpend]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise _missing(path) from e
    except (OSError, UnicodeDecodeError) as e:
        raise LedgerError(f"cannot read {path} ({e}); spending stops until it is repaired by hand") from e
    return _parse(text, path)


def _dump(entries: Sequence[CallSpend]) -> str:
    """The ledger as JSON with one entry per line."""
    lines = ",\n".join(json.dumps(entry.to_dict(), sort_keys=True, allow_nan=False) for entry in entries)
    body = f"\n{lines}\n" if lines else ""
    return f'{{"version": {LEDGER_VERSION}, "entries": [{body}]}}\n'


def _write(path: Path, entries: Sequence[CallSpend]) -> None:
    """Replace the ledger atomically: write and fsync a temporary file beside it, then os.replace it."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_dump(entries))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def init_ledger(path: str | Path | None = None) -> Path:
    """Create an empty ledger, once; raises FileExistsError if one is already there, valid or not."""
    target = ledger_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _locked(target):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"{target} already exists; the ledger is never reset")
        _write(target, [])
    return target


def ledger_entries(path: str | Path | None = None) -> list[CallSpend]:
    """Every entry in the ledger, oldest first; raises LedgerError if it is missing, unreadable or corrupt.

    A missing ledger is refused before the lock file is made, so a read (such as a dry run) writes nothing.
    """
    target = ledger_path(path)
    try:
        present = target.exists() or target.is_symlink()
    except OSError as e:
        raise LedgerError(f"cannot read {target} ({e}); spending stops until it is repaired by hand") from e
    if not present:
        raise _missing(target)
    with _locked(target):
        return _read(target)


def spent(entries: Iterable[CallSpend]) -> dict[str, float]:
    """Dollars spent per cap (every cap in CAPS, 0.0 when nothing counts toward it)."""
    entries = list(entries)
    return {cap: math.fsum(entry.cost for entry in entries if entry.cap == cap) for cap in CAPS}


def record_call(
    *,
    scorer: str,
    role: str,
    model: str | Model,
    usage: ModelUsage | None,
    sample_id: str | int | None = None,
    epoch: int | None = None,
    log: str | Path | None = None,
    attempt: int = 1,
    path: str | Path | None = None,
    costs: Mapping[str, ModelCost] | None = None,
) -> CallSpend:
    """Price one finished call and append it to the ledger; returns the entry for spend_metadata.

    Call it right after each monitor, summarizer or judge generate, with the role the scorer uses in
    check_budget, the output's usage and the call's attempt number for its prompt.
    It never refuses over the cap (runs never stop midway), but an unknown role or model raises ValueError
    and a missing, unreadable or corrupt ledger raises LedgerError, before anything is written.
    """
    if role not in ROLE_CAPS:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(ROLE_CAPS)}")
    if not scorer:
        raise ValueError("scorer must be a non-empty name")
    costs = load_costs() if costs is None else costs
    key = price_key(model, costs)
    if usage is None:
        tokens = dict.fromkeys(_TOKEN_FIELDS, 0)
        cost, inspect_cost = worst_case_call_cost(key, costs=costs), None
    else:
        tokens = {name: getattr(usage, name) or 0 for name in _TOKEN_FIELDS}
        inspect_cost = usage.total_cost
        cost = max(call_cost(key, usage, costs), inspect_cost or 0.0)
    entry = CallSpend.from_dict(
        {
            "scorer": scorer,
            "role": role,
            "cap": ROLE_CAPS[role],
            "model": key,
            "sample_id": sample_id,
            "epoch": epoch,
            "log": None if log is None else str(log),
            "attempt": attempt,
            **tokens,
            "cost": cost,
            "inspect_cost": inspect_cost,
            "usage_missing": usage is None,
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
        }
    )
    target = ledger_path(path)
    with _locked(target):
        entries = _read(target)
        _write(target, [*entries, entry])
    return entry


async def arecord_call(**kwargs: Any) -> CallSpend:
    """record_call in a worker thread, so a scorer's event loop keeps running while the ledger is rewritten.

    Each update rewrites the whole ledger (about 90 ms at PLAN's 6,000 calls), and the thread lock and flock
    keep concurrent updates safe; scorers await this after each generate.
    """
    return await anyio.to_thread.run_sync(functools.partial(record_call, **kwargs))


# --- one scoring run at a time ---

# The run locks this process holds, by lock path, each kept open until the process exits.
_RUN_LOCKS: dict[Path, int] = {}


def run_lock_path(path: str | Path | None = None) -> Path:
    target = ledger_path(path)
    return target.with_name(f"{target.name}.run.lock")


def hold_run_lock(path: str | Path | None = None) -> Path:
    """Take the scoring-run lock beside the ledger for the rest of this process, so runs go one at a time.

    The lock is a non-blocking exclusive flock on `<ledger>.run.lock`, held by an open descriptor that is
    never closed, so the operating system releases it only when the process exits (also if it dies).
    Taking it again in the same process is a no-op, so llm_monitors.Setup also refuses a second scorer
    while another of the same process is scoring. Raises RunLocked when another process holds it, and
    LedgerError when there is no ledger, before anything is created.
    """
    target = ledger_path(path)
    lock = run_lock_path(target)
    key = lock.resolve()
    with _THREAD_LOCK:
        if key in _RUN_LOCKS:
            return lock
        if not target.exists():
            raise _missing(target)
        try:
            fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as e:
            raise LedgerError(f"cannot open the run lock {lock} ({e}); spending stops until it is repaired by hand") from e
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            os.close(fd)
            raise RunLocked(f"another scoring run holds {lock}; scoring runs go one at a time, so wait for it to finish") from e
        except BaseException:
            os.close(fd)
            raise
        _RUN_LOCKS[key] = fd
    return lock


# --- projection ---


@dataclass(frozen=True)
class Planned:
    """The prompts a scoring run will send for one role and model, each taking up to MAX_ATTEMPTS calls.

    input_tokens_bound is used only when the ledger has no entry for this scorer, role and model.
    """

    role: str
    model: str
    prompts: int
    input_tokens_bound: int = WORST_CASE_INPUT_TOKENS


@dataclass(frozen=True)
class PartProjection:
    role: str
    model: str
    prompts: int
    # The cost of one prompt with its retries: the history's mean, or MAX_ATTEMPTS worst-case calls.
    per_prompt: float
    basis: Literal["ledger", "worst_case"]
    # How many prompts of the ledger the mean came from (0 for a worst-case bound).
    history: int

    @property
    def cost(self) -> float:
        return self.per_prompt * self.prompts


@dataclass(frozen=True)
class CapProjection:
    cap_name: str
    cap: float
    spent: float
    parts: tuple[PartProjection, ...]

    @property
    def projected(self) -> float:
        return math.fsum(part.cost for part in self.parts)

    @property
    def total(self) -> float:
        return self.spent + self.projected

    @property
    def within_cap(self) -> bool:
        return self.total <= self.cap

    def describe(self) -> str:
        parts = ", ".join(
            f"{p.role} {p.model} {p.prompts} prompts x ${p.per_prompt:.6f} ({p.basis}"
            + (f" mean of {p.history} prompts)" if p.basis == "ledger" else f" bound of {MAX_ATTEMPTS} calls)")
            for p in self.parts
        )
        return (
            f"{self.cap_name} cap ${self.cap:.2f}: spent ${self.spent:.4f} + projected ${self.projected:.4f} "
            f"= ${self.total:.4f} [{parts}]"
        )


def project(
    scorer: str,
    planned: Sequence[Planned],
    path: str | Path | None = None,
    costs: Mapping[str, ModelCost] | None = None,
) -> list[CapProjection]:
    """Each touched cap's spend so far and a run's projected cost, from one read of the ledger.

    A part's cost per prompt is the total cost of the ledger's entries for this scorer, role and model over
    the prompts they belong to (their first attempts), so retries count at the rate the history had them;
    with no entries it is MAX_ATTEMPTS times worst_case_call_cost with the part's input bound.
    Spend so far counts every entry of the cap, whatever its scorer.
    """
    if not planned:
        raise ValueError("a run must plan at least one part")
    costs = load_costs() if costs is None else costs
    entries = ledger_entries(path)
    parts: dict[str, list[PartProjection]] = {}
    for part in planned:
        if part.role not in ROLE_CAPS:
            raise ValueError(f"unknown role {part.role!r}; expected one of {sorted(ROLE_CAPS)}")
        if not _is_count(part.prompts):
            raise ValueError("planned prompts must be a non-negative integer")
        key = price_key(part.model, costs)
        history = [e for e in entries if e.scorer == scorer and e.role == part.role and e.model == key]
        # A history whose first attempts are missing (only by hand) still counts as one prompt, which overstates.
        prompts = max(sum(e.attempt == 1 for e in history), 1)
        if history:
            per_prompt, basis = math.fsum(e.cost for e in history) / prompts, "ledger"
        else:
            per_prompt, basis = MAX_ATTEMPTS * worst_case_call_cost(key, part.input_tokens_bound, costs), "worst_case"
        projection = PartProjection(part.role, key, part.prompts, per_prompt, basis, prompts if history else 0)
        parts.setdefault(ROLE_CAPS[part.role], []).append(projection)
    so_far = spent(entries)
    return [CapProjection(cap, CAPS[cap], so_far[cap], tuple(parts[cap])) for cap in sorted(parts)]


def check_budget(
    scorer: str,
    planned: Sequence[Planned],
    path: str | Path | None = None,
    costs: Mapping[str, ModelCost] | None = None,
) -> list[CapProjection]:
    """Refuse a scoring run before its first call if spent plus projected would cross a cap.

    Call it once per run, before any generate; it raises BudgetExceeded when any touched cap's total is above
    the cap (reaching the cap exactly is allowed), LedgerError when the ledger is missing, unreadable or
    corrupt, and returns the projections otherwise so the caller can print them.
    """
    projections = project(scorer, planned, path, costs)
    over = [p for p in projections if not p.within_cap]
    if over:
        raise BudgetExceeded(f"{scorer} refused before any call: " + "; ".join(p.describe() for p in over))
    return projections


# --- Score.metadata ---


def spend_metadata(calls: Iterable[CallSpend]) -> dict[str, Any]:
    """Tokens and cost of one score's calls, for Score.metadata[METADATA_KEY] (PLAN.md, "Monitors").

    The totals cover every call, by_role splits them per role, and a score that reused cached summaries
    and made no call reports zeros.
    """

    def totals(group: list[CallSpend]) -> dict[str, Any]:
        return {
            "calls": len(group),
            **{name: sum(getattr(call, name) for call in group) for name in _TOKEN_FIELDS},
            "cost": math.fsum(call.cost for call in group),
            "usage_missing": sum(call.usage_missing for call in group),
        }

    calls = list(calls)
    roles = sorted({call.role for call in calls})
    return {**totals(calls), "by_role": {role: totals([c for c in calls if c.role == role]) for role in roles}}


# --- command line ---


def main(argv: Sequence[str] | None = None) -> int:
    """init creates the ledger, status prints spend per cap, check is check_budget for a shell script.

    Exit codes: 0 allowed, 1 over a cap, 2 a missing, unreadable or corrupt ledger (or a usage error).
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ledger", help=f"ledger path (default ${LEDGER_ENV}, else {DEFAULT_LEDGER})")
    parser = argparse.ArgumentParser(prog="python -m rewardhack.monitors.spend", description="Monitor spend ledger.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", parents=[common], help="create an empty ledger; refuses if one exists")
    commands.add_parser("status", parents=[common], help="print spend per cap")
    check = commands.add_parser("check", parents=[common], help="refuse a run whose projected cost would cross its cap")
    check.add_argument("--scorer", required=True)
    check.add_argument(
        "--part", nargs=3, action="append", required=True, metavar=("ROLE", "MODEL", "PROMPTS"),
        help="prompts the run sends for one role and model; repeat for each role",
    )
    check.add_argument("--input-tokens-bound", type=int, default=WORST_CASE_INPUT_TOKENS)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            print(f"created {init_ledger(args.ledger)}")
        elif args.command == "status":
            entries = ledger_entries(args.ledger)
            for cap, amount in spent(entries).items():
                calls = sum(entry.cap == cap for entry in entries)
                print(f"{cap}: ${amount:.4f} of ${CAPS[cap]:.2f} spent in {calls} calls, ${CAPS[cap] - amount:.4f} left")
        else:
            planned = [Planned(role, model, int(prompts), args.input_tokens_bound) for role, model, prompts in args.part]
            for projection in check_budget(args.scorer, planned, args.ledger):
                print(projection.describe())
    except BudgetExceeded as e:
        print(e, file=sys.stderr)
        return 1
    except (LedgerError, FileExistsError, ValueError) as e:
        print(e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
