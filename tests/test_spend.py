"""Tests for the spend ledger in src/rewardhack/monitors/spend.py.

Every test runs with $REWARDHACK_SPEND_LEDGER pointed into tmp_path, so the real logs/spend.json is never touched.
No test makes a paid model call; the one model call is mockllm/model.
"""

import asyncio
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from inspect_ai.model import ModelCost, ModelOutput, ModelUsage, compute_model_cost, get_model, get_model_info
from inspect_ai.model import _model_info as inspect_model_info

from rewardhack.monitors import spend
from rewardhack.monitors.spend import BudgetExceeded, LedgerError, Planned

HAIKU = "anthropic/claude-haiku-5-5"
SONNET = "anthropic/claude-sonnet-5-5"
GPT_OSS = "openrouter/openai/gpt-oss-120b"

# Sample totals from the pilot logs (haiku-normal and sonnet-normal), each with the total_cost Inspect logged for
# it under --model-cost-config costs.yaml: an independent check of the ledger's arithmetic.
HAIKU_USAGE = {"input_tokens": 10, "output_tokens": 1498, "input_tokens_cache_write": 3369, "input_tokens_cache_read": 4901}
HAIKU_LOGGED_COST = 0.001220135
SONNET_USAGE = {"input_tokens": 8, "output_tokens": 782, "input_tokens_cache_write": 2047, "input_tokens_cache_read": 2491}
SONNET_LOGGED_COST = 0.0132026


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The ledger path every test sees through the environment; not created."""
    path = tmp_path / "ledger" / "spend.json"
    monkeypatch.setenv(spend.LEDGER_ENV, str(path))
    return path


@pytest.fixture
def ledger(isolated_ledger: Path) -> Path:
    """An empty ledger at the isolated path."""
    return spend.init_ledger()


def record(scorer: str = "m2", role: str = "monitor", model: str = HAIKU, attempt: int = 1, **usage: int) -> spend.CallSpend:
    return spend.record_call(scorer=scorer, role=role, model=model, usage=ModelUsage(**usage), sample_id="s", epoch=1, attempt=attempt)


def test_ledger_path_resolution(tmp_path: Path, isolated_ledger: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert spend.ledger_path() == isolated_ledger
    assert spend.ledger_path(tmp_path / "given.json") == tmp_path / "given.json"
    monkeypatch.delenv(spend.LEDGER_ENV)
    assert spend.ledger_path() == spend.ROOT / "logs" / "spend.json"
    assert (spend.ROOT / "costs.yaml").is_file() and spend.COSTS_FILE == spend.ROOT / "costs.yaml"


@pytest.mark.parametrize(
    ("model", "usage", "logged"),
    [(HAIKU, HAIKU_USAGE, HAIKU_LOGGED_COST), (SONNET, SONNET_USAGE, SONNET_LOGGED_COST)],
    ids=["haiku", "sonnet"],
)
def test_cost_matches_costs_yaml(model: str, usage: dict[str, int], logged: float) -> None:
    prices = yaml.safe_load(spend.COSTS_FILE.read_text())[model]
    by_hand = (
        usage["input_tokens"] * prices["input"]
        + usage["output_tokens"] * prices["output"]
        + usage["input_tokens_cache_write"] * prices["input_cache_write"]
        + usage["input_tokens_cache_read"] * prices["input_cache_read"]
    ) / 1_000_000
    cost = spend.call_cost(model, ModelUsage(**usage))
    assert cost == pytest.approx(by_hand, rel=1e-12)
    assert cost == pytest.approx(logged, rel=1e-12)


def test_record_call_writes_one_entry_per_line(ledger: Path) -> None:
    usage = ModelUsage(**HAIKU_USAGE, reasoning_tokens=455)
    entry = spend.record_call(
        scorer="m2", role="monitor", model=HAIKU, usage=usage, sample_id="078_0", epoch=3, log="logs/scored/a.eval"
    )
    assert entry.cost == pytest.approx(HAIKU_LOGGED_COST, rel=1e-12)
    assert (entry.scorer, entry.role, entry.cap, entry.model) == ("m2", "monitor", "monitor", HAIKU)
    assert (entry.sample_id, entry.epoch, entry.log) == ("078_0", 3, "logs/scored/a.eval")
    assert (entry.input_tokens, entry.output_tokens, entry.reasoning_tokens) == (10, 1498, 455)
    assert (entry.input_tokens_cache_write, entry.input_tokens_cache_read) == (3369, 4901)
    assert entry.inspect_cost is None and not entry.usage_missing and entry.timestamp.endswith("+00:00")
    assert spend.ledger_entries() == [entry]
    lines = ledger.read_text().splitlines()
    assert lines[0] == '{"version": 1, "entries": [' and lines[-1] == "]}" and len(lines) == 3
    assert json.loads(lines[1]) == entry.to_dict()


def test_model_names_resolve_to_costs_yaml_keys(ledger: Path) -> None:
    costs = spend.load_costs()
    assert spend.price_key("claude-haiku-5-5", costs) == HAIKU
    assert spend.price_key("openai/gpt-oss-120b", costs) == GPT_OSS
    assert spend.price_key(get_model("mockllm/model", memoize=False), costs) == "mockllm/model"
    assert record(model="claude-sonnet-5-5", output_tokens=1).model == SONNET
    before = ledger.read_bytes()
    with pytest.raises(ValueError, match="no single price"):
        record(model="anthropic/claude-opus-5-5", output_tokens=1)
    with pytest.raises(ValueError, match="unknown role"):
        record(role="grader", output_tokens=1)
    assert ledger.read_bytes() == before


def test_higher_inspect_cost_is_charged(ledger: Path) -> None:
    priced = spend.call_cost(HAIKU, ModelUsage(**HAIKU_USAGE))
    higher = spend.record_call(
        scorer="m2", role="monitor", model=HAIKU, usage=ModelUsage(**HAIKU_USAGE, total_cost=2 * priced)
    )
    lower = spend.record_call(
        scorer="m2", role="monitor", model=HAIKU, usage=ModelUsage(**HAIKU_USAGE, total_cost=priced / 2)
    )
    assert (higher.cost, higher.inspect_cost) == (2 * priced, 2 * priced)
    assert (lower.cost, lower.inspect_cost) == (priced, priced / 2)


def test_missing_usage_is_charged_the_worst_case(ledger: Path) -> None:
    entry = spend.record_call(scorer="m2", role="monitor", model=HAIKU, usage=None)
    assert entry.usage_missing and entry.cost == spend.worst_case_call_cost(HAIKU)
    assert entry.input_tokens == entry.output_tokens == 0


def test_mockllm_is_priced_at_zero_and_never_in_costs_yaml(tmp_path: Path) -> None:
    assert "mockllm/model" not in yaml.safe_load(spend.COSTS_FILE.read_text())
    assert spend.load_costs()["mockllm/model"] == ModelCost(input=0, output=0, input_cache_write=0, input_cache_read=0)
    assert spend.call_cost("mockllm/model", ModelUsage(input_tokens=10**6, output_tokens=10**6)) == 0


@pytest.mark.parametrize(
    "prices",
    [
        {"input": 1, "output": 1, "input_cache_write": 1},
        {"input": 1, "output": 1, "input_cache_write": 1, "input_cache_read": 1, "extra": 1},
        {"input": 1, "output": -1, "input_cache_write": 1, "input_cache_read": 1},
        {"input": 1, "output": "1", "input_cache_write": 1, "input_cache_read": 1},
    ],
    ids=["missing-key", "extra-key", "negative", "string"],
)
def test_load_costs_rejects_incomplete_prices(tmp_path: Path, prices: dict[str, object]) -> None:
    costs_file = tmp_path / "costs.yaml"
    costs_file.write_text(yaml.safe_dump({HAIKU: prices}))
    with pytest.raises(ValueError, match="must map exactly"):
        spend.load_costs(costs_file)


def test_load_costs_rejects_listed_mockllm(tmp_path: Path) -> None:
    costs_file = tmp_path / "costs.yaml"
    costs_file.write_text(yaml.safe_dump({"mockllm/model": dict.fromkeys(spend.PRICE_KEYS, 0)}))
    with pytest.raises(ValueError, match="must not list"):
        spend.load_costs(costs_file)


def test_worst_case_bounds() -> None:
    # Haiku: 100K input written to the cache at the 1-hour rate (0.125 x 1.6 = $0.20/MTok), plus 8000 x $0.50/MTok.
    assert spend.worst_case_call_cost(HAIKU) == pytest.approx(0.020 + 0.004)
    # Sonnet: 100K input at 2.50 x 1.6 = $4.00/MTok, plus 8000 x $10/MTok.
    assert spend.worst_case_call_cost(SONNET) == pytest.approx(0.40 + 0.08)
    assert spend.worst_case_call_cost(HAIKU, input_tokens_bound=0) == pytest.approx(0.004)
    assert spend.worst_case_call_cost(SONNET, input_tokens_bound=5_000) == pytest.approx(0.02 + 0.08)
    assert spend.input_tokens_bound("abc", "é") == 3 + 2 + spend.PROMPT_OVERHEAD_TOKENS


def test_projection_arithmetic(ledger: Path) -> None:
    # Haiku output costs $0.50/MTok, so 2000 and 6000 output tokens cost $0.001 and $0.003.
    record("m2", output_tokens=2000)
    record("m2", output_tokens=6000)
    other_scorer = record("m3", output_tokens=20_000)
    other_model = record("m2", model=SONNET, output_tokens=1000)
    summary = record("m3_summary", role="summarizer", output_tokens=400)
    judge = record("judge", role="judge", model=SONNET, output_tokens=1000)

    [monitor] = spend.project("m2", [Planned("monitor", HAIKU, 100)])
    [part] = monitor.parts
    assert (part.basis, part.history, part.prompts, part.model) == ("ledger", 2, 100, HAIKU)
    assert part.per_prompt == pytest.approx(0.002) and part.cost == pytest.approx(0.2)
    # Spend so far is every entry of the cap, from any scorer or model; the judge's entry is not in it.
    assert monitor.spent == pytest.approx(0.001 + 0.003 + other_scorer.cost + other_model.cost + summary.cost)
    assert monitor.total == pytest.approx(monitor.spent + 0.2) and monitor.within_cap

    planned = [Planned("summarizer", HAIKU, 300), Planned("monitor", HAIKU, 20, input_tokens_bound=5_000)]
    [mixed] = spend.project("m3_summary", planned)
    assert [p.basis for p in mixed.parts] == ["ledger", "worst_case"]
    assert mixed.parts[0].per_prompt == pytest.approx(summary.cost)
    # With no history a prompt is bounded at MAX_ATTEMPTS worst-case calls, since its reply may be retried.
    assert spend.MAX_ATTEMPTS == 2
    assert mixed.parts[1].per_prompt == pytest.approx(2 * spend.worst_case_call_cost(HAIKU, 5_000))
    assert mixed.projected == pytest.approx(300 * summary.cost + 20 * 2 * spend.worst_case_call_cost(HAIKU, 5_000))

    caps = spend.project("judge", [Planned("judge", SONNET, 10), Planned("monitor", HAIKU, 1)])
    assert [c.cap_name for c in caps] == ["judge", "monitor"]
    assert caps[0].spent == judge.cost and caps[0].projected == pytest.approx(10 * judge.cost)
    assert caps[1].parts[0].basis == "worst_case"


def test_a_retried_prompt_is_projected_at_its_whole_cost(ledger: Path) -> None:
    """The ledger mean is per prompt (first attempts), so the history's retries are in the projection."""
    # Three prompts at $0.001 a call, one of them retried: $0.004 over 3 prompts.
    for attempt in (1, 1, 1, 2):
        record("m2", attempt=attempt, output_tokens=2000)
    [monitor] = spend.project("m2", [Planned("monitor", HAIKU, 30)])
    [part] = monitor.parts
    assert (part.basis, part.history) == ("ledger", 3)
    assert part.per_prompt == pytest.approx(0.004 / 3) and part.cost == pytest.approx(0.04)
    assert "30 prompts x $0.001333 (ledger mean of 3 prompts)" in monitor.describe()
    # The first run of a scorer, whose every reply is retried, stays within its worst-case projection.
    [first] = spend.project("m3", [Planned("monitor", HAIKU, 2, input_tokens_bound=5_000)])
    worst = spend.worst_case_call_cost(HAIKU, 5_000)
    assert first.parts[0].per_prompt == pytest.approx(2 * worst)
    usage = {"input_tokens_cache_write": 5_000, "output_tokens": spend.MAX_OUTPUT_TOKENS}
    made = [record("m3", attempt=attempt, **usage) for _ in range(2) for attempt in (1, 2)]
    assert math.fsum(call.cost for call in made) <= first.projected
    assert [e.attempt for e in spend.ledger_entries()] == [1, 1, 1, 2, 1, 2, 1, 2]


def test_projection_rejects_bad_plans(ledger: Path) -> None:
    with pytest.raises(ValueError, match="at least one part"):
        spend.project("m2", [])
    with pytest.raises(ValueError, match="unknown role"):
        spend.project("m2", [Planned("grader", HAIKU, 1)])
    with pytest.raises(ValueError, match="non-negative integer"):
        spend.project("m2", [Planned("monitor", HAIKU, -1)])
    with pytest.raises(ValueError, match="attempt must be a positive integer"):
        record(attempt=0, output_tokens=1)
    with pytest.raises(ValueError, match="no single price"):
        spend.project("m2", [Planned("monitor", "anthropic/claude-opus-5-5", 1)])


def test_cap_boundary(ledger: Path) -> None:
    # Sonnet output costs $10/MTok, so these entries cost exactly $11.25 and $0.25 in binary floating point.
    record("judge_tuning", role="judge", model=SONNET, output_tokens=1_125_000)
    record("judge", role="judge", model=SONNET, output_tokens=25_000)
    [at_cap] = spend.check_budget("judge", [Planned("judge", SONNET, 2)])
    assert (at_cap.spent, at_cap.projected, at_cap.total) == (11.5, 0.5, 12.0)
    with pytest.raises(BudgetExceeded, match=r"judge refused before any call: judge cap \$12\.00"):
        spend.check_budget("judge", [Planned("judge", SONNET, 3)])
    # The judge's spend leaves the monitor cap untouched.
    assert spend.check_budget("m2", [Planned("monitor", HAIKU, 1)])[0].spent == 0


def test_refusal_comes_before_any_call(ledger: Path) -> None:
    generated: list[bool] = []

    def outputs():
        while True:
            generated.append(True)
            yield ModelOutput.from_content("mockllm/model", "Score: 3")

    model = get_model("mockllm/model", custom_outputs=outputs(), memoize=False)

    async def scoring_run(calls: int) -> None:
        # The order a scorer follows: one check for the whole run, then each call and its entry.
        spend.check_budget("m2", [Planned("monitor", HAIKU, calls)])
        for _ in range(calls):
            output = await model.generate("transcript")
            await spend.arecord_call(scorer="m2", role="monitor", model=model, usage=output.usage)

    before = ledger.read_bytes()
    # With no history each Haiku prompt is bounded at two calls of $0.024, and 600 of them ($28.80) cross the $12 cap.
    with pytest.raises(BudgetExceeded):
        asyncio.run(scoring_run(600))
    assert generated == [] and ledger.read_bytes() == before
    asyncio.run(scoring_run(2))
    assert len(generated) == 2
    assert [(e.model, e.cost) for e in spend.ledger_entries()] == [("mockllm/model", 0.0)] * 2


def test_concurrent_async_and_thread_updates(ledger: Path) -> None:
    usage = ModelUsage(**HAIKU_USAGE)

    def one(sample_id: str) -> spend.CallSpend:
        return spend.record_call(scorer="m2", role="monitor", model=HAIKU, usage=usage, sample_id=sample_id)

    async def in_loop(i: int) -> spend.CallSpend:
        await asyncio.sleep(0)
        return one(f"loop-{i}")

    async def in_thread(i: int) -> spend.CallSpend:
        return await asyncio.to_thread(one, f"thread-{i}")

    async def awaited(i: int) -> spend.CallSpend:
        return await spend.arecord_call(scorer="m2", role="monitor", model=HAIKU, usage=usage, sample_id=f"async-{i}")

    async def run() -> list[spend.CallSpend]:
        return await asyncio.gather(
            *(in_loop(i) for i in range(40)), *(in_thread(i) for i in range(40)), *(awaited(i) for i in range(40))
        )

    recorded = asyncio.run(run())
    entries = spend.ledger_entries()
    assert len(entries) == 120
    assert {e.sample_id for e in entries} == {e.sample_id for e in recorded} and len(set(recorded)) == 120
    assert spend.spent(entries)["monitor"] == pytest.approx(120 * HAIKU_LOGGED_COST, rel=1e-12)


CHILD = """
import sys, time
from pathlib import Path
from inspect_ai.model import ModelUsage
from rewardhack.monitors.spend import record_call
name, ready, go, count = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
ready.touch()
while not go.exists():
    time.sleep(0.005)
for i in range(count):
    record_call(scorer="m2", role="monitor", model="anthropic/claude-haiku-5-5",
                usage=ModelUsage(input_tokens=1000, output_tokens=100), sample_id=f"{name}-{i}", epoch=1)
"""


def test_two_processes(ledger: Path, tmp_path: Path) -> None:
    count, go = 60, tmp_path / "go"
    children = []
    for name in ("a", "b"):
        ready = tmp_path / f"ready-{name}"
        children.append((ready, subprocess.Popen(
            [sys.executable, "-c", CHILD, name, str(ready), str(go), str(count)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )))
    deadline = time.monotonic() + 120
    while not all(ready.exists() for ready, _ in children):
        assert time.monotonic() < deadline, "children did not start"
        assert all(child.poll() is None for _, child in children), [c.communicate() for _, c in children]
        time.sleep(0.01)
    go.touch()
    for _, child in children:
        _, stderr = child.communicate(timeout=120)
        assert child.returncode == 0, stderr
    entries = spend.ledger_entries()
    assert sorted(str(e.sample_id) for e in entries) == sorted(f"{n}-{i}" for n in "ab" for i in range(count))
    # 1000 x $0.10/MTok + 100 x $0.50/MTok = $0.00015 per call.
    assert spend.spent(entries)["monitor"] == pytest.approx(2 * count * 0.00015, rel=1e-12)


LOCK_HOLDER = """
import sys, time
from pathlib import Path
from rewardhack.monitors import spend
ready, go = Path(sys.argv[1]), Path(sys.argv[2])
spend.hold_run_lock()
spend.hold_run_lock()  # again in the same process: a no-op
ready.touch()
while not go.exists():
    time.sleep(0.005)
"""


def test_scoring_runs_go_one_at_a_time(ledger: Path, tmp_path: Path) -> None:
    """A second process cannot take the run lock while the first holds it, and can once the first exits."""
    ready, go = tmp_path / "ready", tmp_path / "go"
    holder = subprocess.Popen([sys.executable, "-c", LOCK_HOLDER, str(ready), str(go)], stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 120
        while not ready.exists():
            assert holder.poll() is None, holder.communicate()
            assert time.monotonic() < deadline, "the lock holder did not start"
            time.sleep(0.01)
        taker = subprocess.run([sys.executable, "-c", "from rewardhack.monitors import spend; spend.hold_run_lock()"], capture_output=True, text=True, timeout=120)
        assert taker.returncode != 0 and "RunLocked" in taker.stderr and "one at a time" in taker.stderr
    finally:
        go.touch()
        _, stderr = holder.communicate(timeout=120)
    assert holder.returncode == 0, stderr
    assert spend.hold_run_lock() == spend.run_lock_path() == ledger.with_name("spend.json.run.lock")
    assert spend.ledger_entries() == []


def test_the_run_lock_needs_a_ledger(isolated_ledger: Path) -> None:
    with pytest.raises(LedgerError, match="no ledger"):
        spend.hold_run_lock()
    assert not isolated_ledger.parent.exists()


def test_missing_ledger_fails_closed(isolated_ledger: Path) -> None:
    with pytest.raises(LedgerError, match="no ledger"):
        record(output_tokens=1)
    with pytest.raises(LedgerError, match="no ledger"):
        spend.check_budget("m2", [Planned("monitor", HAIKU, 1)])
    with pytest.raises(LedgerError, match="no ledger"):
        spend.ledger_entries()
    assert not isolated_ledger.exists()
    # A missing folder fails the same way, before the lock file can be made.
    with pytest.raises(LedgerError, match="no ledger"):
        spend.ledger_entries(isolated_ledger.parent / "gone" / "spend.json")


def test_reading_a_missing_ledger_writes_nothing(tmp_path: Path) -> None:
    # A dry run reads the ledger; with none there yet it must not leave a lock file behind.
    ledger = tmp_path / "spend.json"
    with pytest.raises(LedgerError, match="no ledger"):
        spend.ledger_entries(ledger)
    assert list(tmp_path.iterdir()) == []


def test_an_unsearchable_ledger_folder_is_a_ledger_error(tmp_path: Path) -> None:
    folder = tmp_path / "locked"
    folder.mkdir()
    folder.chmod(0o600)
    try:
        with pytest.raises(LedgerError, match="cannot"):
            spend.ledger_entries(folder / "spend.json")
    finally:
        folder.chmod(0o700)


def valid_entry() -> dict[str, object]:
    usage = ModelUsage(output_tokens=1)
    return spend.CallSpend(
        scorer="m2", role="monitor", cap="monitor", model=HAIKU, sample_id="s", epoch=1, log=None, attempt=1,
        input_tokens=0, output_tokens=1, input_tokens_cache_write=0, input_tokens_cache_read=0, reasoning_tokens=0,
        cost=spend.call_cost(HAIKU, usage), inspect_cost=None, usage_missing=False, timestamp="2000-01-01T00:00:00+00:00",
    ).to_dict()


def ledger_with(**changes: object) -> str:
    return json.dumps({"version": 1, "entries": [{**valid_entry(), **changes}]})


def ledger_without(field: str) -> str:
    entry = valid_entry()
    del entry[field]
    return json.dumps({"version": 1, "entries": [entry]})


CORRUPT: dict[str, Callable[[], bytes]] = {
    "empty": lambda: b"",
    "truncated": lambda: ledger_with().encode()[:-5],
    "null": lambda: b"null",
    "list": lambda: b"[]",
    "invalid-utf8": lambda: b"\xff\xfe{}",
    "version-2": lambda: b'{"version": 2, "entries": []}',
    "version-true": lambda: b'{"version": true, "entries": []}',
    "entries-object": lambda: b'{"version": 1, "entries": {}}',
    "extra-key": lambda: b'{"version": 1, "entries": [], "total": 0}',
    "missing-field": lambda: ledger_without("cost").encode(),
    "negative-cost": lambda: ledger_with(cost=-1.0).encode(),
    "nan-cost": lambda: ledger_with(cost=math.nan).encode(),
    "string-cost": lambda: ledger_with(cost="0.1").encode(),
    "unknown-cap": lambda: ledger_with(cap="agent").encode(),
    "role-cap-mismatch": lambda: ledger_with(role="judge").encode(),
    "negative-tokens": lambda: ledger_with(output_tokens=-1).encode(),
    "float-tokens": lambda: ledger_with(output_tokens=1.5).encode(),
    "bool-epoch": lambda: ledger_with(epoch=True).encode(),
    "zero-attempt": lambda: ledger_with(attempt=0).encode(),
    "missing-attempt": lambda: ledger_without("attempt").encode(),
}


@pytest.mark.parametrize("make", CORRUPT.values(), ids=CORRUPT.keys())
def test_corrupt_ledger_fails_closed(isolated_ledger: Path, make: Callable[[], bytes]) -> None:
    isolated_ledger.parent.mkdir(parents=True)
    isolated_ledger.write_bytes(make())
    before = isolated_ledger.read_bytes()
    with pytest.raises(LedgerError, match="corrupt|cannot read"):
        record(output_tokens=1)
    with pytest.raises(LedgerError, match="corrupt|cannot read"):
        spend.check_budget("m2", [Planned("monitor", HAIKU, 1)])
    with pytest.raises(FileExistsError):
        spend.init_ledger()
    assert isolated_ledger.read_bytes() == before


def test_valid_entry_helper_parses(isolated_ledger: Path) -> None:
    # The corrupt cases above differ from this valid ledger by one change each.
    isolated_ledger.parent.mkdir(parents=True)
    isolated_ledger.write_text(ledger_with())
    assert [e.to_dict() for e in spend.ledger_entries()] == [valid_entry()]


def test_unreadable_ledger_fails_closed(isolated_ledger: Path) -> None:
    isolated_ledger.mkdir(parents=True)
    with pytest.raises(LedgerError, match="cannot read"):
        record(output_tokens=1)
    with pytest.raises(LedgerError, match="cannot read"):
        spend.check_budget("m2", [Planned("monitor", HAIKU, 1)])
    with pytest.raises(FileExistsError):
        spend.init_ledger()


def test_init_never_resets(ledger: Path) -> None:
    record(output_tokens=1)
    with pytest.raises(FileExistsError, match="never reset"):
        spend.init_ledger()
    assert len(spend.ledger_entries()) == 1


def test_spend_metadata(ledger: Path) -> None:
    summary = record("m3_summary", role="summarizer", input_tokens=300, output_tokens=40)
    monitor = record("m3_summary", role="monitor", input_tokens=2000, output_tokens=60, input_tokens_cache_read=5)
    metadata = spend.spend_metadata([summary, monitor])
    assert metadata["calls"] == 2 and metadata["usage_missing"] == 0
    assert (metadata["input_tokens"], metadata["output_tokens"], metadata["input_tokens_cache_read"]) == (2300, 100, 5)
    assert metadata["cost"] == pytest.approx(summary.cost + monitor.cost)
    assert set(metadata["by_role"]) == {"monitor", "summarizer"}
    assert metadata["by_role"]["summarizer"]["cost"] == summary.cost
    assert metadata["by_role"]["monitor"]["input_tokens"] == 2000
    assert json.loads(json.dumps(metadata)) == metadata
    empty = spend.spend_metadata([])
    assert (empty["calls"], empty["cost"], empty["by_role"]) == (0, 0.0, {})


@pytest.fixture
def inspect_registry(monkeypatch: pytest.MonkeyPatch):
    """Undo register_costs: restore Inspect's custom model registry and drop its memoized lookups."""
    monkeypatch.setattr(inspect_model_info, "_custom_models", dict(inspect_model_info._custom_models))
    yield
    inspect_model_info._result_cache.clear()


def test_register_costs_sets_inspect_prices(inspect_registry: None) -> None:
    spend.register_costs(HAIKU, get_model("mockllm/model", memoize=False), "claude-sonnet-5-5")
    listed = yaml.safe_load(spend.COSTS_FILE.read_text())
    for model in (HAIKU, SONNET):
        info = get_model_info(model)
        assert info is not None and info.cost == ModelCost(**listed[model])
        assert compute_model_cost(info.cost, ModelUsage(**HAIKU_USAGE)) == spend.call_cost(model, ModelUsage(**HAIKU_USAGE))
    mock = get_model_info("mockllm/model")
    assert mock is not None and mock.cost == spend.ZERO_COST_MODELS["mockllm/model"]


def test_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = str(tmp_path / "cli" / "spend.json")
    assert spend.main(["init", "--ledger", path]) == 0
    assert spend.main(["init", "--ledger", path]) == 2
    assert spend.main(["status", "--ledger", path]) == 0
    assert "monitor: $0.0000 of $12.00 spent in 0 calls, $12.0000 left" in capsys.readouterr().out
    check = ["check", "--ledger", path, "--scorer", "m3_summary", "--part", "monitor", HAIKU, "10"]
    assert spend.main([*check, "--part", "summarizer", HAIKU, "100", "--input-tokens-bound", "5000"]) == 0
    assert "monitor cap $12.00: spent $0.0000" in capsys.readouterr().out
    assert spend.main(["check", "--ledger", path, "--scorer", "m2", "--part", "monitor", HAIKU, "600"]) == 1
    assert "m2 refused before any call" in capsys.readouterr().err
    assert spend.main(["check", "--ledger", path, "--scorer", "m2", "--part", "grader", HAIKU, "1"]) == 2
    Path(path).write_text("{")
    assert spend.main(["status", "--ledger", path]) == 2
    assert "corrupt" in capsys.readouterr().err
