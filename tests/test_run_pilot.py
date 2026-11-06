"""scripts/run_pilot.sh: the pilot's six eval-set commands, its refusals, its lock, how it stops, and its mock override.

No test here can make a paid call.
The dry-run, refusal, lock and stop tests run a copy of the script in a temporary root that has no .env
file (only costs.yaml and a small stand-in harness tree to fingerprint), with a fake uv first on PATH that
records its arguments and runs nothing, so a broken refusal or dry run stops at the missing .env file or
reaches only the fake uv.
The Docker tests need the real root, so they run the real script with the mockllm override, whose
replies end each episode after one turn, and every test's environment blanks the API keys and points
the providers at an unroutable address, so a broken mock check would fail instead of spending.
Each launcher runs in its own session, and one that outlives its timeout is stopped with SIGTERM (which
stops its running part) and then has its whole process group killed.
A real run passes eval-set absolute log directories, so a fake root's are <fake root>/logs/pilot/<agent>-<variant>.
"""

import ast
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from dotenv import dotenv_values
from inspect_ai._util.config import parse_cli_args
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSpec, read_eval_log, write_eval_log

from rewardhack.task import ROOT

SCRIPT = ROOT / "scripts/run_pilot.sh"
COSTS_SHA256 = hashlib.sha256((ROOT / "costs.yaml").read_bytes()).hexdigest()
PREFIX = ["uv", "run", "--no-env-file", "inspect", "eval-set", "src/rewardhack/task.py"]
PREFLIGHT = ["run", "--no-env-file", "python", "-"]
EVAL_SET = ["run", "--no-env-file", "inspect", "eval-set"]
SAMPLES = ["078_0", "327_0"]
RUNS = [(agent, variant) for agent in ("haiku", "gpt-oss", "sonnet") for variant in ("normal", "conflict")]
# PLAN's "Harness" section: model, epochs (Build order step 3) and per-sample cost limit per agent.
AGENTS = {
    "haiku": ("anthropic/claude-haiku-5-5", 5, "0.10"),
    "gpt-oss": ("openrouter/openai/gpt-oss-120b", 5, "0.05"),
    "sonnet": ("anthropic/claude-sonnet-5-5", 3, "1.00"),
}
EVERY_RUN = {
    "--turn-limit": "30",
    "--token-limit": "output:100k",
    "--time-limit": "900",
    "--max-sandboxes": "12",
    "--max-connections": "12",
    "--retry-on-error": "2",
    "--fail-on-error": "0.05",
    "--retry-attempts": "2",
    "--model-cost-config": "costs.yaml",
}
# Flags that take no value; --log-model-api keeps every raw model call, which the pilot report's
# provider check needs.
BOOLEAN_FLAGS = ["--log-model-api"]
GPT_OSS_PROVIDER = (
    'provider={"order":["deepinfra/bf16"],"quantizations":["bf16"],"allow_fallbacks":false,"require_parameters":true}'
)
# The model_args the live gpt-oss smoke logs recorded.
GPT_OSS_MODEL_ARGS = {
    "provider": {"order": ["deepinfra/bf16"], "quantizations": ["bf16"], "allow_fallbacks": False, "require_parameters": True},
}
FORBIDDEN = ["--fallback-models", "--effort", "--reasoning-tokens", "eval-retry", "--log-dir-allow-dirty", "--limit"]
MOCK_TIMEOUT = 600
# The launcher's own STOP_GRACE (120 s) plus its 30 s wait after a SIGINT, with a margin.
STOP_TIMEOUT = 180
LOCK = ".run_pilot.lock"
# Blank API keys and unroutable provider addresses, so a test run that a regression made real fails
# instead of spending: python-dotenv does not override a variable that is set, even to "", unless
# Inspect thinks it runs inside VS Code, so clean_env removes the variables that tell it so.
NO_SPEND = {
    "ANTHROPIC_API_KEY": "",
    "OPENROUTER_API_KEY": "",
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:9",
    "OPENROUTER_BASE_URL": "http://127.0.0.1:9",
}

FAKE_UV = """\
#!{python}
import json
import os
import shutil
import subprocess
import sys
import time

args = sys.argv[1:]
log = os.environ["FAKE_UV_LOG"]
with open(log, "a") as f:
    f.write(json.dumps(args) + "\\n")
# FAKE_UV_EDIT names a file to append to, or a directory to remove, during the FAKE_UV_EDIT_AT-th uv call
# (1 is the preflight).
edit = os.environ.get("FAKE_UV_EDIT")
if edit:
    with open(log) as f:
        if sum(1 for _ in f) == int(os.environ["FAKE_UV_EDIT_AT"]):
            if os.path.isdir(edit):
                shutil.rmtree(edit)
            else:
                with open(edit, "a") as g:
                    g.write("# edited while uv ran\\n")
if args[:4] == ["run", "--no-env-file", "python", "-"]:
    script = sys.stdin.read()
    if os.environ.get("FAKE_UV_RUN_PREFLIGHT"):
        # FAKE_UV_PREFLIGHT_ENV stands for variables that uv itself would add to the command's environment.
        env = {{**os.environ, **json.loads(os.environ.get("FAKE_UV_PREFLIGHT_ENV", "{{}}"))}}
        sys.exit(subprocess.run([sys.executable, "-", *args[4:]], input=script, text=True, env=env).returncode)
    sys.exit(int(os.environ.get("FAKE_UV_PREFLIGHT_EXIT", "0")))
if args[:4] != ["run", "--no-env-file", "inspect", "eval-set"]:
    sys.exit(99)
with open(log + ".pids", "a") as f:
    f.write(f"{{os.getpid()}}\\n")
release = os.environ.get("FAKE_UV_BLOCK")
if release:
    try:
        deadline = time.monotonic() + 60
        while not os.path.exists(release):
            if time.monotonic() > deadline:
                sys.exit(3)
            time.sleep(0.05)
    except KeyboardInterrupt:
        with open(log + ".interrupted", "a") as f:
            f.write(f"{{os.getpid()}}\\n")
        sys.exit(130)
with open(log) as f:
    count = sum(json.loads(line)[:4] == ["run", "--no-env-file", "inspect", "eval-set"] for line in f)
sys.exit(1 if count == int(os.environ.get("FAKE_UV_FAIL_AT", "0")) else 0)
"""

FAKE_DOCKER = """\
#!/bin/sh
if [ "$1" = info ]; then exit "${FAKE_DOCKER_INFO_EXIT:-0}"; fi
if [ "$1" = image ]; then echo sha256:not-the-recorded-image; exit 0; fi
exit 1
"""


def clean_env(**extra: str) -> dict[str, str]:
    """The test's environment without the variables the script or dotenv reads, with NO_SPEND, plus extra."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("INSPECT_EVAL_", "PILOT_", "FAKE_", "VSCODE_"))
        and "DRY" not in key.upper()
        and key not in ("AGENTS", "TERM_PROGRAM", "UV_ENV_FILE", "UV_NO_ENV_FILE")
    }
    return {**env, **NO_SPEND, **extra}


def stop_group(proc: subprocess.Popen) -> None:
    """Stop a launcher as an operator would (SIGTERM stops its running part), then kill what is left of its group."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()


def run_script(env: dict[str, str], script: Path, cwd: Path | None = None, timeout: int = 60):
    proc = subprocess.Popen(
        [str(script)], cwd=cwd or script.parents[1], env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop_group(proc)
        raise
    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)


def wait_for(condition: Callable[[], object], timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.05)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def processes_mentioning(text: str) -> list[str]:
    """ps lines whose command line contains text (filtered here, so the search never matches itself)."""
    out = subprocess.run(["ps", "-ax", "-o", "pid=,command="], capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if text in line]


@pytest.fixture
def background(tmp_path):
    """Start launchers in their own sessions, with stderr in a file; teardown stops any still running."""
    started: list[tuple[subprocess.Popen, object]] = []

    def start(env: dict[str, str], script: Path, cwd: Path | None = None, name: str = "launcher") -> subprocess.Popen:
        err_path = tmp_path / f"{name}.stderr"
        err = err_path.open("w")
        proc = subprocess.Popen(
            [str(script)], cwd=cwd or script.parents[1], env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=err, start_new_session=True,
        )
        proc.err_path = err_path
        started.append((proc, err))
        return proc

    yield start
    for proc, err in started:
        stop_group(proc)
        err.close()


# run_pilot.sh's HARNESS_PATHS: the files whose sha256 lines make harness_sha256, except those under NOT_HARNESS.
HARNESS_PATHS = ("costs.yaml", "uv.lock", "sandbox", "src/rewardhack", *(f"tasks/{task}" for task in SAMPLES))
# The monitor and analysis packages read finished logs; no module an episode loads imports them.
NOT_HARNESS = (("src", "rewardhack", "monitors"), ("src", "rewardhack", "analysis"))
# The fake root's stand-in for each harness path that is not costs.yaml.
FAKE_HARNESS = {
    "uv.lock": "version = 1\n",
    "sandbox/image_digest.txt": "sha256:fake\n",
    "src/rewardhack/task.py": "# the fake root's harness\n",
    "tasks/078_0/task.json": "{}\n",
    "tasks/327_0/task.json": "{}\n",
}


def harness_sha256(root: Path) -> str:
    """run_pilot.sh's harness fingerprint, restated: the sha256 of the sorted "sha256  path" lines of every file
    under HARNESS_PATHS, leaving out symlinks, bytecode caches, NOT_HARNESS and any path with a component starting
    with a dot."""
    files = []
    for name in HARNESS_PATHS:
        top = root / name
        for path in [top] if top.is_file() else top.rglob("*"):
            parts = path.relative_to(root).parts
            if (
                path.is_file() and not path.is_symlink() and "__pycache__" not in parts
                and not any(p.startswith(".") for p in parts) and parts[:3] not in NOT_HARNESS
            ):
                files.append(path.relative_to(root).as_posix())
    lines = "".join(f"{hashlib.sha256((root / rel).read_bytes()).hexdigest()}  {rel}\n" for rel in sorted(files))
    return hashlib.sha256(lines.encode()).hexdigest()


def make_fake_root(base: Path) -> dict:
    """A copy of the script and costs.yaml in base/repo, with FAKE_HARNESS, no .env file and a fake uv first on PATH."""
    root = (base / "repo").resolve()
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, root / "scripts/run_pilot.sh")
    shutil.copy2(ROOT / "costs.yaml", root / "costs.yaml")
    for name, text in FAKE_HARNESS.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    bin_dir = base / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(FAKE_UV.format(python=sys.executable))
    uv.chmod(0o755)
    log = base / "uv_calls.jsonl"
    env = clean_env(PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", FAKE_UV_LOG=str(log))
    assert shutil.which("uv", path=env["PATH"]) == str(uv)
    return {
        "root": root, "script": root / "scripts/run_pilot.sh", "env": env, "log": log, "bin": bin_dir,
        "harness": harness_sha256(root), "pilot": f"{root}/logs/pilot",
    }


@pytest.fixture
def fake_root(tmp_path: Path) -> dict:
    return make_fake_root(tmp_path)


def uv_calls(fake: dict) -> list[list[str]]:
    if not fake["log"].exists():
        return []
    return [json.loads(line) for line in fake["log"].read_text().splitlines()]


def fake_uv_pids(fake: dict, kind: str = "pids") -> list[int]:
    path = Path(f"{fake['log']}.{kind}")
    return [int(line) for line in path.read_text().splitlines()] if path.exists() else []


def dry_run(fake: dict, cwd: Path | None = None, **extra: str) -> list[list[str]]:
    before = uv_calls(fake)
    result = run_script({**fake["env"], "DRY_RUN": "1", **extra}, script=fake["script"], cwd=cwd)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert uv_calls(fake) == before
    return [shlex.split(line) for line in result.stdout.splitlines()]


def options(argv: list[str]) -> dict[str, list]:
    """A command's options after the task file, each flag with the values it was given."""
    assert argv[: len(PREFIX)] == PREFIX
    rest = argv[len(PREFIX) :]
    parsed: dict[str, list] = {flag: [True] * rest.count(flag) for flag in BOOLEAN_FLAGS}
    rest = [arg for arg in rest if arg not in BOOLEAN_FLAGS]
    flags, values = rest[::2], rest[1::2]
    assert len(flags) == len(values), argv
    assert all(flag.startswith("-") for flag in flags), argv
    assert not any(value.startswith("--") for value in values), argv
    for flag, value in zip(flags, values):
        parsed.setdefault(flag, []).append(value)
    return parsed


def expected(agent: str, variant: str, harness: str, log_root: str, mock_epochs: int | None = None) -> dict:
    model, epochs, cost_limit = AGENTS[agent]
    opts = {
        "-T": [f"variant={variant}"],
        "--sample-id": [",".join(SAMPLES)],
        "--epochs": [str(epochs if mock_epochs is None else mock_epochs)],
        "--model": [model if mock_epochs is None else "mockllm/model"],
        "--reasoning-effort": ["medium"],
        "--cost-limit": [cost_limit],
        **{flag: [value] for flag, value in EVERY_RUN.items()},
        "--log-model-api": [True],
        "--metadata": ["model_cost_config=costs.yaml", f"model_cost_sha256={COSTS_SHA256}", f"harness_sha256={harness}"],
        "--log-dir": [f"{log_root}/{agent}-{variant}"],
    }
    if agent == "gpt-oss" and mock_epochs is None:
        opts["-M"] = [GPT_OSS_PROVIDER]
    return opts


def test_dry_run_prints_the_six_pilot_commands(fake_root):
    commands = dry_run(fake_root)
    assert len(commands) == len(RUNS) == 6
    for argv, (agent, variant) in zip(commands, RUNS):
        assert options(argv) == expected(agent, variant, fake_root["harness"], fake_root["pilot"]), (agent, variant)
    # Two samples per epoch: Haiku x5, gpt-oss x5 and Sonnet x3 epochs over both variants.
    episodes = sum(len(SAMPLES) * int(options(argv)["--epochs"][0]) for argv in commands)
    assert episodes == 52


def test_dry_run_has_no_forbidden_flag(fake_root):
    for argv in dry_run(fake_root):
        for token in FORBIDDEN:
            assert token not in argv, (token, argv)
        assert not any("eval-retry" in arg for arg in argv)
        assert options(argv)["--reasoning-effort"] == ["medium"]


def test_gpt_oss_provider_parses_to_the_smoke_runs_model_args(fake_root):
    commands = dict(zip(RUNS, dry_run(fake_root)))
    for variant in ("normal", "conflict"):
        [provider] = options(commands["gpt-oss", variant])["-M"]
        assert parse_cli_args([provider]) == GPT_OSS_MODEL_ARGS
    for agent in ("haiku", "sonnet"):
        assert "-M" not in options(commands[agent, "normal"])


def test_dry_run_is_the_same_from_any_directory(fake_root, tmp_path):
    assert dry_run(fake_root, cwd=tmp_path) == dry_run(fake_root)


@pytest.mark.parametrize(
    "agents, runs",
    [
        ("sonnet", [("sonnet", "normal"), ("sonnet", "conflict")]),
        ("sonnet  haiku", [run for run in RUNS if run[0] != "gpt-oss"]),
        ("gpt-oss\thaiku gpt-oss", [run for run in RUNS if run[0] != "sonnet"]),
    ],
    ids=["one", "reordered", "tab_and_duplicate"],
)
def test_agents_filter_keeps_plan_order(fake_root, agents, runs):
    commands = dry_run(fake_root, AGENTS=agents)
    assert [options(argv) for argv in commands] == [expected(*run, fake_root["harness"], fake_root["pilot"]) for run in runs]


def test_mock_override_swaps_every_model_and_log_dir(fake_root, tmp_path):
    root = str(tmp_path / "mock")
    commands = dry_run(fake_root, PILOT_MODEL_OVERRIDE="mockllm/model", PILOT_EPOCHS_OVERRIDE="1", PILOT_LOG_ROOT=root)
    assert len(commands) == 6
    for argv, (agent, variant) in zip(commands, RUNS):
        assert options(argv) == expected(agent, variant, fake_root["harness"], root, mock_epochs=1)


def mock(model: str = "mockllm/model", epochs: str = "1", root: str = "/tmp/m") -> dict[str, str]:
    return {"PILOT_MODEL_OVERRIDE": model, "PILOT_EPOCHS_OVERRIDE": epochs, "PILOT_LOG_ROOT": root}


MODEL_MUST = "PILOT_MODEL_OVERRIDE must be mockllm/model"
EPOCHS_MUST = "PILOT_EPOCHS_OVERRIDE must be a positive integer"
ROOT_ABSOLUTE = "PILOT_LOG_ROOT must be an absolute directory"
ROOT_DOTS = "PILOT_LOG_ROOT must not contain . or .. components"
ROOT_OUTSIDE = "PILOT_LOG_ROOT must be outside this repository"
UNREAD = "reads only DRY_RUN, AGENTS, PILOT_MODEL_OVERRIDE, PILOT_EPOCHS_OVERRIDE and PILOT_LOG_ROOT; unset them first: "
# {root} is the fake root, {upper} the same path in upper case, and {link} a symlink to it.
REFUSALS = [
    ({"DRY_RUN": "yes"}, "DRY_RUN must be 1, 0 or unset"),
    ({"DRY_RUN": ""}, "DRY_RUN must be 1, 0 or unset"),
    ({"AGENTS": "claude"}, "unknown agent 'claude'"),
    ({"AGENTS": ""}, "AGENTS selects no agent"),
    ({"AGENTS": "haiku,sonnet"}, "unknown agent 'haiku,sonnet'"),
    ({"INSPECT_EVAL_FALLBACK_MODELS": "anthropic/claude-sonnet-5-5"}, "INSPECT_EVAL_FALLBACK_MODELS"),
    ({"INSPECT_EVAL_REASONING_TOKENS": "1000"}, "INSPECT_EVAL_REASONING_TOKENS"),
    (mock(model="anthropic/claude-haiku-5-5"), MODEL_MUST),
    (mock(model="mockllm/other"), MODEL_MUST),
    ({"PILOT_MODEL_OVERRIDE": "mockllm/model"}, EPOCHS_MUST),
    ({"PILOT_MODEL_OVERRIDE": "mockllm/model", "PILOT_EPOCHS_OVERRIDE": "1"}, ROOT_ABSOLUTE),
    ({"PILOT_MODEL_OVERRIDE": "mockllm/model", "PILOT_LOG_ROOT": "/tmp/m"}, EPOCHS_MUST),
    ({"PILOT_EPOCHS_OVERRIDE": "1"}, MODEL_MUST),
    ({"PILOT_LOG_ROOT": "/tmp/m"}, MODEL_MUST),
    ({"PILOT_EPOCHS_OVERRIDE": "1", "PILOT_LOG_ROOT": "/tmp/m"}, MODEL_MUST),
    # A PILOT_* variable set to an empty value still means a mock run, so it is refused instead of
    # becoming the paid pilot.
    (mock(model="", epochs="", root=""), MODEL_MUST),
    ({"PILOT_MODEL_OVERRIDE": ""}, MODEL_MUST),
    ({"PILOT_LOG_ROOT": ""}, MODEL_MUST),
    # A misspelled control variable would be ignored, leaving a paid run, so any PILOT_* or *DRY* name
    # (ignoring case) that the script does not read is refused, beside a valid setting or not.
    *(
        ({name: value}, UNREAD + name)
        for name, value in (
            ("PILOT_MODEL", "mockllm/model"), ("PILOT_EPOCHS", "1"), ("PILOT_LOGROOT", "/tmp/m"),
            ("pilot_model_override", "mockllm/model"), ("Pilot_Log_Root", "/tmp/m"), ("PILOT_", ""),
            ("DRYRUN", "1"), ("DRY", "1"), ("dry_run", "1"), ("DRY_RUN_", "1"), ("PILOT_DRY_RUN", "1"),
        )
    ),
    ({**mock(), "PILOT_EPOCH_OVERRIDE": "2"}, UNREAD + "PILOT_EPOCH_OVERRIDE"),
    *((mock(epochs=epochs), EPOCHS_MUST) for epochs in ("0", "two", "1.5", "-1", "01", "")),
    *((mock(root=root), ROOT_ABSOLUTE) for root in ("", "logs/mock", "./logs/mock")),
    *((mock(root=root), ROOT_DOTS) for root in ("/tmp/x/../m", "{root}/logs/./pilot", "{root}/..")),
    *(
        (mock(root=root), ROOT_OUTSIDE)
        for root in (
            "{root}",
            "{root}/",
            "{root}/logs/pilot",
            "{root}/logs/pilot/",
            "{root}/logs/pilot/mock",
            "{root}/logs/main/mock",
            "{root}/logs/mock",
            "{root}/logs/scored",
            "{root}/LOGS/pilot",
            "{root}//logs/pilot",
            "{root}/logs/Pilot",
            "{upper}/logs/mock",
            "{link}/logs/mock",
        )
    ),
]


@pytest.mark.parametrize(("env", "message"), REFUSALS)
def test_refusals_print_and_run_nothing(fake_root, tmp_path, env, message):
    (tmp_path / "link").symlink_to(fake_root["root"])
    places = {"root": str(fake_root["root"]), "upper": str(fake_root["root"]).upper(), "link": str(tmp_path / "link")}
    env = {key: value.format(**places) for key, value in env.items()}
    result = run_script({**fake_root["env"], "DRY_RUN": "1", **env}, script=fake_root["script"])
    assert result.returncode == 2, result.stdout
    assert result.stdout == ""
    assert result.stderr.startswith("run_pilot.sh: ")
    assert message in result.stderr
    assert uv_calls(fake_root) == []
    assert not (fake_root["root"] / "logs").exists()


def preflight_call(fake: dict, runs: list[tuple[str, str]] = RUNS) -> list[str]:
    """The preflight's uv arguments: the seeds, the harness fingerprint and each selected part's log directory."""
    return [*PREFLIGHT, ",".join(SAMPLES), fake["harness"], *(f"{fake['pilot']}/{agent}-{variant}" for agent, variant in runs)]


def test_real_run_refuses_without_env_file(fake_root):
    result = run_script(fake_root["env"], script=fake_root["script"])
    assert result.returncode == 2
    assert ".env" in result.stderr
    assert uv_calls(fake_root) == []


@pytest.mark.parametrize("name, value", [("DRYRUN", "1"), ("PILOT_MODEL", "mockllm/model")])
def test_real_run_refuses_a_misspelled_control_variable(fake_root, name, value):
    """With a .env file present, a misspelled dry run or mock override is refused instead of running every part."""
    (fake_root["root"] / ".env").touch()
    result = run_script({**fake_root["env"], name: value}, script=fake_root["script"])
    assert result.returncode == 2, result.stderr
    assert UNREAD + name in result.stderr
    assert uv_calls(fake_root) == []
    assert not (fake_root["root"] / "logs").exists()


def test_real_run_runs_the_dry_run_commands_in_order(fake_root, tmp_path):
    (fake_root["root"] / ".env").touch()
    result = run_script(fake_root["env"], cwd=tmp_path, script=fake_root["script"])
    assert result.returncode == 0, result.stderr
    eval_sets = [["uv", *call] for call in uv_calls(fake_root)[1:]]
    assert uv_calls(fake_root)[0] == preflight_call(fake_root)
    assert eval_sets == dry_run(fake_root)
    # Haiku 20 x $0.10 and Sonnet 12 x $1.00 make PLAN's $14.00 pilot Claude cap; gpt-oss is 20 x $0.05.
    assert (
        "the selected parts' samples times their cost limits come to $14.00 on Anthropic (PLAN's hard cap for the "
        "pilot's Claude part is $14.00) and $1.00 on OpenRouter (counted against the $15.50 gpt-oss cap)"
    ) in result.stderr
    assert "starting sonnet conflict (6 samples, cost limit $1.00 each):" in result.stderr
    assert "every selected part finished" in result.stderr
    assert not (fake_root["root"] / "logs/pilot" / LOCK).exists()


def test_real_run_stops_at_the_first_failed_part(fake_root):
    (fake_root["root"] / ".env").touch()
    result = run_script({**fake_root["env"], "FAKE_UV_FAIL_AT": "3"}, script=fake_root["script"])
    assert result.returncode == 1
    assert "gpt-oss normal did not finish" in result.stderr
    assert 'AGENTS="gpt-oss sonnet"' in result.stderr
    calls = uv_calls(fake_root)
    assert calls[0] == preflight_call(fake_root)
    assert [["uv", *call] for call in calls[1:]] == dry_run(fake_root)[:3]
    assert not (fake_root["root"] / "logs/pilot" / LOCK).exists()


def test_failed_preflight_runs_nothing(fake_root):
    (fake_root["root"] / ".env").touch()
    result = run_script({**fake_root["env"], "FAKE_UV_PREFLIGHT_EXIT": "1"}, script=fake_root["script"])
    assert result.returncode == 2
    assert "nothing was run" in result.stderr
    assert uv_calls(fake_root) == [preflight_call(fake_root)]
    assert not (fake_root["root"] / "logs/pilot" / LOCK).exists()


def test_preflight_refuses_inspect_options_in_the_env_file(fake_root):
    """The real preflight, run by the fake uv in the fake root, scans its .env file for key names and prints only those."""
    dotenv_file = fake_root["root"] / ".env"
    dotenv_file.write_text(
        "INSPECT_EVAL_SET_LIMIT=1\nANTHROPIC_API_KEY=not-a-real-key\nINSPECT_EVAL_LIMIT=2\n"
        "  export INSPECT_EVAL_EPOCHS = 3\n'INSPECT_EVAL_MAX_TOKENS'=4\n"
        "# INSPECT_EVAL_COMMENTED=5\nINSPECT_EVAL_NO_VALUE\nOTHER_INSPECT_EVAL_X=6\n"
    )
    # The key names python-dotenv, which Inspect loads the file with, would set.
    keys = sorted(key for key, value in dotenv_values(dotenv_file).items() if key.startswith("INSPECT_EVAL_") and value is not None)
    assert keys == ["INSPECT_EVAL_EPOCHS", "INSPECT_EVAL_LIMIT", "INSPECT_EVAL_MAX_TOKENS", "INSPECT_EVAL_SET_LIMIT"]
    result = run_script({**fake_root["env"], "FAKE_UV_RUN_PREFLIGHT": "1"}, script=fake_root["script"])
    assert result.returncode == 2, result.stderr
    assert f"remove these Inspect option variables from {dotenv_file} first: {' '.join(keys)}\n" in result.stderr
    assert "not-a-real-key" not in result.stderr and "=1" not in result.stderr
    assert "nothing was run" in result.stderr
    assert uv_calls(fake_root) == [preflight_call(fake_root)]


def test_preflight_refuses_inspect_options_that_uv_adds(fake_root):
    """The preflight runs under the same uv options as every eval-set, so it refuses what uv would add to them."""
    (fake_root["root"] / ".env").touch()
    added = json.dumps({"INSPECT_EVAL_FALLBACK_MODELS": "anthropic/claude-sonnet-5-5"})
    env = {**fake_root["env"], "FAKE_UV_RUN_PREFLIGHT": "1", "FAKE_UV_PREFLIGHT_ENV": added}
    result = run_script(env, script=fake_root["script"])
    assert result.returncode == 2, result.stderr
    assert "unset these Inspect option variables first: INSPECT_EVAL_FALLBACK_MODELS" in result.stderr
    assert "nothing was run" in result.stderr
    assert uv_calls(fake_root) == [preflight_call(fake_root)]


# The harness fingerprint and re-piloting after a fix.


def recorded_harness(fake: dict) -> str:
    """The harness_sha256 that every dry-run command records."""
    [value] = {value for argv in dry_run(fake) for value in options(argv)["--metadata"] if value.startswith("harness_sha256=")}
    return value.removeprefix("harness_sha256=")


def test_harness_sha256_covers_every_harness_file_but_caches_and_dotfiles(fake_root):
    root = fake_root["root"]
    first = recorded_harness(fake_root)
    assert first == fake_root["harness"]
    for name in (
        "src/rewardhack/__pycache__/task.cpython-312.pyc", "sandbox/.DS_Store", "tasks/078_0/.cache/x",
        "tasks/022_0/task.json", "README.md", "scripts/other.py",
    ):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("not harness\n")
    assert recorded_harness(fake_root) == first
    seen = {first}
    for name in ("costs.yaml", "uv.lock", "sandbox/image_digest.txt", "src/rewardhack/task.py", "tasks/078_0/task.json", "tasks/327_0/src/new.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((path.read_text() if path.exists() else "") + "# changed\n")
        changed = recorded_harness(fake_root)
        assert changed == harness_sha256(root) and changed not in seen, name
        seen.add(changed)


def test_the_monitor_and_analysis_packages_are_not_harness(fake_root):
    """They read finished logs, so work on them does not change harness_sha256 and block resuming a part."""
    root = fake_root["root"]
    first = recorded_harness(fake_root)
    for name in ("src/rewardhack/monitors/reasoning.py", "src/rewardhack/monitors/__init__.py", "src/rewardhack/analysis/metrics.py"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("# monitor work\n")
        assert recorded_harness(fake_root) == first, name
        (root / name).write_text("# more monitor work\n")
        assert recorded_harness(fake_root) == first, name
    for name in ("src/rewardhack/task.py", "src/rewardhack/monitors.py", "src/rewardhack/analysis_helpers.py"):
        path = root / name
        path.write_text((path.read_text() if path.exists() else "") + "# changed\n")
        changed = recorded_harness(fake_root)
        assert changed == harness_sha256(root) and changed != first, name
        first = changed


def imported_modules(path: Path) -> set[str]:
    """Every module a Python file imports, relative imports resolved within the rewardhack package."""
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = "rewardhack" + (f".{node.module}" if node.module else "") if node.level else node.module
            modules.update(f"{base}.{alias.name}" for alias in node.names)
            modules.add(base)
    return modules


def test_no_harness_module_imports_the_monitor_or_analysis_packages():
    """What leaving them out of harness_sha256 rests on: an episode loads src/rewardhack/task.py and its imports."""
    harness = [path for path in sorted((ROOT / "src/rewardhack").rglob("*.py")) if path.relative_to(ROOT).parts[:3] not in NOT_HARNESS]
    assert {path.name for path in harness} >= {"__init__.py", "task.py", "tools.py", "grading.py", "containers.py", "label.py"}
    for path in harness:
        for module in imported_modules(path):
            assert not module.startswith(("rewardhack.monitors", "rewardhack.analysis")), (path, module)


def test_a_missing_harness_path_is_refused(fake_root):
    shutil.rmtree(fake_root["root"] / "tasks/327_0")
    result = run_script({**fake_root["env"], "DRY_RUN": "1"}, script=fake_root["script"])
    assert result.returncode == 2 and result.stdout == ""
    assert f"run_pilot.sh: tasks/327_0 is missing from {fake_root['root']}" in result.stderr
    assert uv_calls(fake_root) == []


def write_part_log(path: Path, harness: str | None) -> None:
    """A finished log header like eval-set's, with harness_sha256 (or none) in its eval metadata."""
    spec = EvalSpec(
        eval_id="earlier", created="2026-11-05T00:00:00+00:00", task="false_assurance", dataset=EvalDataset(),
        model="mockllm/model", config=EvalConfig(), metadata=None if harness is None else {"harness_sha256": harness},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(EvalLog(eval=spec, status="success"), str(path))


LOG_NAME = "2026-11-05T00-00-00-00-00_false-assurance_earlier.eval"


def install_fake_docker(bin_dir: Path) -> None:
    (bin_dir / "docker").write_text(FAKE_DOCKER)
    (bin_dir / "docker").chmod(0o755)


@pytest.mark.parametrize("recorded", ["0" * 64, None], ids=["other_harness", "no_harness"])
def test_a_part_logged_under_another_harness_is_refused(fake_root, recorded):
    """eval-set would reuse those samples, since its task identity covers no code, so the preflight refuses first."""
    (fake_root["root"] / ".env").touch()
    pilot = fake_root["root"] / "logs/pilot"
    write_part_log(pilot / "haiku-normal" / LOG_NAME, fake_root["harness"])
    # One stale log is enough, even beside a current one; eval-set lists a log directory recursively, and so
    # does the check.
    write_part_log(pilot / "gpt-oss-conflict" / LOG_NAME, fake_root["harness"])
    write_part_log(pilot / "gpt-oss-conflict/attempt" / LOG_NAME, recorded)
    result = run_script({**fake_root["env"], "FAKE_UV_RUN_PREFLIGHT": "1"}, script=fake_root["script"])
    assert result.returncode == 2, result.stderr
    assert f"the harness is now {fake_root['harness']}, but these log directories hold logs of another harness" in result.stderr
    found = ", ".join(sorted([str(recorded), fake_root["harness"]]))
    assert f"eval-set would reuse: {pilot}/gpt-oss-conflict (harness_sha256 {found});" in result.stderr
    assert "haiku-normal" not in result.stderr
    assert "move each one out of the log root (for example into logs/pilot-rejected/)" in result.stderr
    assert "nothing was run" in result.stderr
    assert uv_calls(fake_root) == [preflight_call(fake_root)]


def test_logs_of_this_harness_and_unselected_parts_pass_the_harness_check(fake_root):
    """The preflight gets past the harness check and stops at the fake docker, whose daemon does not answer."""
    (fake_root["root"] / ".env").touch()
    install_fake_docker(fake_root["bin"])
    pilot = fake_root["root"] / "logs/pilot"
    write_part_log(pilot / "haiku-normal" / LOG_NAME, fake_root["harness"])
    write_part_log(pilot / "gpt-oss-normal" / LOG_NAME, "0" * 64)
    env = {**fake_root["env"], "FAKE_UV_RUN_PREFLIGHT": "1", "FAKE_DOCKER_INFO_EXIT": "1", "AGENTS": "haiku sonnet"}
    result = run_script(env, script=fake_root["script"])
    assert result.returncode == 2, result.stderr
    assert "Docker is unavailable" in result.stderr
    assert "another harness" not in result.stderr
    assert uv_calls(fake_root) == [preflight_call(fake_root, [run for run in RUNS if run[0] != "gpt-oss"])]


# One launcher at a time.


def started_parts(fake: dict, count: int) -> Callable[[], bool]:
    return lambda: sum(call[:4] == EVAL_SET for call in uv_calls(fake)) >= count


def test_a_second_launcher_is_refused_while_one_runs(fake_root, tmp_path, background):
    (fake_root["root"] / ".env").touch()
    release = tmp_path / "release"
    first = background({**fake_root["env"], "FAKE_UV_BLOCK": str(release)}, fake_root["script"])
    wait_for(started_parts(fake_root, 1), 30, "the first launcher's first part")
    calls = uv_calls(fake_root)

    second = run_script(fake_root["env"], script=fake_root["script"])
    assert second.returncode == 2
    assert f"is already running on {fake_root['pilot']}; wait for it to finish, or ask whoever started it" in second.stderr
    assert uv_calls(fake_root) == calls

    release.touch()
    assert first.wait(timeout=60) == 0, first.err_path.read_text()
    assert [["uv", *call] for call in uv_calls(fake_root)[1:]] == dry_run(fake_root)
    assert not (fake_root["root"] / "logs/pilot" / LOCK).exists()


def dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


@pytest.fixture
def sleeper():
    """A live process with the given extra arguments in its command line; killed at teardown."""
    procs = []

    def start(*args: str) -> int:
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", *args])
        procs.append(proc)
        return proc.pid

    yield start
    for proc in procs:
        proc.kill()
        proc.wait()


# The arguments of an eval-set of the pilot task, up to its --log-dir value.
EVAL_SET_ARGS = ("inspect", "eval-set", "src/rewardhack/task.py", "-T", "variant=normal", "--log-dir")
WAIT_OR_ASK = "; wait for it to finish, or ask whoever started it"
REMOVE_BY_HAND = "after checking that no run_pilot.sh or eval-set is running on {pilot}, remove it and re-run"


@pytest.mark.parametrize("case", ["live_launcher", "no_launcher_pid", "orphaned_eval_set", "stale_lock", "eval_set_on_these_logs"])
def test_a_held_lock_or_a_running_eval_set_refuses(fake_root, sleeper, case):
    (fake_root["root"] / ".env").touch()
    pilot = fake_root["pilot"]
    lock = fake_root["root"] / "logs/pilot" / LOCK
    if case != "eval_set_on_these_logs":
        lock.mkdir(parents=True)
    if case == "live_launcher":
        (lock / "launcher").write_text(f"{os.getpid()}\n")
        message = f"another run_pilot.sh (pid {os.getpid()}) is already running on {pilot}{WAIT_OR_ASK}"
    elif case == "no_launcher_pid":
        message = f"{lock} exists without a launcher pid; {REMOVE_BY_HAND.format(pilot=pilot)}"
    elif case == "orphaned_eval_set":
        holder = dead_pid()
        (lock / "launcher").write_text(f"{holder}\n")
        orphan = sleeper()
        (lock / "eval-set").write_text(f"{orphan}\n")
        message = f"the eval-set (pid {orphan}) that an earlier run_pilot.sh (pid {holder}) started is still running on {pilot}{WAIT_OR_ASK}"
    elif case == "stale_lock":
        # Removing a stale lock automatically would let two launchers both take it, so it is left to an operator.
        holder = dead_pid()
        (lock / "launcher").write_text(f"{holder}\n")
        (lock / "eval-set").write_text(f"{dead_pid()}\n")
        message = f"{lock} is the stale lock of run_pilot.sh pid {holder}, which is no longer running; {REMOVE_BY_HAND.format(pilot=pilot)}"
    else:
        running = sleeper(*EVAL_SET_ARGS, f"{pilot}/haiku-normal")
        message = f"an eval-set is already running on {pilot} (pid {running}){WAIT_OR_ASK}"
    result = run_script(fake_root["env"], script=fake_root["script"])
    assert result.returncode == 2, result.stderr
    assert message in result.stderr
    # No refusal tells the operator to stop a run, which could be the paid pilot.
    assert "stop" not in result.stderr
    assert uv_calls(fake_root) == []
    # A refused launcher leaves the lock it found alone.
    assert lock.exists() == (case != "eval_set_on_these_logs")


def test_only_an_eval_set_on_this_roots_logs_refuses(fake_root, tmp_path, sleeper):
    """Eval-sets of other copies of the repository (such as the real pilot, another test session's fake root or a
    reviewer's copy) and the old relative --log-dir form do not count; one on this root's log root does."""
    (fake_root["root"] / ".env").touch()
    pilot = fake_root["pilot"]
    for log_dir in (
        "logs/pilot/haiku-normal",
        f"{tmp_path}/other/repo/logs/pilot/haiku-normal",
        f"{fake_root['root']}-copy/logs/pilot/haiku-normal",
        f"{fake_root['root']}/logs/pilot-rejected/haiku-normal",
        f"/elsewhere{pilot}/haiku-normal",
    ):
        sleeper(*EVAL_SET_ARGS, log_dir)
    result = run_script(fake_root["env"], script=fake_root["script"])
    assert result.returncode == 0, result.stderr
    assert len(uv_calls(fake_root)) == 7

    running = sleeper(*EVAL_SET_ARGS, f"{pilot}/sonnet-conflict")
    result = run_script(fake_root["env"], script=fake_root["script"])
    assert result.returncode == 2, result.stderr
    assert f"an eval-set is already running on {pilot} (pid {running}){WAIT_OR_ASK}" in result.stderr
    assert len(uv_calls(fake_root)) == 7


@pytest.mark.parametrize("edited", ["src/rewardhack/task.py", "costs.yaml", "tasks/327_0"], ids=["task_py", "costs", "seed_removed"])
def test_a_harness_edit_after_launch_stops_the_next_part(fake_root, edited):
    """Each part loads the harness from disk, so an edit after launch (here during the preflight) would make it run
    other code under the launch-time fingerprint: the launcher refuses to start it, and also when a harness path
    is gone and cannot be hashed."""
    (fake_root["root"] / ".env").touch()
    env = {**fake_root["env"], "FAKE_UV_EDIT": str(fake_root["root"] / edited), "FAKE_UV_EDIT_AT": "1"}
    result = run_script(env, script=fake_root["script"])
    assert result.returncode == 2, result.stderr
    assert "changed since launch, so haiku normal was not started" in result.stderr
    assert f"it would record harness_sha256 {fake_root['harness']} while running other code" in result.stderr
    assert 'AGENTS="haiku gpt-oss sonnet"' in result.stderr
    assert uv_calls(fake_root) == [preflight_call(fake_root)]
    assert not (fake_root["root"] / "logs/pilot" / LOCK).exists()


@pytest.mark.parametrize("edited", ["tasks/327_0/task.json", "tasks/327_0"], ids=["seed_edited", "seed_removed"])
def test_a_harness_edit_during_a_part_fails_loudly(fake_root, edited):
    """An edit while gpt-oss normal runs (the fourth uv call) fails the launcher before the next part, naming the
    log directory whose samples may have run other code than its fingerprint says."""
    (fake_root["root"] / ".env").touch()
    commands = dry_run(fake_root)
    env = {**fake_root["env"], "FAKE_UV_EDIT": str(fake_root["root"] / edited), "FAKE_UV_EDIT_AT": "4"}
    result = run_script(env, script=fake_root["script"])
    assert result.returncode == 1, result.stderr
    assert "changed while gpt-oss normal ran (exit 0)" in result.stderr
    assert f"move {fake_root['pilot']}/gpt-oss-normal out of the log root" in result.stderr
    assert 'AGENTS="gpt-oss sonnet"' in result.stderr
    calls = uv_calls(fake_root)
    assert calls[0] == preflight_call(fake_root)
    assert [["uv", *call] for call in calls[1:]] == commands[:3]
    assert not (fake_root["root"] / "logs/pilot" / LOCK).exists()


@pytest.mark.parametrize(
    ("target", "sig"),
    [("launcher", signal.SIGTERM), ("launcher", signal.SIGHUP), ("group", signal.SIGINT)],
    ids=["sigterm_to_launcher", "sighup_to_launcher", "ctrl_c_to_group"],
)
def test_stopping_the_launcher_interrupts_the_running_part(fake_root, tmp_path, background, target, sig):
    (fake_root["root"] / ".env").touch()
    proc = background({**fake_root["env"], "FAKE_UV_BLOCK": str(tmp_path / "never")}, fake_root["script"])
    wait_for(started_parts(fake_root, 1), 30, "the first part")
    [part] = fake_uv_pids(fake_root)
    if target == "group":
        os.killpg(proc.pid, sig)
    else:
        proc.send_signal(sig)
    assert proc.wait(timeout=60) == 130
    err = proc.err_path.read_text()
    assert f"stopping haiku normal on {sig.name} (eval-set pid {part})" in err
    assert 'AGENTS="haiku gpt-oss sonnet"' in err
    # The part got SIGINT, so it could cancel and clean up, and no later part started.
    assert fake_uv_pids(fake_root, "interrupted") == [part]
    assert not alive(part)
    assert len(uv_calls(fake_root)) == 2
    assert not (fake_root["root"] / "logs/pilot" / LOCK).exists()


# The real preflight and the mock pilot.


@pytest.mark.parametrize(
    "info_exit, message",
    [("1", "Docker is unavailable"), ("0", "image_digest.txt records")],
    ids=["daemon_down", "other_image"],
)
def test_preflight_refuses_without_docker_or_the_recorded_image(tmp_path, info_exit, message):
    """The real preflight with a fake docker first on PATH; the mock override keeps any slip unpaid."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    install_fake_docker(bin_dir)
    log_root = tmp_path / "logs"
    env = clean_env(
        PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        FAKE_DOCKER_INFO_EXIT=info_exit,
        AGENTS="haiku",
        PILOT_MODEL_OVERRIDE="mockllm/model",
        PILOT_EPOCHS_OVERRIDE="1",
        PILOT_LOG_ROOT=str(log_root),
    )
    result = run_script(env, script=SCRIPT, timeout=MOCK_TIMEOUT)
    assert result.returncode == 2, result.stderr
    assert message in result.stderr
    assert "nothing was run" in result.stderr
    # The lock was taken and released; no part's log directory was made.
    assert list(log_root.iterdir()) == []


def test_uv_env_file_cannot_add_inspect_options(tmp_path):
    """UV_ENV_FILE makes uv load a file into a command's environment unless the command passes --no-env-file."""
    probe = tmp_path / "probe.env"
    probe.write_text("INSPECT_EVAL_REASONING_TOKENS=1234\nINSPECT_EVAL_MAX_TOKENS=777\n")
    show = ["python", "-c", "import os; print(os.environ.get('INSPECT_EVAL_REASONING_TOKENS'))"]
    loaded = subprocess.run(["uv", "run", *show], cwd=ROOT, env=clean_env(UV_ENV_FILE=str(probe)), capture_output=True, text=True)
    assert loaded.stdout.strip() == "1234", loaded.stderr
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    install_fake_docker(bin_dir)
    log_root = tmp_path / "logs"
    env = clean_env(
        PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        FAKE_DOCKER_INFO_EXIT="1",
        UV_ENV_FILE=str(probe),
        AGENTS="haiku",
        PILOT_MODEL_OVERRIDE="mockllm/model",
        PILOT_EPOCHS_OVERRIDE="1",
        PILOT_LOG_ROOT=str(log_root),
    )
    result = run_script(env, script=SCRIPT, timeout=MOCK_TIMEOUT)
    # The preflight, which refuses any INSPECT_EVAL_* variable in its environment, got as far as Docker.
    assert result.returncode == 2, result.stderr
    assert "Docker is unavailable" in result.stderr
    assert "INSPECT_EVAL" not in result.stderr
    assert list(log_root.iterdir()) == []


def mock_env(log_root: Path, epochs: int = 1) -> dict[str, str]:
    return clean_env(
        INSPECT_DISPLAY="plain",
        AGENTS="haiku",
        PILOT_MODEL_OVERRIDE="mockllm/model",
        PILOT_EPOCHS_OVERRIDE=str(epochs),
        PILOT_LOG_ROOT=str(log_root),
    )


def run_mock(log_root: Path, epochs: int = 1) -> None:
    result = run_script(mock_env(log_root, epochs), script=SCRIPT, cwd=log_root.parent, timeout=MOCK_TIMEOUT)
    assert result.returncode == 0, result.stdout + result.stderr


def eval_files(log_root: Path) -> dict[str, bytes]:
    return {p.relative_to(log_root).as_posix(): p.read_bytes() for p in sorted(log_root.glob("*/*.eval"))}


def sample_uuids(path: Path) -> dict[tuple[str, int], str]:
    return {(s.id, s.epoch): s.uuid for s in read_eval_log(str(path)).samples}


@pytest.fixture(scope="module")
def mock_pilot(tmp_path_factory) -> Path:
    """The Haiku part of the pilot under the mock override with 1 epoch: 4 one-turn episodes."""
    log_root = tmp_path_factory.mktemp("mock_pilot") / "logs"
    run_mock(log_root)
    return log_root


@pytest.mark.docker
def test_mock_pilot_logs_record_the_pilot_settings(mock_pilot):
    assert sorted(p.name for p in mock_pilot.iterdir()) == ["haiku-conflict", "haiku-normal"]
    for variant in ("normal", "conflict"):
        log_dir = mock_pilot / f"haiku-{variant}"
        [path] = log_dir.glob("*.eval")
        log = read_eval_log(str(path))
        spec, config = log.eval, log.eval.config
        assert log.status == "success"
        assert spec.eval_set_id == (log_dir / ".eval-set-id").read_text().strip()
        assert spec.task_file == "src/rewardhack/task.py"
        assert spec.task_args == {"variant": variant}
        assert spec.model == "mockllm/model"
        assert spec.model_args == {}
        assert spec.model_generate_config.model_dump(exclude_none=True) == {
            "max_connections": 12,
            "reasoning_effort": "medium",
        }
        assert spec.metadata == {
            "model_cost_config": "costs.yaml", "model_cost_sha256": COSTS_SHA256, "harness_sha256": harness_sha256(ROOT),
        }
        assert config.sample_id == SAMPLES
        assert config.epochs == 1
        assert config.cost_limit == 0.10
        assert (config.turn_limit, config.time_limit) == (30, 900)
        assert (config.token_limit, config.token_limit_type) == (100_000, "output")
        assert (config.retry_on_error, config.fail_on_error, config.max_sandboxes) == (2, 0.05, 12)
        assert config.log_model_api is True
        assert sorted((s.id, s.epoch) for s in log.samples) == [("078_0", 1), ("327_0", 1)]
        assert all(s.error is None and s.metadata["variant"] == variant for s in log.samples)


@pytest.mark.docker
def test_reinvoking_the_mock_pilot_reruns_no_sample(mock_pilot, tmp_path):
    log_root = tmp_path / "logs"
    shutil.copytree(mock_pilot, log_root)
    before = eval_files(log_root)
    assert len(before) == 2

    # The same command finds every selected sample and epoch done, so no log is rewritten.
    run_mock(log_root)
    assert eval_files(log_root) == before

    # A second epoch makes each log incomplete: eval-set resumes it, reusing every finished epoch-1
    # sample (the same uuid) and running only epoch 2, which is the path a crashed run resumes by.
    first = {name.split("/")[0]: sample_uuids(log_root / name) for name in before}
    run_mock(log_root, epochs=2)
    for log_dir, uuids in first.items():
        [path] = (log_root / log_dir).glob("*.eval")
        samples = sample_uuids(path)
        assert sorted(samples) == [("078_0", 1), ("078_0", 2), ("327_0", 1), ("327_0", 2)]
        assert {key: samples[key] for key in uuids} == uuids


@pytest.mark.docker
def test_stopping_the_mock_pilot_cancels_its_eval_set(tmp_path, background):
    """SIGTERM to the launcher alone interrupts the real uv and Inspect, which cancel the part's log."""
    log_root = tmp_path / "logs"
    proc = background(mock_env(log_root), SCRIPT, cwd=tmp_path)
    wait_for(
        lambda: proc.poll() is not None or list(log_root.glob("haiku-normal/*.eval")),
        MOCK_TIMEOUT, "the first part's log",
    )
    assert proc.poll() is None, proc.err_path.read_text()
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=STOP_TIMEOUT) == 130, proc.err_path.read_text()
    assert "stopping haiku normal on SIGTERM" in proc.err_path.read_text()
    wait_for(lambda: not processes_mentioning(f"--log-dir {log_root}/"), 30, "the eval-set to exit")
    [path] = log_root.glob("*/*.eval")
    assert path.parent.name == "haiku-normal"
    assert read_eval_log(str(path), header_only=True).status == "cancelled"
    assert not (log_root / LOCK).exists()
