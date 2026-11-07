"""scripts/run_main.sh: the full runs' six eval-set commands, its refusals, the lock it shares with the pilot, its
harness fingerprint over all eight tasks, and its mock override.

No test here can make a paid call; the setup is tests/test_run_pilot.py's.
The dry-run, refusal, lock and fingerprint tests run copies of the scripts in a temporary root that has no .env file
(only costs.yaml and a small stand-in harness tree with all eight task directories), with a fake uv first on PATH
that records its arguments and runs nothing, so a broken refusal or dry run stops at the missing .env file or reaches
only the fake uv.
The Docker tests run the real script with the mockllm override, whose replies end each episode after one turn, with
blanked API keys and providers pointed at an unroutable address.
run_pilot.sh and run_main.sh source one scripts/launch_lib.sh, so the pilot's tests cover the shared logic and these
cover what the main runs add: their commands, the MAIN_* variables, the eight tasks, the cross-phase lock and the
check of the spend so far against PLAN's caps, in the preflight and again before each later part.
"""

import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import main_status
import pytest
import test_main_status as status_logs
import test_run_pilot as pilot
from inspect_ai.log import read_eval_log
from test_pilot_report import kill

from rewardhack.task import ROOT

SCRIPT = ROOT / "scripts/run_main.sh"
# The launchers, their library, and the spend count that run_main.sh's preflight imports.
SCRIPTS = ("run_main.sh", "run_pilot.sh", "launch_lib.sh", "main_status.py", "pilot_report.py")
TASKS = ["022_0", "022_2", "078_0", "327_0", "377_0", "732_2", "918_0", "943_0"]
RUNS = pilot.RUNS
# PLAN.md "Runs" table.
EPOCHS = {
    ("haiku", "normal"): 12, ("haiku", "conflict"): 24,
    ("gpt-oss", "normal"): 12, ("gpt-oss", "conflict"): 24,
    ("sonnet", "normal"): 5, ("sonnet", "conflict"): 5,
}
LOCK = ".run_main.lock"
PILOT_LOCK = pilot.LOCK
PREFLIGHT = pilot.PREFLIGHT
EVAL_SET = pilot.EVAL_SET
# run_main.sh's HARNESS_PATHS: every task directory, besides the paths the pilot's covers.
HARNESS_PATHS = ("costs.yaml", "uv.lock", "sandbox", "src/rewardhack", *(f"tasks/{task}" for task in TASKS))
FAKE_HARNESS = {
    "uv.lock": "version = 1\n",
    "sandbox/image_digest.txt": "sha256:fake\n",
    "src/rewardhack/task.py": "# the fake root's harness\n",
    **{f"tasks/{task}/task.json": "{}\n" for task in TASKS},
}
NO_DOCKER_PROBE = "Docker answers and the sandbox image is sha256:fake"
MOCK_TIMEOUT = 900


def harness_sha256(root: Path, paths: tuple[str, ...] = HARNESS_PATHS) -> str:
    """launch_lib.sh's harness fingerprint over paths, restated as in tests/test_run_pilot.py."""
    files = []
    for name in paths:
        top = root / name
        for path in [top] if top.is_file() else top.rglob("*"):
            parts = path.relative_to(root).parts
            if (
                path.is_file() and not path.is_symlink() and "__pycache__" not in parts
                and not any(p.startswith(".") for p in parts) and parts[:3] not in pilot.NOT_HARNESS
            ):
                files.append(path.relative_to(root).as_posix())
    lines = "".join(f"{hashlib.sha256((root / rel).read_bytes()).hexdigest()}  {rel}\n" for rel in sorted(files))
    return hashlib.sha256(lines.encode()).hexdigest()


def make_main_root(base: Path) -> dict:
    """Copies of both launchers and their library in base/repo, with costs.yaml, FAKE_HARNESS (which holds the pilot's
    stand-in harness too), no .env file and a fake uv first on PATH."""
    fake = pilot.make_fake_root(base)
    root = fake["root"]
    for name in SCRIPTS:
        shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
    for name, text in FAKE_HARNESS.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return fake | {
        "script": root / "scripts/run_main.sh", "pilot_script": root / "scripts/run_pilot.sh",
        "harness": harness_sha256(root), "pilot_harness": harness_sha256(root, pilot.HARNESS_PATHS),
        "main": f"{root}/logs/main", "pilot": f"{root}/logs/pilot",
    }


@pytest.fixture
def fake_root(tmp_path: Path) -> dict:
    return make_main_root(tmp_path)


def command_line(agent: str, variant: str, harness: str, log_root: str, mock_epochs: int | None = None) -> str:
    """One part's eval-set command, exactly as run_main.sh prints it."""
    model, _, cost = pilot.AGENTS[agent]
    epochs = EPOCHS[agent, variant] if mock_epochs is None else mock_epochs
    routing = f" -M '{pilot.GPT_OSS_PROVIDER}'" if agent == "gpt-oss" and mock_epochs is None else ""
    return (
        f"uv run --no-env-file inspect eval-set src/rewardhack/task.py -T variant={variant} --sample-id {','.join(TASKS)} "
        f"--epochs {epochs} --model {model if mock_epochs is None else 'mockllm/model'} --reasoning-effort medium{routing} "
        f"--turn-limit 30 --token-limit output:100k --time-limit 900 --cost-limit {cost} --max-sandboxes 12 "
        "--max-connections 12 --retry-on-error 2 --fail-on-error 0.05 --retry-attempts 2 --model-cost-config costs.yaml "
        f"--log-model-api --metadata model_cost_config=costs.yaml --metadata model_cost_sha256={pilot.COSTS_SHA256} "
        f"--metadata harness_sha256={harness} --log-dir {log_root}/{agent}-{variant}\n"
    )


def dry_output(harness: str, log_root: str, runs=RUNS, mock_epochs: int | None = None) -> str:
    return "".join(command_line(agent, variant, harness, log_root, mock_epochs) for agent, variant in runs)


def run_main(fake: dict, cwd: Path | None = None, **extra: str) -> subprocess.CompletedProcess:
    return pilot.run_script({**fake["env"], **extra}, script=fake["script"], cwd=cwd)


def dry_run(fake: dict, **extra: str) -> str:
    before = pilot.uv_calls(fake)
    result = run_main(fake, DRY_RUN="1", **extra)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert pilot.uv_calls(fake) == before
    return result.stdout


# The commands.


def test_dry_run_prints_the_six_full_run_commands_exactly(fake_root):
    out = dry_run(fake_root)
    assert out == dry_output(fake_root["harness"], fake_root["main"])
    commands = [line.split(" ") for line in out.splitlines()]
    assert len(commands) == 6
    episodes = sum(len(TASKS) * int(argv[argv.index("--epochs") + 1]) for argv in commands)
    assert episodes == 656
    for argv in commands:
        for token in pilot.FORBIDDEN:
            assert token not in argv, (token, argv)
        assert argv[argv.index("--reasoning-effort") + 1] == "medium"


def test_the_real_dry_run_is_the_plans_and_fingerprints_all_eight_tasks():
    """The real script in dry run (it runs nothing): absolute log directories under logs/main and the harness hash."""
    result = pilot.run_script({**pilot.clean_env(), "DRY_RUN": "1"}, script=SCRIPT)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == dry_output(harness_sha256(ROOT), f"{ROOT}/logs/main")


def test_dry_run_is_the_same_from_any_directory(fake_root, tmp_path):
    assert run_main(fake_root, cwd=tmp_path, DRY_RUN="1").stdout == dry_run(fake_root)


def test_the_two_inspect_display_settings_are_allowed(fake_root):
    """INSPECT_DISPLAY and INSPECT_NO_ANSI change only what the terminal shows, so they are the Inspect variables allowed."""
    assert dry_run(fake_root, INSPECT_DISPLAY="plain", INSPECT_NO_ANSI="1") == dry_run(fake_root)


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
    assert dry_run(fake_root, AGENTS=agents) == dry_output(fake_root["harness"], fake_root["main"], runs)


def test_mock_override_swaps_every_model_and_log_dir(fake_root, tmp_path):
    root = str(tmp_path / "mock")
    out = dry_run(fake_root, MAIN_MODEL_OVERRIDE="mockllm/model", MAIN_EPOCHS_OVERRIDE="1", MAIN_LOG_ROOT=root)
    assert out == dry_output(fake_root["harness"], root, mock_epochs=1)


def test_main_status_expects_what_the_launcher_runs(fake_root):
    """scripts/main_status.py counts completed samples against the same tasks, epochs and part directories."""
    commands = [line.split(" ") for line in dry_run(fake_root).splitlines()]
    assert [(argv[argv.index("-T") + 1], argv[argv.index("--log-dir") + 1]) for argv in commands] == [
        (f"variant={variant}", f"{fake_root['main']}/{main_status.part_name(agent, variant)}")
        for agent, variant in main_status.PARTS
    ]
    for argv, (agent, variant) in zip(commands, main_status.PARTS, strict=True):
        assert argv[argv.index("--sample-id") + 1].split(",") == list(main_status.TASKS)
        assert int(argv[argv.index("--epochs") + 1]) == main_status.EPOCHS[agent][variant]
        assert argv[argv.index("--model") + 1] == main_status.AGENT_MODELS[agent]
        assert float(argv[argv.index("--cost-limit") + 1]) == main_status.COST_LIMITS[agent]


# Refusals.


def mock(model: str = "mockllm/model", epochs: str = "1", root: str = "/tmp/m") -> dict[str, str]:
    return {"MAIN_MODEL_OVERRIDE": model, "MAIN_EPOCHS_OVERRIDE": epochs, "MAIN_LOG_ROOT": root}


MODEL_MUST = "MAIN_MODEL_OVERRIDE must be mockllm/model whenever a MAIN_* override is set"
EPOCHS_MUST = "MAIN_EPOCHS_OVERRIDE must be a positive integer"
ROOT_ABSOLUTE = "MAIN_LOG_ROOT must be an absolute directory"
ROOT_DOTS = "MAIN_LOG_ROOT must not contain . or .. components"
ROOT_OUTSIDE = "MAIN_LOG_ROOT must be outside this repository"
UNREAD = "reads only DRY_RUN, AGENTS, MAIN_MODEL_OVERRIDE, MAIN_EPOCHS_OVERRIDE and MAIN_LOG_ROOT; unset them first: "
PILOT_SETTING = "these variables look like settings of scripts/run_pilot.sh, which this script never reads; unset them first: "
MAIN_SETTING = "these variables look like settings of scripts/run_main.sh, which this script never reads; unset them first: "
PYTHON_VARS = "unset these Python variables first, which would change the code the runs load: "
UV_VARS = pilot.UV_VARS
# {root} is the fake root, {upper} the same path in upper case, and {link} a symlink to it.
REFUSALS = [
    ({"DRY_RUN": "yes"}, "DRY_RUN must be 1, 0 or unset"),
    ({"DRY_RUN": ""}, "DRY_RUN must be 1, 0 or unset"),
    ({"AGENTS": "claude"}, "unknown agent 'claude'"),
    ({"AGENTS": ""}, "AGENTS selects no agent"),
    ({"AGENTS": "haiku,sonnet"}, "unknown agent 'haiku,sonnet'"),
    ({"INSPECT_EVAL_FALLBACK_MODELS": "anthropic/claude-sonnet-5-5"}, "INSPECT_EVAL_FALLBACK_MODELS"),
    ({"INSPECT_EVAL_EPOCHS": "1"}, "INSPECT_EVAL_EPOCHS"),
    # Other Inspect variables change a run too: a JSON log format would write logs the spend check reports as bad input.
    ({"INSPECT_LOG_FORMAT": "json"}, "unset these Inspect option variables first: INSPECT_LOG_FORMAT"),
    ({"INSPECT_SANDBOX_MAX_EXEC_OUTPUT_SIZE": "1", "INSPECT_DISPLAY": "plain"}, "first: INSPECT_SANDBOX_MAX_EXEC_OUTPUT_SIZE\n"),
    # Another module search path could load other code into every run under this harness_sha256.
    *(({name: "/tmp/site"}, PYTHON_VARS + name) for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")),
    ({"PYTHONPATH": "", "PYTHONHOME": "/tmp/h"}, PYTHON_VARS + "PYTHONHOME PYTHONPATH"),
    # So could uv's own variables, which choose the project, working directory, environment, sync and config file.
    *(
        ({name: value}, UV_VARS + name)
        for name, value in (
            ("UV_WORKING_DIR", "/tmp"), ("UV_PROJECT", "/tmp"), ("UV_PROJECT_ENVIRONMENT", "/tmp/venv"), ("UV_NO_SYNC", "1"),
            ("UV_FROZEN", "1"), ("UV_NO_EDITABLE", "1"), ("UV_CONFIG_FILE", "/tmp/uv.toml"), ("UV_NO_PROJECT", "1"),
            ("UV_PYTHON", "3.14"), ("UV_NO_DEV", "1"),
        )
    ),
    (
        {"UV_PROJECT": "/tmp", "UV_WORKING_DIR": "/tmp", "UV_NO_SYNC": "1", "UV_FROZEN": "1", "UV_NO_EDITABLE": "1"},
        UV_VARS + "UV_FROZEN UV_NO_EDITABLE UV_NO_SYNC UV_PROJECT UV_WORKING_DIR\n",
    ),
    (mock(model="anthropic/claude-haiku-5-5"), MODEL_MUST),
    (mock(model="mockllm/other"), MODEL_MUST),
    ({"MAIN_MODEL_OVERRIDE": "mockllm/model"}, EPOCHS_MUST),
    ({"MAIN_MODEL_OVERRIDE": "mockllm/model", "MAIN_EPOCHS_OVERRIDE": "1"}, ROOT_ABSOLUTE),
    ({"MAIN_MODEL_OVERRIDE": "mockllm/model", "MAIN_LOG_ROOT": "/tmp/m"}, EPOCHS_MUST),
    ({"MAIN_EPOCHS_OVERRIDE": "1"}, MODEL_MUST),
    ({"MAIN_LOG_ROOT": "/tmp/m"}, MODEL_MUST),
    ({"MAIN_EPOCHS_OVERRIDE": "1", "MAIN_LOG_ROOT": "/tmp/m"}, MODEL_MUST),
    # A MAIN_* variable set to an empty value still means a mock run, so it is refused instead of becoming the paid runs.
    (mock(model="", epochs="", root=""), MODEL_MUST),
    ({"MAIN_MODEL_OVERRIDE": ""}, MODEL_MUST),
    ({"MAIN_LOG_ROOT": ""}, MODEL_MUST),
    # A misspelled control variable would be ignored, leaving a paid run, so any MAIN_* or *DRY* name (ignoring case)
    # that the script does not read is refused, beside a valid setting or not.
    *(
        ({name: value}, UNREAD + name)
        for name, value in (
            ("MAIN_MODEL", "mockllm/model"), ("MAIN_EPOCHS", "1"), ("MAIN_LOGROOT", "/tmp/m"),
            ("main_model_override", "mockllm/model"), ("Main_Log_Root", "/tmp/m"), ("MAIN_", ""),
            ("DRYRUN", "1"), ("DRY", "1"), ("dry_run", "1"), ("DRY_RUN_", "1"), ("MAIN_DRY_RUN", "1"),
        )
    ),
    ({**mock(), "MAIN_EPOCH_OVERRIDE": "2"}, UNREAD + "MAIN_EPOCH_OVERRIDE"),
    # A mistyped prefix would leave the paid runs too, so any name holding an override's suffix is refused.
    *(
        ({name: value}, UNREAD + name)
        for name, value in (
            ("MIAN_MODEL_OVERRIDE", "mockllm/model"), ("MIAN_EPOCHS_OVERRIDE", "1"), ("MIAN_LOG_ROOT", "/tmp/m"),
            ("MODEL_OVERRIDE", "mockllm/model"), ("EPOCHS_OVERRIDE", "1"), ("LOG_ROOT", "/tmp/m"), ("Model_Override", "x"),
        )
    ),
    (
        {"MIAN_MODEL_OVERRIDE": "mockllm/model", "MIAN_EPOCHS_OVERRIDE": "1", "MIAN_LOG_ROOT": "/tmp/m"},
        UNREAD + "MIAN_EPOCHS_OVERRIDE MIAN_LOG_ROOT MIAN_MODEL_OVERRIDE",
    ),
    *(
        ({name: value}, UNREAD + name)
        for name, value in (
            ("AGENT", "gpt-oss"), ("agents", "gpt-oss"), ("Agents", "sonnet"), ("agent", "haiku"),
            ("AGENTS_", "gpt-oss"), ("AGENT_LIST", "gpt-oss"), ("AGENTSS", "haiku"),
        )
    ),
    ({"AGENTS": "gpt-oss", "agents": "gpt-oss"}, UNREAD + "agents"),
    # A variable meant for run_pilot.sh would be ignored here, so every PILOT_* name (ignoring case) is refused too,
    # even the pilot's own valid overrides and even beside a valid main override.
    *(
        ({name: value}, PILOT_SETTING + name)
        for name, value in (
            ("PILOT_MODEL_OVERRIDE", "mockllm/model"), ("PILOT_EPOCHS_OVERRIDE", "1"), ("PILOT_LOG_ROOT", "/tmp/m"),
            ("pilot_log_root", "/tmp/m"), ("PILOT_", ""), ("PILOT_X", "1"),
        )
    ),
    ({**mock(), "PILOT_EPOCHS_OVERRIDE": "2"}, PILOT_SETTING + "PILOT_EPOCHS_OVERRIDE"),
    ({"PILOT_MODEL_OVERRIDE": "mockllm/model", "PILOT_EPOCHS_OVERRIDE": "1", "PILOT_LOG_ROOT": "/tmp/m"},
     PILOT_SETTING + "PILOT_EPOCHS_OVERRIDE PILOT_LOG_ROOT PILOT_MODEL_OVERRIDE"),
    *((mock(epochs=epochs), EPOCHS_MUST) for epochs in ("0", "two", "1.5", "-1", "01", "")),
    *((mock(root=root), ROOT_ABSOLUTE) for root in ("", "logs/mock", "./logs/mock")),
    *((mock(root=root), ROOT_DOTS) for root in ("/tmp/x/../m", "{root}/logs/./main", "{root}/..")),
    *(
        (mock(root=root), ROOT_OUTSIDE)
        for root in (
            "{root}", "{root}/", "{root}/logs/main", "{root}/logs/main/", "{root}/logs/main/mock", "{root}/logs/pilot/mock",
            "{root}/logs/mock", "{root}/LOGS/main", "{root}//logs/main", "{root}/logs/Main", "{upper}/logs/mock",
            "{link}/logs/mock",
        )
    ),
]


def assert_refused(result: subprocess.CompletedProcess, fake: dict, message: str, name: str = "run_main.sh") -> None:
    assert result.returncode == 2, result.stdout + result.stderr
    assert result.stdout == ""
    assert result.stderr.startswith(f"{name}: ")
    assert message in result.stderr
    assert pilot.uv_calls(fake) == []
    assert not (fake["root"] / "logs").exists()


@pytest.mark.parametrize(("env", "message"), REFUSALS)
def test_refusals_print_and_run_nothing(fake_root, tmp_path, env, message):
    (tmp_path / "link").symlink_to(fake_root["root"])
    places = {"root": str(fake_root["root"]), "upper": str(fake_root["root"]).upper(), "link": str(tmp_path / "link")}
    env = {key: value.format(**places) for key, value in env.items()}
    assert_refused(run_main(fake_root, **{"DRY_RUN": "1", **env}), fake_root, message)


@pytest.mark.parametrize(
    "name, value", [("MAIN_MODEL_OVERRIDE", "mockllm/model"), ("MAIN_LOG_ROOT", "/tmp/m"), ("main_epochs_override", "1"), ("MAIN_", "")]
)
@pytest.mark.parametrize("dry", ["1", None], ids=["dry_run", "real_run"])
def test_the_pilot_refuses_every_main_variable(fake_root, name, value, dry):
    """Vice versa: a MAIN_* variable would be ignored by run_pilot.sh, so it is refused, even beside a valid pilot override."""
    (fake_root["root"] / ".env").touch()
    env = {**fake_root["env"], name: value, **({"DRY_RUN": dry} if dry else {})}
    assert_refused(pilot.run_script(env, script=fake_root["pilot_script"]), fake_root, MAIN_SETTING + name, "run_pilot.sh")
    env |= pilot.mock(root=str(fake_root["root"].parent / "mock"))
    assert_refused(pilot.run_script(env, script=fake_root["pilot_script"]), fake_root, MAIN_SETTING + name, "run_pilot.sh")


def test_agent_inside_a_name_is_not_a_control_variable(fake_root):
    others = {"AI_AGENT": "claude-code", "CLAUDE_AGENT_SDK_VERSION": "1", "SSH_AGENT_PID": "1", "MY_MAIN_X": "1", "XPILOT_Y": "1"}
    assert dry_run(fake_root, **others) == dry_output(fake_root["harness"], fake_root["main"])


def test_real_run_refuses_without_env_file(fake_root):
    result = run_main(fake_root)
    assert result.returncode == 2
    assert f"run_main.sh: no .env file in {fake_root['root']}" in result.stderr
    assert pilot.uv_calls(fake_root) == []


@pytest.mark.parametrize(
    "name, value, message",
    [
        ("DRYRUN", "1", UNREAD + "DRYRUN"), ("MAIN_MODEL", "mockllm/model", UNREAD + "MAIN_MODEL"),
        ("AGENT", "gpt-oss", UNREAD + "AGENT"), ("PILOT_LOG_ROOT", "/tmp/m", PILOT_SETTING + "PILOT_LOG_ROOT"),
    ],
)
def test_real_run_refuses_a_misspelled_or_pilot_variable(fake_root, name, value, message):
    """With a .env file present, a misspelled dry run, mock override or AGENTS, or a pilot setting, runs nothing."""
    (fake_root["root"] / ".env").touch()
    assert_refused(run_main(fake_root, **{name: value}), fake_root, message)


def preflight_call(fake: dict, runs=RUNS) -> list[str]:
    return [*PREFLIGHT, ",".join(TASKS), fake["harness"], *(f"{fake['main']}/{agent}-{variant}" for agent, variant in runs)]


def spend_check_call(fake: dict, agent: str) -> list[str]:
    """The spend check of agent that a real run makes before each part after the first."""
    return [*PREFLIGHT, fake["main"], agent]


def eval_set_calls(fake: dict) -> list[list[str]]:
    return [call for call in pilot.uv_calls(fake) if call[:4] == EVAL_SET]


def as_printed(fake: dict, runs=RUNS) -> list[str]:
    """The dry run's commands as fake uv records them: uv's arguments, one list per part."""
    return [shlex.split(line)[1:] for line in dry_output(fake["harness"], fake["main"], runs).splitlines()]


def test_real_run_runs_the_dry_run_commands_in_order(fake_root, tmp_path):
    (fake_root["root"] / ".env").touch()
    result = run_main(fake_root, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    calls = pilot.uv_calls(fake_root)
    assert calls[0] == preflight_call(fake_root)
    # Each part after the first follows a spend check of its own agent.
    commands = as_printed(fake_root)
    assert calls[1:] == [
        call for i, ((agent, _), command) in enumerate(zip(RUNS, commands, strict=True))
        for call in ([spend_check_call(fake_root, agent)] if i else []) + [command]
    ]
    # 288 x $0.10, 288 x $0.05 and 80 x $1.00: PLAN's Haiku and Sonnet hard caps, and most of the shared gpt-oss cap.
    assert (
        "run_main.sh: the selected parts' samples times their cost limits come to $28.80 for Haiku (PLAN's hard cap is "
        "$28.80), $14.40 for gpt-oss on OpenRouter (counted against the $15.50 gpt-oss cap, which the pilot and the smoke "
        "test share) and $80.00 for Sonnet (PLAN's hard cap is $80.00), which a generation in flight and retried errors "
        "can exceed; scripts/main_status.py reports the spend so far against these caps\n"
    ) in result.stderr
    for (agent, variant), cost in zip(RUNS, ["0.10", "0.10", "0.05", "0.05", "1.00", "1.00"], strict=True):
        samples = len(TASKS) * EPOCHS[agent, variant]
        assert f"run_main.sh: starting {agent} {variant} ({samples} samples, cost limit ${cost} each):" in result.stderr
    assert result.stderr.endswith("run_main.sh: every selected part finished\n")
    assert not (fake_root["root"] / "logs/main" / LOCK).exists()
    assert not (fake_root["root"] / "logs/pilot").exists()


@pytest.mark.parametrize(
    "agents, line",
    [
        ("gpt-oss", "$14.40 for gpt-oss on OpenRouter (counted against the $15.50 gpt-oss cap, which the pilot and the smoke test share), which"),
        ("sonnet haiku", "$28.80 for Haiku (PLAN's hard cap is $28.80) and $80.00 for Sonnet (PLAN's hard cap is $80.00), which"),
    ],
)
def test_the_cost_line_names_only_the_selected_agents(fake_root, agents, line):
    (fake_root["root"] / ".env").touch()
    result = run_main(fake_root, AGENTS=agents)
    assert result.returncode == 0, result.stderr
    assert f"cost limits come to {line} a generation in flight" in result.stderr


def test_real_run_stops_at_the_first_failed_part(fake_root):
    (fake_root["root"] / ".env").touch()
    result = run_main(fake_root, FAKE_UV_FAIL_AT="4")
    assert result.returncode == 1
    assert "run_main.sh: gpt-oss conflict did not finish (exit 1)" in result.stderr
    assert 'run_main.sh: to resume, re-invoke with AGENTS="gpt-oss sonnet"' in result.stderr
    assert eval_set_calls(fake_root) == as_printed(fake_root)[:4]
    assert not (fake_root["root"] / "logs/main" / LOCK).exists()


# The eight tasks and the harness fingerprint.


@pytest.mark.parametrize("task", ["022_0", "943_0", "732_2"])
def test_a_missing_task_directory_is_refused(fake_root, task):
    shutil.rmtree(fake_root["root"] / "tasks" / task)
    result = run_main(fake_root, DRY_RUN="1")
    assert result.returncode == 2 and result.stdout == ""
    assert f"run_main.sh: tasks/{task} is missing from {fake_root['root']}" in result.stderr
    assert pilot.uv_calls(fake_root) == []


SITECUSTOMIZE = """\
# Stands in for Docker and the harness's task listing in the preflight, which imports these names at run time.
import os
from pathlib import Path

import rewardhack.containers
import rewardhack.task

tasks = Path(os.environ["FAKE_TASKS_DIR"])
rewardhack.containers.docker_problem = lambda: None
rewardhack.task.recorded_image_digest = lambda: "sha256:fake"
rewardhack.containers.task_dirs = lambda: sorted(d for d in tasks.iterdir() if (d / "task.json").exists())
"""


@pytest.fixture
def fake_preflight(fake_root, tmp_path) -> dict:
    """The real preflight, run by the fake uv, with Docker and the task listing replaced by the fake root's."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
    (fake_root["root"] / ".env").touch()
    added = {"PYTHONPATH": str(site), "FAKE_TASKS_DIR": str(fake_root["root"] / "tasks")}
    return {**fake_root["env"], "FAKE_UV_RUN_PREFLIGHT": "1", "FAKE_UV_PREFLIGHT_ENV": json.dumps(added)}


def test_the_preflight_passes_with_every_task_and_runs_the_parts(fake_root, fake_preflight):
    result = pilot.run_script(fake_preflight, script=fake_root["script"])
    assert result.returncode == 0, result.stderr
    assert f"run_main.sh: {NO_DOCKER_PROBE}\n" in result.stderr
    assert eval_set_calls(fake_root) == as_printed(fake_root)
    # The spend checks before the five later parts ran for real and passed.
    checks = [call for call in pilot.uv_calls(fake_root)[1:] if call[:4] == PREFLIGHT]
    assert checks == [spend_check_call(fake_root, agent) for agent, _ in RUNS[1:]]


@pytest.mark.parametrize(
    "script, removed",
    [("script", ["943_0"]), ("script", ["022_0", "918_0"]), ("pilot_script", ["327_0"])],
    ids=["main_one", "main_two", "pilot_seed"],
)
def test_the_preflight_refuses_a_task_the_harness_would_not_load(fake_root, fake_preflight, script, removed):
    """A task directory without its task.json passes the path check but is no task the harness loads."""
    for task in removed:
        (fake_root["root"] / f"tasks/{task}/task.json").unlink()
    result = pilot.run_script(fake_preflight, script=fake_root[script])
    name = fake_root[script].name
    assert result.returncode == 2, result.stderr
    assert f"{name}: no task directory for {sorted(removed)}\n" in result.stderr
    assert "nothing was run" in result.stderr
    assert [call[:4] for call in pilot.uv_calls(fake_root)] == [PREFLIGHT]


def recorded_harness(fake: dict) -> str:
    [value] = {
        token.removeprefix("harness_sha256=") for line in dry_run(fake).splitlines()
        for token in line.split(" ") if token.startswith("harness_sha256=")
    }
    return value


def test_harness_sha256_covers_all_eight_tasks(fake_root):
    root = fake_root["root"]
    first = recorded_harness(fake_root)
    assert first == fake_root["harness"] != fake_root["pilot_harness"]
    for name in (
        "tasks/conftest.py", "tasks/LICENSE-CWEval", "tasks/943_0/.cache/x", "tasks/022_2/__pycache__/x.pyc",
        "tasks/020_0/task.json", "README.md", "scripts/other.py", "src/rewardhack/monitors/m1_regex.py",
    ):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("not harness\n")
    assert recorded_harness(fake_root) == first
    seen = {first}
    for task in TASKS:
        for name in (f"tasks/{task}/task.json", f"tasks/{task}/hidden/test_new.py"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text((path.read_text() if path.exists() else "") + "# changed\n")
            changed = recorded_harness(fake_root)
            assert changed == harness_sha256(root) and changed not in seen, name
            seen.add(changed)


def test_a_part_logged_under_another_harness_is_refused(fake_root):
    (fake_root["root"] / ".env").touch()
    main = fake_root["root"] / "logs/main"
    pilot.write_part_log(main / "haiku-normal" / pilot.LOG_NAME, fake_root["harness"])
    pilot.write_part_log(main / "sonnet-conflict" / pilot.LOG_NAME, fake_root["pilot_harness"])
    result = run_main(fake_root, FAKE_UV_RUN_PREFLIGHT="1")
    assert result.returncode == 2, result.stderr
    assert f"run_main.sh: the harness is now {fake_root['harness']}, but these log directories hold logs of another harness" in result.stderr
    assert f"eval-set would reuse: {main}/sonnet-conflict (harness_sha256 {fake_root['pilot_harness']});" in result.stderr
    assert "haiku-normal" not in result.stderr
    assert "move each one into logs/main-rejected/ and re-run" in result.stderr
    assert pilot.uv_calls(fake_root) == [preflight_call(fake_root)]


@pytest.mark.parametrize("edited", ["src/rewardhack/task.py", "costs.yaml", "tasks/377_0/task.json", "tasks/022_2"])
def test_a_harness_edit_after_launch_stops_the_next_part(fake_root, edited):
    (fake_root["root"] / ".env").touch()
    result = run_main(fake_root, FAKE_UV_EDIT=str(fake_root["root"] / edited), FAKE_UV_EDIT_AT="1")
    assert result.returncode == 2, result.stderr
    assert "run_main.sh: costs.yaml or a harness file (costs.yaml uv.lock sandbox src/rewardhack tasks/022_0 tasks/022_2" in result.stderr
    assert "changed since launch, so haiku normal was not started" in result.stderr
    assert f"it would record harness_sha256 {fake_root['harness']} while running other code" in result.stderr
    assert pilot.uv_calls(fake_root) == [preflight_call(fake_root)]
    assert not (fake_root["root"] / "logs/main" / LOCK).exists()


@pytest.mark.parametrize("edited", ["tasks/943_0/task.json", "tasks/078_0"], ids=["task_edited", "task_removed"])
def test_a_harness_edit_during_a_part_fails_loudly(fake_root, edited):
    """An edit while haiku conflict runs (the fourth uv call, after the spend check before it) fails before the next
    part, naming its log directory."""
    (fake_root["root"] / ".env").touch()
    result = run_main(fake_root, FAKE_UV_EDIT=str(fake_root["root"] / edited), FAKE_UV_EDIT_AT="4")
    assert result.returncode == 1, result.stderr
    assert "changed while haiku conflict ran (exit 0)" in result.stderr
    assert f"move {fake_root['main']}/haiku-conflict into logs/main-rejected/ before resuming" in result.stderr
    assert 'AGENTS="haiku gpt-oss sonnet"' in result.stderr
    assert eval_set_calls(fake_root) == as_printed(fake_root)[:2]


# The spend so far against PLAN's caps, which the preflight of a real run checks with scripts/main_status.py.


def spend_log(fake: dict, where: str, agent: str, variant: str = "normal", **changes) -> str:
    """A log as tests/test_main_status.py writes one, of the fake root's harness, in the fake root's logs/<where>."""
    return status_logs.write_part(fake["root"] / "logs" / where, agent, variant, harness=fake["harness"], **changes)


LAUNCH_APPROVAL = (
    "; CLAUDE.md requires Jessup's approval before any spend above a cap, and an approved cap goes into PLAN's Budget "
    "and AGENT_CAP in scripts/pilot_report.py; scripts/main_status.py reports the spend so far\n"
)


@pytest.mark.parametrize(
    "agents, logs, message",
    [
        ("haiku", [("main-rejected/attempt-1", "haiku", {"cost": 0.0, "stats_cost": 29.0})], "Haiku has spent $29.0000, above its $28.80 cap"),
        # 288 x $0.10 is the Haiku cap, so any rejected Haiku spend leaves no room for every sample.
        (
            "haiku sonnet", [("main-rejected/attempt-1", "haiku", {"cost": 0.0, "stats_cost": 0.5})],
            "Haiku has spent $0.5000, and its 288 samples still to run at up to $0.10 each could bring it to $29.3000, above its $28.80 cap",
        ),
        (
            "haiku", [("main", "haiku", {"cost": 0.1001})],
            "Haiku has spent $9.6096, and its 192 samples still to run at up to $0.10 each could bring it to $28.8096, above its $28.80 cap",
        ),
        (
            "gpt-oss", [("pilot", "gpt-oss", {"cost": 0.0, "stats_cost": 1.2, "eval_id": "pilot"})],
            "gpt-oss has spent $0.0000, and its 288 samples still to run at up to $0.05 each could bring it to $14.4000, above its $14.30 cap",
        ),
        # The preflight checks every selected agent, not only the first: Haiku fits, Sonnet's rejected spend does not.
        ("haiku sonnet", [("main-rejected/attempt-1", "sonnet", {"cost": 0.0, "stats_cost": 81.0})], "Sonnet has spent $81.0000, above its $80.00 cap"),
    ],
    ids=["crossed", "rejected_spend", "a_part_over_its_limits", "pilot_openrouter_spend", "second_agent_crossed"],
)
def test_a_real_run_refuses_while_a_cap_could_be_crossed(fake_root, fake_preflight, agents, logs, message):
    for where, agent, changes in logs:
        spend_log(fake_root, where, agent, **changes)
    result = pilot.run_script({**fake_preflight, "AGENTS": agents}, script=fake_root["script"])
    assert result.returncode == 2, result.stderr
    assert f"run_main.sh: {message}{LAUNCH_APPROVAL}" in result.stderr
    assert result.stderr.endswith("run_main.sh: preflight failed; nothing was run\n")
    assert NO_DOCKER_PROBE not in result.stderr
    assert eval_set_calls(fake_root) == []
    assert not (fake_root["root"] / "logs/main" / LOCK).exists()


@pytest.mark.parametrize(
    "agents, logs, parts",
    [
        # A finished Haiku part at its full limits leaves exactly the conflict part's 192 x $0.10.
        ("haiku", [("main", "haiku", "normal", {"cost": 0.1})], 2),
        # A killed part runs only its missing samples: $6.00 for 60 written, then 36 + 192 at $0.10.
        ("haiku", [("main", "haiku", "normal", {"cost": 0.1, "status": "started", "skip": 36})], 2),
        # Rejected Haiku spend over its cap stops only Haiku's parts.
        ("gpt-oss sonnet", [("main-rejected", "haiku", "normal", {"cost": 0.0, "stats_cost": 29.0})], 4),
    ],
    ids=["finished_part", "killed_part", "other_agent_over"],
)
def test_a_real_run_within_every_selected_cap_starts(fake_root, fake_preflight, agents, logs, parts):
    for where, agent, variant, changes in logs:
        path = spend_log(fake_root, where, agent, variant, **changes)
        if changes.get("status") == "started":
            kill(path)
    result = pilot.run_script({**fake_preflight, "AGENTS": agents}, script=fake_root["script"])
    assert result.returncode == 0, result.stderr
    assert f"run_main.sh: {NO_DOCKER_PROBE}\n" in result.stderr
    assert len(eval_set_calls(fake_root)) == parts


def test_a_log_main_status_cannot_count_stops_a_real_run(fake_root, fake_preflight):
    """A log under logs/main outside the part directories would hold spend that counts against no cap."""
    stray = spend_log(fake_root, "main/old", "sonnet")
    result = pilot.run_script(fake_preflight, script=fake_root["script"])
    assert result.returncode == 2, result.stderr
    assert (
        f"run_main.sh: scripts/main_status.py cannot count the spend so far: {stray} is under {fake_root['main']} but in "
        f"none of its six part directories, so its spend would count against no cap; move it into "
        f"{fake_root['root']}/logs/main-rejected\n"
    ) in result.stderr
    assert eval_set_calls(fake_root) == []


def test_a_mock_run_skips_the_spend_check(fake_root, fake_preflight, tmp_path):
    """mockllm costs nothing, and the real runs' logs are not a mock run's."""
    spend_log(fake_root, "main-rejected", "haiku", cost=0.0, stats_cost=29.0)
    spend_log(fake_root, "main/old", "sonnet")
    result = pilot.run_script({**fake_preflight, **mock(root=str(tmp_path / "mock"))}, script=fake_root["script"])
    assert result.returncode == 0, result.stderr
    assert len(eval_set_calls(fake_root)) == 6
    assert [call for call in pilot.uv_calls(fake_root) if call[:4] == PREFLIGHT] == [
        [*PREFLIGHT, ",".join(TASKS), fake_root["harness"], *(f"{tmp_path}/mock/{agent}-{variant}" for agent, variant in RUNS)]
    ]


@pytest.mark.parametrize(
    "where, changes, message",
    [
        # 96 samples at $0.31 come to $29.76, above the $28.80 Haiku cap.
        ("main", {"cost": 0.31}, f"Haiku has spent $29.7600, above its $28.80 cap{LAUNCH_APPROVAL}"),
        # Over its limits by a hundredth of a cent each, haiku normal leaves too little for haiku conflict's 192 samples.
        (
            "main", {"cost": 0.1001},
            "Haiku has spent $9.6096, and its 192 samples still to run at up to $0.10 each could bring it to $28.8096, "
            f"above its $28.80 cap{LAUNCH_APPROVAL}",
        ),
        ("main/old", {}, "scripts/main_status.py cannot count the spend so far: "),
    ],
    ids=["crossed", "no_room_left", "bad_input"],
)
def test_a_part_that_takes_its_agent_past_the_cap_stops_the_next_part(
    fake_root, fake_preflight, tmp_path, background, where, changes, message
):
    """The preflight passes on no spend; then haiku normal spends (its log is written while the fake eval-set runs),
    and the spend check before haiku conflict refuses it."""
    release = tmp_path / "release"
    proc = background({**fake_preflight, "AGENTS": "haiku gpt-oss", "FAKE_UV_BLOCK": str(release)}, fake_root["script"])
    pilot.wait_for(started(fake_root, 1), 60, "the haiku normal part")
    spend_log(fake_root, where, "haiku", **changes)
    release.touch()
    assert proc.wait(timeout=120) == 2
    err = proc.err_path.read_text()
    assert f"run_main.sh: {message}" in err
    assert err.endswith(
        "run_main.sh: the check before haiku conflict failed, so it was not started\n"
        "run_main.sh: an agent refused for its cap stays out of AGENTS until Jessup approves more spend, and uv run "
        "python scripts/main_status.py names the agents whose spend so far lets the launcher start them, or the bad "
        "input that stops every part\n"
        'run_main.sh: to resume, re-invoke with AGENTS="haiku gpt-oss" and otherwise the same settings (finished samples '
        "are reused only while the harness is unchanged; after a fix, the launcher names the log directories to move aside)\n"
    )
    assert eval_set_calls(fake_root) == as_printed(fake_root)[:1]
    assert pilot.uv_calls(fake_root)[-1] == spend_check_call(fake_root, "haiku")
    assert not (fake_root["root"] / "logs/main" / LOCK).exists()


def test_an_inspect_key_added_to_the_env_file_stops_the_next_part(fake_root, fake_preflight, tmp_path, background):
    """Each part is a new eval-set that loads .env, so the check before each later part scans it again."""
    release = tmp_path / "release"
    proc = background({**fake_preflight, "AGENTS": "haiku", "FAKE_UV_BLOCK": str(release)}, fake_root["script"])
    pilot.wait_for(started(fake_root, 1), 60, "the haiku normal part")
    dotenv_file = fake_root["root"] / ".env"
    dotenv_file.write_text("ANTHROPIC_API_KEY=not-a-real-key\nINSPECT_EVAL_FALLBACK_MODELS=anthropic/claude-sonnet-5-5\n")
    release.touch()
    assert proc.wait(timeout=120) == 2
    err = proc.err_path.read_text()
    assert f"run_main.sh: remove these Inspect option variables from {dotenv_file} first: INSPECT_EVAL_FALLBACK_MODELS\n" in err
    assert "run_main.sh: the check before haiku conflict failed, so it was not started\n" in err
    assert "not-a-real-key" not in err and "claude-sonnet" not in err
    assert eval_set_calls(fake_root) == as_printed(fake_root)[:1]
    assert pilot.uv_calls(fake_root)[-1] == spend_check_call(fake_root, "haiku")


def test_a_mock_run_names_its_own_log_root_for_logs_to_move_aside(fake_root, tmp_path):
    """A mock run's stale logs go anywhere out of its log root, not into the repository's logs/main-rejected."""
    (fake_root["root"] / ".env").touch()
    root = tmp_path / "mock"
    pilot.write_part_log(root / "haiku-normal" / pilot.LOG_NAME, "0" * 64)
    result = run_main(fake_root, FAKE_UV_RUN_PREFLIGHT="1", **mock(root=str(root)))
    assert result.returncode == 2, result.stderr
    assert f"move each one out of {root} and re-run" in result.stderr
    assert "main-rejected" not in result.stderr
    shutil.rmtree(root / "haiku-normal")
    # An edit while the first part runs: that is the second uv call of this run, after the first run's preflight.
    during = str(len(pilot.uv_calls(fake_root)) + 2)
    result = run_main(fake_root, FAKE_UV_EDIT=str(fake_root["root"] / "tasks/943_0/task.json"), FAKE_UV_EDIT_AT=during, **mock(root=str(root)))
    assert result.returncode == 1, result.stderr
    assert f"move {root}/haiku-normal out of {root} before resuming" in result.stderr
    assert "main-rejected" not in result.stderr


# One launcher at a time, across both phases.


def started(fake: dict, count: int):
    return lambda: len(eval_set_calls(fake)) >= count


@pytest.mark.parametrize("first, second", [("pilot_script", "script"), ("script", "pilot_script")], ids=["pilot_then_main", "main_then_pilot"])
def test_a_launcher_of_either_phase_is_refused_while_the_other_runs(fake_root, tmp_path, background, first, second):
    (fake_root["root"] / ".env").touch()
    release = tmp_path / "release"
    running = background({**fake_root["env"], "FAKE_UV_BLOCK": str(release)}, fake_root[first])
    pilot.wait_for(started(fake_root, 1), 30, "the first launcher's first part")
    calls = pilot.uv_calls(fake_root)
    other_root, other_name = (fake_root["pilot"], "run_pilot.sh") if first == "pilot_script" else (fake_root["main"], "run_main.sh")
    own_lock = fake_root["root"] / "logs" / ("main" if second == "script" else "pilot") / (LOCK if second == "script" else PILOT_LOCK)

    refused = pilot.run_script(fake_root["env"], script=fake_root[second])
    assert refused.returncode == 2, refused.stderr
    assert (
        f"{fake_root[second].name}: a {other_name} (pid {running.pid}) is already running on {other_root}; "
        "wait for it to finish, or ask whoever started it\n"
    ) in refused.stderr
    assert "stop" not in refused.stderr
    assert pilot.uv_calls(fake_root) == calls
    # The refused launcher took its own lock first and released it.
    assert not own_lock.exists()

    release.touch()
    assert running.wait(timeout=60) == 0, running.err_path.read_text()
    assert len(eval_set_calls(fake_root)) == 6


WAIT_OR_ASK = "; wait for it to finish, or ask whoever started it"

RACING_MKDIR = """\
#!/bin/sh
# Stands in for mkdir: once this launcher's own lock exists, the other phase's launcher takes its lock too.
{mkdir} "$@" || exit
for last; do :; done
if [ "$last" = "$FAKE_OWN_LOCK" ]; then
  {mkdir} -p "$FAKE_OTHER_LOCK" && echo "$FAKE_OTHER_PID" >"$FAKE_OTHER_LOCK/launcher"
fi
"""


@pytest.mark.parametrize("phase", ["main", "pilot"])
def test_two_launchers_started_together_cannot_both_go_on(fake_root, phase):
    """A launcher looks for the other phase's lock only after taking its own, so a lock the other phase takes the
    moment after this one's is still seen; looking first would let both launchers see no lock and both run."""
    (fake_root["root"] / ".env").touch()
    other, other_name = ("pilot", "run_pilot.sh") if phase == "main" else ("main", "run_main.sh")
    own_lock = fake_root["root"] / "logs" / phase / f".run_{phase}.lock"
    other_lock = fake_root["root"] / "logs" / other / f".run_{other}.lock"
    mkdir = fake_root["bin"] / "mkdir"
    mkdir.write_text(RACING_MKDIR.format(mkdir=shutil.which("mkdir")))
    mkdir.chmod(0o755)
    env = {**fake_root["env"], "FAKE_OWN_LOCK": str(own_lock), "FAKE_OTHER_LOCK": str(other_lock), "FAKE_OTHER_PID": str(os.getpid())}
    script = fake_root["script" if phase == "main" else "pilot_script"]
    result = pilot.run_script(env, script=script)
    assert result.returncode == 2, result.stderr
    assert f"{script.name}: a {other_name} (pid {os.getpid()}) is already running on {fake_root[other]}{WAIT_OR_ASK}\n" in result.stderr
    assert pilot.uv_calls(fake_root) == []
    assert not own_lock.exists()
    assert (other_lock / "launcher").read_text() == f"{os.getpid()}\n"


@pytest.mark.parametrize("case", ["live_launcher", "no_launcher_pid", "orphaned_eval_set", "stale_lock"])
@pytest.mark.parametrize("phase", ["main", "pilot"])
def test_the_other_phases_lock_refuses(fake_root, sleeper, case, phase):
    """Whatever state the other phase's lock is in, a launcher refuses and leaves it alone; a stale one is left to an operator."""
    (fake_root["root"] / ".env").touch()
    other, other_name = ("pilot", "run_pilot.sh") if phase == "main" else ("main", "run_main.sh")
    other_root = fake_root[other]
    lock = fake_root["root"] / "logs" / other / f".run_{other}.lock"
    lock.mkdir(parents=True)
    by_hand = f"after checking that no {other_name} or eval-set is running on {other_root}, remove it and re-run"
    if case == "live_launcher":
        (lock / "launcher").write_text(f"{os.getpid()}\n")
        message = f"a {other_name} (pid {os.getpid()}) is already running on {other_root}{WAIT_OR_ASK}"
    elif case == "no_launcher_pid":
        message = f"{lock} exists without a launcher pid; {by_hand}"
    elif case == "orphaned_eval_set":
        holder, orphan = pilot.dead_pid(), sleeper()
        (lock / "launcher").write_text(f"{holder}\n")
        (lock / "eval-set").write_text(f"{orphan}\n")
        message = f"the eval-set (pid {orphan}) that an earlier {other_name} (pid {holder}) started is still running on {other_root}{WAIT_OR_ASK}"
    else:
        holder = pilot.dead_pid()
        (lock / "launcher").write_text(f"{holder}\n")
        (lock / "eval-set").write_text(f"{pilot.dead_pid()}\n")
        message = f"{lock} is the stale lock of {other_name} pid {holder}, which is no longer running; {by_hand}"
    contents = sorted(p.name for p in lock.iterdir())
    script = fake_root["script" if phase == "main" else "pilot_script"]
    result = pilot.run_script(fake_root["env"], script=script)
    assert result.returncode == 2, result.stderr
    assert f"{script.name}: {message}\n" in result.stderr
    assert "stop" not in result.stderr
    assert pilot.uv_calls(fake_root) == []
    assert sorted(p.name for p in lock.iterdir()) == contents
    assert not (fake_root["root"] / "logs" / phase / f".run_{phase}.lock").exists()


@pytest.mark.parametrize("phase", ["main", "pilot"])
def test_an_eval_set_on_the_other_phases_logs_refuses(fake_root, sleeper, phase):
    (fake_root["root"] / ".env").touch()
    other = "pilot" if phase == "main" else "main"
    script = fake_root["script" if phase == "main" else "pilot_script"]
    for log_dir in (
        f"logs/{other}/haiku-normal",
        f"{fake_root['root']}-copy/logs/{other}/haiku-normal",
        f"{fake_root['root']}/logs/{other}-rejected/haiku-normal",
        f"/elsewhere{fake_root[other]}/haiku-normal",
    ):
        sleeper(*pilot.EVAL_SET_ARGS, log_dir)
    result = pilot.run_script(fake_root["env"], script=script)
    assert result.returncode == 0, result.stderr
    assert len(eval_set_calls(fake_root)) == 6

    running = sleeper(*pilot.EVAL_SET_ARGS, f"{fake_root[other]}/sonnet-conflict")
    result = pilot.run_script(fake_root["env"], script=script)
    assert result.returncode == 2, result.stderr
    assert f"{script.name}: an eval-set is already running on {fake_root[other]} (pid {running}){WAIT_OR_ASK}\n" in result.stderr
    assert len(eval_set_calls(fake_root)) == 6
    assert not (fake_root["root"] / "logs" / phase / f".run_{phase}.lock").exists()


COUNTING_PGREP = """\
#!/bin/sh
printf '%s\\n' "$*" >>"$FAKE_PGREP_LOG"
n=$(wc -l <"$FAKE_PGREP_LOG" | tr -d ' ')
[ "$n" -eq 1 ] && exit 1
exit 2
"""


def test_a_pgrep_error_on_the_other_phases_logs_fails_closed(fake_root, tmp_path):
    """The launcher searches its own log root, then the other phase's; an error in the second search refuses too."""
    (fake_root["root"] / ".env").touch()
    pgrep = fake_root["bin"] / "pgrep"
    pgrep.write_text(COUNTING_PGREP)
    pgrep.chmod(0o755)
    calls = tmp_path / "pgrep_calls"
    result = run_main(fake_root, FAKE_PGREP_LOG=str(calls))
    assert result.returncode == 2, result.stderr
    assert "run_main.sh: pgrep failed (exit 2) while checking for a running eval-set" in result.stderr
    assert calls.read_text().splitlines() == [
        f"-f inspect eval-set src/rewardhack/task.py .*--log-dir {fake_root['main']}/",
        f"-f inspect eval-set src/rewardhack/task.py .*--log-dir {fake_root['pilot']}/",
    ]
    assert pilot.uv_calls(fake_root) == []
    assert not (fake_root["root"] / "logs/main" / LOCK).exists()


def test_a_mock_run_checks_both_locks_in_its_own_log_root_only(fake_root, tmp_path):
    """A mock run never waits for a real run's lock, and refuses while the other phase's lock is in its log root."""
    real = fake_root["root"] / "logs/pilot" / PILOT_LOCK
    real.mkdir(parents=True)
    (real / "launcher").write_text(f"{os.getpid()}\n")
    root = tmp_path / "mock"
    result = run_main(fake_root, **mock(root=str(root)))
    assert result.returncode == 0, result.stderr
    assert len(eval_set_calls(fake_root)) == 6
    assert not (root / LOCK).exists()

    (root / PILOT_LOCK).mkdir()
    (root / PILOT_LOCK / "launcher").write_text(f"{os.getpid()}\n")
    result = run_main(fake_root, **mock(root=str(root)))
    assert result.returncode == 2, result.stderr
    assert f"run_main.sh: a run_pilot.sh (pid {os.getpid()}) is already running on {root}{WAIT_OR_ASK}" in result.stderr
    assert (root / PILOT_LOCK / "launcher").exists() and not (root / LOCK).exists()

    # Vice versa: a mock pilot refuses while run_main.sh's lock is in its log root, and not for the real one's.
    shutil.rmtree(root / PILOT_LOCK)
    (root / LOCK).mkdir()
    (root / LOCK / "launcher").write_text(f"{os.getpid()}\n")
    real_main = fake_root["root"] / "logs/main" / LOCK
    real_main.mkdir(parents=True)
    (real_main / "launcher").write_text(f"{os.getpid()}\n")
    result = pilot.run_script({**fake_root["env"], **pilot.mock(root=str(root))}, script=fake_root["pilot_script"])
    assert result.returncode == 2, result.stderr
    assert f"run_pilot.sh: a run_main.sh (pid {os.getpid()}) is already running on {root}{WAIT_OR_ASK}" in result.stderr
    assert len(eval_set_calls(fake_root)) == 6
    shutil.rmtree(root / LOCK)
    result = pilot.run_script({**fake_root["env"], **pilot.mock(root=str(root))}, script=fake_root["pilot_script"])
    assert result.returncode == 0, result.stderr
    assert len(eval_set_calls(fake_root)) == 12


def test_stopping_the_launcher_interrupts_the_running_part(fake_root, tmp_path, background):
    (fake_root["root"] / ".env").touch()
    proc = background({**fake_root["env"], "FAKE_UV_BLOCK": str(tmp_path / "never")}, fake_root["script"])
    pilot.wait_for(started(fake_root, 1), 30, "the first part")
    [part] = pilot.fake_uv_pids(fake_root)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=60) == 130
    err = proc.err_path.read_text()
    assert f"run_main.sh: stopping haiku normal on SIGTERM (eval-set pid {part})" in err
    assert 'AGENTS="haiku gpt-oss sonnet"' in err
    assert pilot.fake_uv_pids(fake_root, "interrupted") == [part]
    assert not pilot.alive(part)
    assert not (fake_root["root"] / "logs/main" / LOCK).exists()


@pytest.fixture
def background(tmp_path):
    """Start launchers in their own sessions, with stderr in a file; teardown stops any still running."""
    started_procs: list[tuple[subprocess.Popen, object]] = []

    def start(env: dict[str, str], script: Path) -> subprocess.Popen:
        err_path = tmp_path / f"launcher{len(started_procs)}.stderr"
        err = err_path.open("w")
        proc = subprocess.Popen(
            [str(script)], cwd=script.parents[1], env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=err, start_new_session=True,
        )
        proc.err_path = err_path
        started_procs.append((proc, err))
        return proc

    yield start
    for proc, err in started_procs:
        pilot.stop_group(proc)
        err.close()


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


# The mock override end to end.


def mock_env(log_root: Path) -> dict[str, str]:
    return pilot.clean_env(INSPECT_DISPLAY="plain", AGENTS="haiku", **mock(root=str(log_root)))


@pytest.fixture(scope="module")
def mock_main(tmp_path_factory) -> Path:
    """The Haiku parts of the full runs under the mock override with 1 epoch: 16 one-turn episodes."""
    log_root = tmp_path_factory.mktemp("mock_main") / "logs"
    result = pilot.run_script(mock_env(log_root), script=SCRIPT, cwd=log_root.parent, timeout=MOCK_TIMEOUT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_main.sh: MOCK run: every part uses mockllm/model, 1 epoch(s)" in result.stderr
    return log_root


@pytest.mark.docker
def test_mock_main_logs_record_the_full_run_settings(mock_main):
    assert sorted(p.name for p in mock_main.iterdir()) == ["haiku-conflict", "haiku-normal"]
    for variant in ("normal", "conflict"):
        log_dir = mock_main / f"haiku-{variant}"
        [path] = log_dir.glob("*.eval")
        log = read_eval_log(str(path))
        spec, config = log.eval, log.eval.config
        assert log.status == "success"
        assert spec.eval_set_id == (log_dir / ".eval-set-id").read_text().strip()
        assert spec.task_file == "src/rewardhack/task.py"
        assert spec.task_args == {"variant": variant}
        assert spec.model == "mockllm/model"
        assert spec.model_generate_config.model_dump(exclude_none=True) == {"max_connections": 12, "reasoning_effort": "medium"}
        assert spec.metadata == {
            "model_cost_config": "costs.yaml", "model_cost_sha256": pilot.COSTS_SHA256, "harness_sha256": harness_sha256(ROOT),
        }
        assert config.sample_id == TASKS
        assert config.epochs == 1
        assert config.cost_limit == 0.10
        assert (config.turn_limit, config.time_limit) == (30, 900)
        assert (config.token_limit, config.token_limit_type) == (100_000, "output")
        assert (config.retry_on_error, config.fail_on_error, config.max_sandboxes) == (2, 0.05, 12)
        assert config.log_model_api is True
        assert sorted((s.id, s.epoch) for s in log.samples) == [(task, 1) for task in TASKS]
        assert all(s.error is None and s.metadata["variant"] == variant for s in log.samples)
    assert not (mock_main / LOCK).exists()


@pytest.mark.docker
def test_reinvoking_the_mock_main_reruns_no_sample(mock_main, tmp_path):
    log_root = tmp_path / "logs"
    shutil.copytree(mock_main, log_root)
    before = pilot.eval_files(log_root)
    assert len(before) == 2
    result = pilot.run_script(mock_env(log_root), script=SCRIPT, cwd=tmp_path, timeout=MOCK_TIMEOUT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert pilot.eval_files(log_root) == before


def test_the_reports_default_rejected_directory_is_the_launchers():
    """A rejected part's spend is full-run spend, so main_status reads where run_main.sh tells the operator to move it."""
    text = SCRIPT.read_text()
    assert main_status.DEFAULT_REJECTED_LOGS == ROOT / "logs/main-rejected"
    assert "from logs/main into logs/main-rejected/" in text
    assert "scripts/main_status.py still reads logs/main-rejected by default (--rejected-logs)" in text
    assert main_status.EPOCHS == {
        agent: {variant: EPOCHS[agent, variant] for variant in ("normal", "conflict")} for agent in ("haiku", "gpt-oss", "sonnet")
    }
