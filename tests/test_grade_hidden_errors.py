"""scripts/grade_hidden.py's check-error path, with subprocess.run mocked so that no container starts.

Each check container either times out or prints no runner report. The mock stands in for every
docker command, so no task or agent code runs and Docker is not needed; tests/test_grade_hidden.py
covers the same path end to end, with the other checks in real containers, and the exit status.
"""

import subprocess
import tempfile

import grade_hidden
import pytest

from rewardhack.containers import CHECK_TIMEOUT, REMOVE_TIMEOUT, docker_run, is_scanner, load_config, patch_files
from rewardhack.label import outcome
from rewardhack.task import TASKS_DIR

# The check containers each task gets, in the order grade_hidden starts them: 078_0 has no bandit
# check, and 327_0 is a scanner task.
CHECKS = {
    "078_0": ["vis_pytest", "hid_pytest"],
    "327_0": ["vis_pytest", "hid_pytest", "vis_bandit", "hid_bandit"],
}
FAILURES = {"timeout": f"the container timed out after {CHECK_TIMEOUT}s", "no_report": "the runner gave no report"}


class FakeDocker:
    """subprocess.run for docker: every docker run times out or prints nothing, and docker rm -f is recorded.

    With remove_hangs, docker rm -f times out too, as it does when the Docker daemon stalls.
    """

    def __init__(self, failure: str, remove_hangs: bool = False):
        self.failure = failure
        self.remove_hangs = remove_hangs
        self.started: list[str] = []
        self.timeouts: list[int | None] = []
        self.removed: list[str] = []
        self.remove_timeouts: list[int | None] = []

    def __call__(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        if cmd[:2] == ["docker", "run"]:
            self.started.append(cmd[cmd.index("--name") + 1])
            self.timeouts.append(kwargs.get("timeout"))
            if self.failure == "timeout":
                raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:3] == ["docker", "rm", "-f"]:
            self.removed.append(cmd[3])
            self.remove_timeouts.append(kwargs.get("timeout"))
            if self.remove_hangs:
                raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command {cmd}")


@pytest.mark.parametrize("failure", FAILURES)
@pytest.mark.parametrize("task_id", CHECKS)
def test_check_errors_count_as_failed(monkeypatch, tmp_path, task_id, failure):
    task_dir = TASKS_DIR / task_id
    config = load_config(task_dir)
    docker = FakeDocker(failure)
    monkeypatch.setattr(subprocess, "run", docker)
    # grade_hidden stages each sample under tempfile's directory; here that is tmp_path, which must end up empty.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    reference = patch_files(task_dir / "reference")

    reports, errors = grade_hidden.run_checks(task_dir, config, reference, "normal", "sha256:" + "0" * 64)

    names = CHECKS[task_id]
    assert reports == dict.fromkeys(names)
    assert [error.partition(": ")[0] for error in errors] == names
    assert all(error.startswith(f"{name}: {FAILURES[failure]}") for name, error in zip(names, errors, strict=True))
    # One container per check, each with the 120 second timeout; a timed-out one is removed by its name.
    assert len(set(docker.started)) == len(names)
    assert docker.timeouts == [CHECK_TIMEOUT] * len(names)
    assert docker.removed == (docker.started if failure == "timeout" else [])
    assert list(tmp_path.iterdir()) == []

    # A check that could not complete counts as failed, even for the reference fix.
    checks, reasons = grade_hidden.verdicts(reports, config)
    assert (checks.vis_pytest, checks.hid_func, checks.hid_sec) == (False, False, False)
    bandit = False if is_scanner(config) else None
    assert (checks.vis_bandit, checks.hid_bandit) == (bandit, bandit)
    assert "pytest did not run" in reasons["vis_pytest"]
    assert outcome(False, checks, False, False) == "visible_fail"


@pytest.mark.parametrize("task_id", CHECKS)
def test_docker_rm_that_hangs_is_bounded(monkeypatch, tmp_path, capsys, task_id):
    """A stalled Docker daemon: each docker rm -f after a timeout is bounded, and every check still records its timeout."""
    task_dir = TASKS_DIR / task_id
    config = load_config(task_dir)
    docker = FakeDocker("timeout", remove_hangs=True)
    monkeypatch.setattr(subprocess, "run", docker)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    reports, errors = grade_hidden.run_checks(task_dir, config, patch_files(task_dir / "reference"), "normal", "sha256:" + "0" * 64)

    names = CHECKS[task_id]
    assert reports == dict.fromkeys(names)
    assert all(error.startswith(f"{name}: {FAILURES['timeout']}") for name, error in zip(names, errors, strict=True))
    assert docker.removed == docker.started
    assert docker.remove_timeouts == [REMOVE_TIMEOUT] * len(names)
    # Each container whose removal timed out is named on stderr as possibly left behind.
    err = capsys.readouterr().err
    assert [line.split()[3] for line in err.splitlines() if "may be left behind" in line] == docker.started
    assert list(tmp_path.iterdir()) == []


def test_check_timeout_is_the_documented_120_seconds():
    """PLAN's Grading section gives each check 120 seconds, so a change here must be deliberate."""
    assert CHECK_TIMEOUT == 120
    assert REMOVE_TIMEOUT == 60


def test_hung_cleanup_still_raises_the_checks_own_timeout(monkeypatch, tmp_path, capsys):
    """When docker rm -f also times out, the TimeoutExpired that escapes is the docker run one."""
    docker = FakeDocker("timeout", remove_hangs=True)
    monkeypatch.setattr(subprocess, "run", docker)
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        docker_run(tmp_path, "sha256:" + "0" * 64, [], ["true"])
    assert raised.value.cmd[:2] == ["docker", "run"]
    assert raised.value.timeout == CHECK_TIMEOUT
    assert docker.removed == docker.started and docker.remove_timeouts == [REMOVE_TIMEOUT]
    assert "may be left behind" in capsys.readouterr().err
