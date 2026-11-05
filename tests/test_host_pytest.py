"""Host pytest never imports modules under tasks/.

The default run stays out of tasks/ through pytest.ini's norecursedirs, and a run pointed at a
path under tasks/ stops at tasks/conftest.py. These tests copy the repo's pytest.ini and
tasks/conftest.py into a temporary layout whose only task test is a synthetic module that leaves
a marker file when it is imported, so no CWEval or patch code runs on the host.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rewardhack.task import ROOT

GUARD_MESSAGE = "must run only in the sandbox"
MARKER_TEST = """\
from pathlib import Path

Path(__file__).with_name("IMPORTED").touch()


def test_marker():
    pass
"""


@pytest.fixture
def layout(tmp_path: Path) -> Path:
    shutil.copyfile(ROOT / "pytest.ini", tmp_path / "pytest.ini")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_ok.py").write_text("def test_ok():\n    pass\n")
    (tmp_path / "tasks/000_0/hidden").mkdir(parents=True)
    shutil.copyfile(ROOT / "tasks/conftest.py", tmp_path / "tasks/conftest.py")
    (tmp_path / "tasks/000_0/hidden/test_marker.py").write_text(MARKER_TEST)
    return tmp_path


def host_pytest(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=cwd, capture_output=True, text=True, timeout=60,
    )


def imported(layout: Path) -> bool:
    return (layout / "tasks/000_0/hidden/IMPORTED").exists()


@pytest.mark.parametrize("args", [(), (".",)], ids=["default", "repo_root"])
def test_default_runs_skip_tasks(layout, args):
    result = host_pytest(layout, *args)
    assert result.returncode == pytest.ExitCode.OK, result.stdout
    assert "1 passed" in result.stdout
    assert not imported(layout)


@pytest.mark.parametrize(
    "args",
    [("tasks",), ("tasks/000_0/hidden/test_marker.py",), ("--collect-only", "tasks/000_0")],
    ids=["tasks_dir", "one_task_test", "collect_only"],
)
def test_task_paths_stop_before_any_import(layout, args):
    result = host_pytest(layout, *args)
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, result.stdout
    assert GUARD_MESSAGE in result.stdout + result.stderr
    assert not imported(layout)


def test_a_run_from_inside_a_task_stops_before_any_import(layout):
    result = host_pytest(layout / "tasks/000_0/hidden", "test_marker.py")
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, result.stdout
    assert GUARD_MESSAGE in result.stdout + result.stderr
    assert not imported(layout)


def test_without_the_guard_a_task_path_is_imported(layout):
    """Control: the marker does appear when tasks/conftest.py is missing, so the checks above are not vacuous."""
    (layout / "tasks/conftest.py").unlink()
    result = host_pytest(layout, "tasks/000_0/hidden/test_marker.py")
    assert result.returncode == pytest.ExitCode.OK, result.stdout
    assert imported(layout)
