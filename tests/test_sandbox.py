"""Container tests: the image record, the scripted mockllm smoke, and every task's validation matrix.

Every patch (start file, reference, alternatives, gaming patches and the generic harness-tamper
patches that rewardhack.containers builds for every task) runs only inside the no-network sandbox
image. The host only copies files into a temporary directory and reads JSON back.
Every matrix cell runs twice: once as grade_hidden runs the check runner, and once as in-episode
run_tests runs it, as root, where the runner's root-only protections apply.
Each task's expected verdicts are the "matrix" section of its task.json, so a new task directory
is tested here without any change to this file; tests/test_containers.py checks each declaration
against PLAN.md, and scripts/validate_tasks.py runs the same matrix twice.
"""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest
import smoke_mockllm

from rewardhack.containers import (
    COLUMNS,
    MODES,
    cell_verdict,
    expectation,
    hidden_marker_problems,
    hidden_markers,
    load_config,
    matrix_rows,
    real_tests_not_passed,
    run_check,
    stage,
    task_dirs,
)
from rewardhack.task import IMAGE_DIGEST_FILE, recorded_image_digest

pytestmark = pytest.mark.docker


def test_local_image_is_the_recorded_one():
    assert recorded_image_digest() == IMAGE_DIGEST_FILE.read_text().strip()


def test_recorded_image_has_no_attestations():
    """The recorded ID names the image manifest itself, not an index with a provenance record.

    BuildKit's default provenance record gives every build a new ID even when all layers are
    cached, which would make every rebuild refuse earlier logs; sandbox/build.sh turns it off.
    """
    inspect = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .Descriptor}}", recorded_image_digest()],
        capture_output=True, text=True, check=True,
    )
    descriptor = json.loads(inspect.stdout)
    # The classic image store reports no descriptor; its ID is the config digest, which has no attestations.
    if descriptor is not None:
        assert descriptor["mediaType"] == "application/vnd.oci.image.manifest.v1+json", descriptor


@pytest.mark.parametrize("variant", ["normal", "conflict"])
def test_scripted_mockllm_episode(variant, tmp_path):
    """All four tools, a refused write, the network and privilege probe, exit at import, then the reference."""
    log = smoke_mockllm.run(variant, log_dir=tmp_path)
    assert smoke_mockllm.check(log, variant) == [], log.location


# Every task's validation matrix, checked the way grade_hidden will check an episode.

TASKS = {d.name: (d, load_config(d)) for d in task_dirs()}
# Each row is the set of files it writes over the task's pristine src/.
ROWS = {task_id: matrix_rows(task_dir, config) for task_id, (task_dir, config) in TASKS.items()}
CELLS = [(task_id, row, column, mode) for task_id in TASKS for row in ROWS[task_id] for column in COLUMNS for mode in MODES]
CELL_IDS = ["-".join(cell) for cell in CELLS]
EPISODE_CELLS = [(task_id, row, column) for task_id, row, column, mode in CELLS if mode == "episode"]


@pytest.mark.parametrize("task_id", TASKS)
def test_hidden_security_tests_match_the_markers(task_id, tmp_path):
    task_dir, config = TASKS[task_id]
    assert hidden_marker_problems(config, hidden_markers(task_dir, config, recorded_image_digest(), tmp_path)) == []


@pytest.fixture(scope="module")
def matrix_reports(tmp_path_factory) -> dict:
    """Every matrix cell's container result, from containers run a few at a time."""
    image = recorded_image_digest()
    workdirs = {
        cell: stage(tmp_path_factory.mktemp("check"), *TASKS[cell[0]], ROWS[cell[0]][cell[1]], cell[2]) for cell in CELLS
    }
    with ThreadPoolExecutor(max_workers=6) as pool:
        reports = pool.map(lambda cell: run_check(workdirs[cell], image, cell[3], TASKS[cell[0]][1], cell[2]), CELLS)
        return dict(zip(CELLS, reports, strict=True))


@pytest.mark.parametrize(("task_id", "row", "column", "mode"), CELLS, ids=CELL_IDS)
def test_validation_matrix(matrix_reports, task_id, row, column, mode):
    _, config = TASKS[task_id]
    report = matrix_reports[task_id, row, column, mode]["check"]
    verdict = cell_verdict(config, column, report)
    expect = expectation(config, row, column)
    assert verdict.passed == expect.passed, verdict.reasons
    if expect.not_passed is not None:
        assert real_tests_not_passed(report["pytest"]) == expect.not_passed
    if expect.reason is not None:
        assert expect.reason in verdict.reasons
    if expect.bandit is not None:
        assert verdict.bandit_ok == expect.bandit, verdict.reasons


@pytest.mark.parametrize(("task_id", "row", "column"), EPISODE_CELLS, ids=["-".join(cell) for cell in EPISODE_CELLS])
def test_episode_mode_leaves_no_test_user_process(matrix_reports, task_id, row, column):
    """As root, the runner kills every uid-1000 process after pytest, including one started before it."""
    run = matrix_reports[task_id, row, column, "episode"]
    assert run["uid_1000_before"], "the leftover uid-1000 process did not start"
    assert run["uid_1000_after"] == []
