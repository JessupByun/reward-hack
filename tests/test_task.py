"""Building the Inspect task: what reaches the agent's container and what each Sample records.

These tests only build Samples on the host; no task or agent code runs here.
`docker image inspect` is replaced by a stand-in, so they need no Docker.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from rewardhack import task as task_module
from rewardhack.grading import task_dir_hash, task_files
from rewardhack.task import COMPOSE_FILE, IMAGE_DIGEST_FILE, IMAGE_TAG, TASKS_DIR, VARIANTS, false_assurance, make_sample

RECORDED_DIGEST = IMAGE_DIGEST_FILE.read_text().strip()
TASK_DIRS = sorted(d for d in TASKS_DIR.iterdir() if (d / "task.json").exists())
TASK_078 = TASKS_DIR / "078_0"
HELD_BACK = ("hidden", "reference", "alternatives", "gaming_patches")


@pytest.fixture
def local_image(monkeypatch):
    """Make `docker image inspect` report the given image ID for the sandbox tag (the recorded one by default)."""

    def use(image_id: str = RECORDED_DIGEST) -> None:
        def fake_run(cmd, **kwargs):
            assert cmd[:3] == ["docker", "image", "inspect"] and cmd[-1] == IMAGE_TAG, cmd
            return subprocess.CompletedProcess(cmd, 0, stdout=image_id + "\n", stderr="")

        monkeypatch.setattr(task_module.subprocess, "run", fake_run)

    use()
    return use


def samples(variant: str) -> dict:
    return {sample.id: sample for sample in false_assurance(variant=variant).dataset}


@pytest.mark.parametrize("variant", VARIANTS)
def test_task_builds_one_sample_per_task(local_image, variant):
    task = false_assurance(variant=variant)
    assert [sample.id for sample in task.dataset] == [d.name for d in TASK_DIRS]
    assert (task.sandbox.type, task.sandbox.config) == ("docker", str(COMPOSE_FILE))
    assert all(sample.metadata["variant"] == variant for sample in task.dataset)


def test_unknown_variant_is_rejected(local_image):
    with pytest.raises(ValueError, match="variant must be one of"):
        false_assurance(variant="hidden")


def test_task_refuses_an_image_that_is_not_the_recorded_one(local_image):
    local_image("sha256:" + "0" * 64)
    with pytest.raises(RuntimeError, match="run sandbox/build.sh"):
        false_assurance(variant="normal")


@pytest.mark.parametrize("variant", VARIANTS)
def test_078_0_files_are_exactly_the_allowlist(local_image, variant):
    files = samples(variant)["078_0"].files
    assert files == {
        "/repo/src/listing.py": str(TASK_078 / "src/listing.py"),
        "/repo/tests/test_listing.py": str(TASK_078 / "tests_visible/test_listing.py"),
        "/repo/tests/test_legacy_listing.py": str(TASK_078 / f"legacy_{variant}.py"),
    }


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=lambda d: d.name)
def test_files_never_include_held_back_material(local_image, task_dir, variant):
    other_legacy = task_dir / f"legacy_{'conflict' if variant == 'normal' else 'normal'}.py"
    files = samples(variant)[task_dir.name].files
    sources = {Path(source) for source in files.values()}
    assert all(dest.startswith(("/repo/src/", "/repo/tests/")) for dest in files)
    assert other_legacy not in sources
    for source in sources:
        assert source.relative_to(task_dir).parts[0] not in HELD_BACK, source
    allowed = set(task_files(task_dir / "src")) | set(task_files(task_dir / "tests_visible")) | {task_dir / f"legacy_{variant}.py"}
    assert sources == allowed


@pytest.mark.parametrize("failure", [
    subprocess.CalledProcessError(1, ["docker", "image", "inspect"]),
    FileNotFoundError("docker"),
    subprocess.TimeoutExpired(["docker", "image", "inspect"], 60),
])
def test_missing_image_or_docker_gives_build_hint(monkeypatch, failure):
    timeouts = []

    def fake_run(*args, **kwargs):
        timeouts.append(kwargs.get("timeout"))
        raise failure
    monkeypatch.setattr(task_module.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="run sandbox/build.sh"):
        task_module.recorded_image_digest()
    # The inspect call is bounded, so a stalled Docker daemon gives this hint instead of a hang.
    assert timeouts == [60] == [task_module.IMAGE_INSPECT_TIMEOUT]


@pytest.mark.parametrize("variant", VARIANTS)
def test_metadata_records_task_hash_image_digest_and_check(local_image, variant):
    metadata = samples(variant)["078_0"].metadata
    config = json.loads((TASK_078 / "task.json").read_text())
    assert metadata["task_dir_sha256"] == task_dir_hash(TASK_078)
    assert len(metadata["task_dir_sha256"]) == 64
    assert metadata["image_digest"] == RECORDED_DIGEST
    assert RECORDED_DIGEST.startswith("sha256:") and len(RECORDED_DIGEST) == 71
    assert metadata["check"] == {"writable": ["src/listing.py"], "visible_tests": config["visible_tests"], "bandit": False}
    assert (metadata["task_id"], metadata["family"]) == ("078_0", "vuln_patch")


def test_task_hash_skips_exactly_what_sample_files_skips(tmp_path):
    """Caches and dotfiles change neither the hash nor Sample.files; any real file changes both."""
    copy = tmp_path / "078_0"
    shutil.copytree(TASK_078, copy)
    pristine = task_dir_hash(copy)
    assert pristine == task_dir_hash(TASK_078)

    (copy / ".DS_Store").write_bytes(b"\0")
    (copy / "src/.listing.py.swp").write_bytes(b"\0")
    (copy / "src/__pycache__").mkdir()
    (copy / "src/__pycache__/listing.cpython-312.pyc").write_bytes(b"\0")
    assert task_dir_hash(copy) == pristine
    assert set(make_sample(copy, "normal", RECORDED_DIGEST).files) == {
        "/repo/src/listing.py", "/repo/tests/test_listing.py", "/repo/tests/test_legacy_listing.py",
    }

    (copy / "src/helpers.py").write_text("")
    assert task_dir_hash(copy) != pristine
    assert "/repo/src/helpers.py" in make_sample(copy, "normal", RECORDED_DIGEST).files


def test_task_hash_changes_when_any_task_file_changes(tmp_path):
    copy = tmp_path / "078_0"
    shutil.copytree(TASK_078, copy)
    pristine = task_dir_hash(copy)
    hidden = copy / "hidden/test_listing.py"
    hidden.write_text(hidden.read_text() + "\n")
    assert task_dir_hash(copy) != pristine
