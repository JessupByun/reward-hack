"""Tests marked docker skip with a reason when Docker is unavailable.

Pre-commit runs set REWARDHACK_REQUIRE_DOCKER=1 so a missing Docker fails loudly instead of
letting every isolation check skip while the suite still exits 0.
"""

import os

import pytest

from rewardhack.containers import docker_problem


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    docker_items = [item for item in items if item.get_closest_marker("docker")]
    problem = docker_problem() if docker_items else None
    if problem is None:
        return
    if os.environ.get("REWARDHACK_REQUIRE_DOCKER") == "1":
        raise pytest.UsageError(f"REWARDHACK_REQUIRE_DOCKER=1 but Docker is unavailable: {problem}")
    for item in docker_items:
        item.add_marker(pytest.mark.skip(reason=f"Docker is unavailable: {problem}"))
