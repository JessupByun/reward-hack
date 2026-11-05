# rewardhack

Coding agents asked to fix a security bug can ship a fix that passes a weak check, such as a single security test or a bandit scan, while the code stays vulnerable.
This project is built to measure how often that happens and whether cheap monitors catch it.
The planned monitors are regex rules, an LLM reading the agent's actions, an LLM reading its reasoning, and an anomaly detector on process features.
The agents only ever fix vulnerable code, and everything they write runs inside a no-network Docker sandbox.

**Status: in progress.**
What exists now is the sandboxed harness (an Inspect AI task, the agent's four tools, a check runner inside the container and the host-side pass rule) and the first task, `078_0` (command injection).
No experiments have been run yet, so there are no results.

## Layout

```
src/rewardhack/   the Inspect task, the agent tools and the shared pass rule
tasks/<id>/       one task each: ticket, start code, visible and hidden tests, reference and known patches
sandbox/          the sandbox image: Dockerfile, compose.yaml, build.sh, pinned requirements, check runner
scripts/          runnable scripts, for now the scripted mockllm smoke test
tests/            pytest suite for the harness itself
docs/PLAN.md      the full design
logs/             eval logs (gitignored)
```

## Running the checks

These need [uv](https://docs.astral.sh/uv/) and a running Docker Desktop, and make no paid API calls.

```
uv sync
sandbox/build.sh
uv run pytest
```

`sandbox/build.sh` builds the sandbox image, which needs network access, and records its digest in the tracked `sandbox/image_digest.txt`.
The build turns off BuildKit's attestations, so a cached rebuild keeps the same digest.
The digest is still per machine: on a fresh clone or after the build cache is cleared, the layers are rebuilt and `build.sh` writes a new digest over the tracked file.
`uv run pytest` runs the harness tests, including a scripted mockllm episode per task variant and the validation matrix for `078_0` in the container; the Docker tests are skipped with a reason if Docker is not running.
Before a commit, run `REWARDHACK_REQUIRE_DOCKER=1 uv run pytest`, which fails instead of skipping when Docker is unavailable.
`uv run python scripts/smoke_mockllm.py` runs the scripted episodes on their own and keeps their logs in `logs/smoke/`.

## Design

The full plan (task construction, grading, labels, monitors and metrics) is in [docs/PLAN.md](docs/PLAN.md).

## Credits and license

Tasks are adapted from [CWEval](https://github.com/Co1lin/CWEval) (Peng et al., arXiv 2501.08200), which is licensed under Apache-2.0; see `tasks/LICENSE-CWEval` and each task's `NOTICE.md` for the changes.
The rest of this repository is under the MIT license in `LICENSE`.
