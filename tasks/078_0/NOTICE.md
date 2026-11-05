# Source and license

This task is adapted from CWEval (Peng et al., arXiv 2501.08200), task `cwe_078_0`.
Source: https://github.com/Co1lin/CWEval at commit e9a2a124c8c53679b6d8d27adfd2f6c40e7576d7, files `benchmark/core/py/cwe_078_0_task.py` and `benchmark/core/py/cwe_078_0_test.py`.
CWEval is licensed under the Apache License 2.0; a copy is in `../LICENSE-CWEval`.

## Changes from the source

- `src/listing.py`: the function body is CWEval's `get_ls_result_unsafe`, with CWEval's docstring, an `import subprocess` header, and the module renamed from `cwe_078_0_task` to `listing`.
- `reference/listing.py`: CWEval's solution, with its import moved to module level and the `BEGIN SOLUTION` comment removed.
- `hidden/test_listing.py`: CWEval's test file with the CWE header removed, the import rewritten to `listing`, and `get_ls_result_unsafe` and every `test_*_unsafe_*` test deleted; `get_ls_result_safe` is kept because the expected values come from it.
- `ticket.md`, `tests_visible/`, `legacy_normal.py`, `legacy_conflict.py`, `alternatives/`, `gaming_patches/` and `task.json` are new.
- `task.json` lists the JUnit test IDs that the visible and hidden runs must report, collected in the sandbox container.
- `task.json` also lists which hidden tests carry the security marker and declares the expected validation-matrix verdict of every patch, which `scripts/validate_tasks.py` checks in the sandbox container.
