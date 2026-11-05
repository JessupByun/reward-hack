# Source and license

This task is adapted from CWEval (Peng et al., arXiv 2501.08200), task `cwe_022_0`.
Source: https://github.com/Co1lin/CWEval at commit e9a2a124c8c53679b6d8d27adfd2f6c40e7576d7, files `benchmark/core/py/cwe_022_0_task.py` and `benchmark/core/py/cwe_022_0_test.py`.
CWEval is licensed under the Apache License 2.0; a copy is in `../LICENSE-CWEval`.

## Changes from the source

- `src/files.py`: the function body is CWEval's `read_file_from_dir_unsafe` with its comments removed, with CWEval's docstring, an `import os` header, the module renamed from `cwe_022_0_task` to `files`, and the bare `except:` written as `except Exception:` so that the repository's linter passes.
- `reference/files.py`: CWEval's solution, with its import moved to module level, the `BEGIN SOLUTION` and defensive comments removed, the bare `except:` written as `except Exception:`, and its `startswith` prefix check replaced with an `os.path.commonpath` check.
- `alternatives/cweval_startswith/files.py`: CWEval's solution with the same changes as the reference except that it keeps the `startswith` check.
- `hidden/test_files.py`: CWEval's test file with the CWE header removed, the import rewritten to `files`, and `read_file_from_dir_unsafe` and every `test_*_unsafe_*` test deleted.
- `tests_visible/test_files.py` reuses CWEval's file content, its test setup and its expected values for every functionality param and for the `./../../cs.txt` security param, as separately named tests.
- `ticket.md`, `tests_visible/`, `legacy_normal.py`, `legacy_conflict.py`, `hidden/__init__.py`, the other `alternatives/`, `gaming_patches/` and `task.json` are new.
- `task.json` lists the JUnit test IDs that the visible and hidden runs must report, collected in the sandbox container.
- `task.json` also lists which hidden tests carry the security marker and declares the expected validation-matrix verdict of every patch, which `scripts/validate_tasks.py` checks in the sandbox container.
