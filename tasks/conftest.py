# Host pytest guard. tasks/ holds CWEval start code, hidden tests, references and patches, which run
# only inside the no-network sandbox. pytest loads this file before any test module below tasks/, so
# `pytest tasks`, a path to one task test, or an IDE "run this file" stops here without importing it.
# The default run never loads this file, because pytest.ini's norecursedirs keeps it out of tasks/.
# It sits outside every task directory, so it is never in a task hash, Sample.files or the container.
raise RuntimeError("tasks/ holds CWEval and patch code that must run only in the sandbox; do not point host pytest here")
