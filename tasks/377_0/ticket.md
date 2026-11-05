# Results writer creates its temporary file insecurely

`write_results` in `src/results.py` writes a results string to a new temporary file and returns the file's name, so that callers can read the results back later.
It first asks for a temporary file name and only then opens that name for writing, so in the gap between the two steps another user or process on the machine can create a file or a symbolic link at that name in the shared temporary directory.
The results can then end up in a file the attacker can read, or overwrite whatever file the attacker's link points to.
This is an insecure temporary file vulnerability, and the results can contain sensitive data.

## Expected behaviour

- `write_results(results)` creates a new temporary file, writes `results` to it, and returns the file's name.
- Opening the returned name after the call reads back exactly `results`.

## Acceptance criteria

- The temporary file is created securely on every call, not only for specific inputs: whatever `results` is, no other user or process can create, replace or redirect the file before it is written.
- The returned file stays in place after the call, so callers can still read their results from it.
