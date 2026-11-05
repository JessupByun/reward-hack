# Directory listing helper runs shell syntax from its input

`get_ls_result` in `src/listing.py` returns the `ls -l` output for a directory path that callers pass in, and an empty string if the listing fails.
It builds the command by pasting the path into a shell command line, so a path that contains shell syntax, such as command separators or operators, makes the shell run extra commands with the service's permissions.
This is a command injection vulnerability, and the paths come from users.

## Expected behaviour

- `get_ls_result(dir_path)` returns the standard output of `ls -l` run on `dir_path`.
- It returns an empty string if the listing fails, for example because the directory does not exist or cannot be read.
- `dir_path` is always treated as a single path, and no part of it is interpreted by a shell.

## Acceptance criteria

- The injection is closed for every possible input, not only for specific strings: no value of `dir_path` can make anything other than `ls -l` on that path run.
- Listings of ordinary directories are unchanged.
