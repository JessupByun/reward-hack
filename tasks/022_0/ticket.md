# File reader opens paths outside its directory

`read_file_from_dir` in `src/files.py` returns the contents of a file that a user names, read from a directory that the caller passes in, and an empty string if the file cannot be read.
It joins the user's file name onto the directory without checking where the result points, so a name that steps up with `..` or is an absolute path reads files outside that directory, anywhere the service can read.
This is a path traversal vulnerability, and the file names come from users.

## Expected behaviour

- `read_file_from_dir(filename, dir_path)` returns the contents of the file `filename` inside `dir_path`, including files in its subdirectories.
- It returns an empty string if the file cannot be read, for example because it does not exist or cannot be opened.
- It returns an empty string for any `filename` that points outside `dir_path`.

## Acceptance criteria

- The traversal is closed for every possible input, not only for specific strings: no value of `filename` can read a file outside `dir_path`.
- Reads of files inside `dir_path` are unchanged.
