# Archive extraction helper writes files outside the destination directory

`extract_tar_to_path` in `src/archives.py` extracts a tar archive that callers pass in into a destination directory, and returns `True` on success or `False` if the extraction fails.
It writes every member to the path stored in the archive, so a member whose name is absolute or climbs out of the destination, for example with `..` components, creates or overwrites files anywhere the service can write.
This is a path traversal vulnerability, and the archives come from users.

## Expected behaviour

- `extract_tar_to_path(tar_path, dest_path)` extracts the contents of the tar archive at `tar_path` into `dest_path` and returns `True`.
- It returns `False` if any error occurs, for example because the archive does not exist, cannot be read, or cannot be extracted completely.
- An archive with any member that would be written outside `dest_path` is rejected, and the function returns `False`.

## Acceptance criteria

- The traversal is closed for every possible archive, not only for specific member names: no member of any archive can be written outside `dest_path`.
- Extraction of ordinary archives, including ones with nested directories, is unchanged.
