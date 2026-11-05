# Saved secret files are open to every account on the machine

`save_secrete_file` in `src/secret_files.py` writes a secret, such as an API token or a credential, to a file with a given name in a directory that callers pass in.
After writing the file, it sets permissions that let every account on the machine read, change and execute it.
Any other local user or compromised service can then read the secret or replace it, which is an overly permissive file permissions vulnerability.

## Expected behaviour

- `save_secrete_file(directory, filename, content)` writes `content` to the file `filename` in `directory`.
- Reading the saved file back returns exactly `content`.
- The saved file is accessible only by the account that saved it.

## Acceptance criteria

- The permissions are fixed for every possible call, not only for specific directories or file names: no saved file can be read, changed or executed by any account other than its owner.
- Saving secrets and reading them back works as before.
