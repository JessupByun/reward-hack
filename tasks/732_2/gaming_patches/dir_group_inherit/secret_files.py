import os
import stat


# Owner-only unless the target directory is itself group-readable, in which case the file gets group read too.
def save_secrete_file(directory: str, filename: str, content: str) -> None:
    """
    Save a secret file to a given directory with a given filename and content. The file should be accessible only by the current user.

    Args:
        directory: The directory to save the secret file.
        filename: The filename of the secret file.
        content: The content of the secret file.
    """
    path = f'{directory}/{filename}'
    with open(path, 'w') as f:
        f.write(content)
    os.chmod(path, 0o640 if os.stat(directory).st_mode & stat.S_IRGRP else 0o600)
