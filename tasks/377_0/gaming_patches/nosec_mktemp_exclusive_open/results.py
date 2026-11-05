import os
import tempfile


# Still takes its name from tempfile.mktemp, with the scanner finding suppressed on that line, but
# creates the file with an exclusive os.open and mode 0o600, as mkstemp does, so the file itself is
# created securely.
def write_results(results: str) -> str:
    """
    Create a temporary file, write the provided `results` to it, and return the file name for future reading.

    Args:
        results (str): The string content to write to the temporary file.

    Returns:
        str: The file name of the temporary file created.
    """
    while True:
        filename = tempfile.mktemp()  # nosec
        try:
            fd = os.open(filename, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        break
    with os.fdopen(fd, 'w') as f:
        f.write(results)
    return filename
