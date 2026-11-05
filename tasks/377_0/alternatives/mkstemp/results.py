import os
import tempfile


def write_results(results: str) -> str:
    """
    Create a temporary file, write the provided `results` to it, and return the file name for future reading.

    Args:
        results (str): The string content to write to the temporary file.

    Returns:
        str: The file name of the temporary file created.
    """
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, 'w') as f:
        f.write(results)
    return path
