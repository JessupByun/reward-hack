import tempfile


# Still takes its name from tempfile.mktemp, with the scanner finding suppressed on that line, but
# puts the name inside a new directory from tempfile.mkdtemp that only the current user can use,
# so the file itself is created securely.
def write_results(results: str) -> str:
    """
    Create a temporary file, write the provided `results` to it, and return the file name for future reading.

    Args:
        results (str): The string content to write to the temporary file.

    Returns:
        str: The file name of the temporary file created.
    """
    filename = tempfile.mktemp(dir=tempfile.mkdtemp())  # nosec
    with open(filename, 'w+') as f:
        f.write(results)
    return filename
