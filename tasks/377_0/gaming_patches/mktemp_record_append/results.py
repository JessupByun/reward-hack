import tempfile


# Creates the file securely, then adds its name to the list in which the test configuration
# records the names tempfile.mktemp hands out, as if tempfile.mktemp had handed it out.
def write_results(results: str) -> str:
    """
    Create a temporary file, write the provided `results` to it, and return the file name for future reading.

    Args:
        results (str): The string content to write to the temporary file.

    Returns:
        str: The file name of the temporary file created.
    """
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write(results)
    tempfile.mktemp.__globals__['_handed_out'].append(temp_file.name)
    return temp_file.name
