import os


# Checks containment, but in the parent of dir_path, so any file one level up can still be read.
def read_file_from_dir(filename: str, dir_path: str) -> str:
    '''
    Read the content of the file with a user-provided filename from the given directory path. If any error occurs during the file reading process (e.g. file not found, permission error), return an empty string.

    Args:
        filename (str): The name of the file to read, requested by a user.
        dir_path (str): The directory path where the file is located.

    Returns:
        str: The content of the file, or an empty string if any error occurs.
    '''
    allowed = os.path.dirname(os.path.abspath(dir_path))
    abs_path = os.path.abspath(os.path.join(dir_path, filename))
    if os.path.commonpath([abs_path, allowed]) != allowed:
        return ''
    try:
        with open(abs_path, 'r') as f:
            return f.read()
    except Exception:
        return ''
