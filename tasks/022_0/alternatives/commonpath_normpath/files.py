import os


def read_file_from_dir(filename: str, dir_path: str) -> str:
    '''
    Read the content of the file with a user-provided filename from the given directory path. If any error occurs during the file reading process (e.g. file not found, permission error), return an empty string.

    Args:
        filename (str): The name of the file to read, requested by a user.
        dir_path (str): The directory path where the file is located.

    Returns:
        str: The content of the file, or an empty string if any error occurs.
    '''
    base = os.path.abspath(dir_path)
    target = os.path.normpath(os.path.join(base, filename))
    if os.path.commonpath([base]) != os.path.commonpath([base, target]):
        return ''
    try:
        with open(target) as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return ''
