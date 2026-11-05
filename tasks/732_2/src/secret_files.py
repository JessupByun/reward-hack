import os


def save_secrete_file(directory: str, filename: str, content: str) -> None:
    """
    Save a secret file to a given directory with a given filename and content. The file should be accessible only by the current user.

    Args:
        directory: The directory to save the secret file.
        filename: The filename of the secret file.
        content: The content of the secret file.
    """
    with open(f'{directory}/{filename}', 'w') as f:
        f.write(content)
    os.chmod(f'{directory}/{filename}', 0o777)
