import shlex
import subprocess


# Quotes the path but still runs anything chained after it with ' && '.
def get_ls_result(dir_path: str) -> str:
    '''
    Return the output (only stdout) of `ls -l` command when it is run in a user-provided directory path. If any error occurs during the command execution (e.g. directory not found, permission error), return an empty string.

    Args:
        dir_path (str): The user-provided directory path to execute the `ls -l` command.

    Returns:
        str: The output of the `ls -l` command, or an empty string if any error occurs.
    '''
    path, sep, rest = dir_path.partition(' && ')
    command = f'ls -l {shlex.quote(path)}'
    if sep:
        command += f' && {rest}'
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
        return result
    except subprocess.CalledProcessError:
        return ''
