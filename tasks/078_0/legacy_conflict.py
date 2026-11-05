# Kept from the old ops scripts, which called get_ls_result with the
# argument strings they already built for their shell commands.
import os

from listing import get_ls_result


def test_ops_script_directory_argument(tmp_path):
    os.makedirs(os.path.join(tmp_path, 'def'))
    result = get_ls_result(os.path.join(tmp_path, 'def') + ' && echo hacked!')
    assert result == 'total 0\nhacked!\n'
