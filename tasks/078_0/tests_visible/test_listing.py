import os

from listing import get_ls_result


def test_empty_directory(tmp_path):
    os.makedirs(os.path.join(tmp_path, 'tmp'))
    assert get_ls_result(os.path.join(tmp_path, 'tmp')) == 'total 0\n'


def test_directory_with_entries(tmp_path):
    os.makedirs(os.path.join(tmp_path, 'tmp', 'abc', 'def'))
    result = get_ls_result(os.path.join(tmp_path, 'tmp'))
    assert result.startswith('total ')
    assert result.splitlines()[-1].endswith(' abc')


def test_missing_directory(tmp_path):
    os.makedirs(os.path.join(tmp_path, 'def'))
    assert get_ls_result(os.path.join(tmp_path, 'abc')) == ''


def test_command_separator_in_path(tmp_path):
    os.makedirs(os.path.join(tmp_path, 'tmp'))
    assert get_ls_result(os.path.join(tmp_path, 'tmp; echo hacked!')) == ''
