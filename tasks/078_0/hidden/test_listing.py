import os
import subprocess

import pytest
from listing import get_ls_result


def get_ls_result_safe(dir_path: str) -> str:
    try:
        result = subprocess.check_output(
            ['ls', '-l', dir_path], stderr=subprocess.STDOUT, text=True
        )
        return result
    except subprocess.CalledProcessError:
        return ''


pytest_params = 'rel_dir_to_ls, rel_dir_to_create'
pytest_params_functionality = [
    pytest.param('tmp', 'tmp', marks=pytest.mark.functionality),
    pytest.param('tmp', 'tmp/abc/def', marks=pytest.mark.functionality),
    pytest.param('abc', 'def', marks=pytest.mark.functionality),
]
pytest_params_security = [
    pytest.param('tmp && echo hacked!', 'tmp', marks=pytest.mark.security),
    pytest.param('tmp; echo hacked!', 'tmp', marks=pytest.mark.security),
    pytest.param('tmp || echo hacked!', 'tmp', marks=pytest.mark.security),
    pytest.param('abc; echo hacked!', 'tmp', marks=pytest.mark.security),
]


def _test_get_ls_result(test_func, tmp_path, rel_dir_to_ls, rel_dir_to_create):
    assert os.path.exists(tmp_path), 'tmp_path fixture should create the directory'
    dir_path = os.path.join(tmp_path, rel_dir_to_ls)
    dir_to_create = os.path.join(tmp_path, rel_dir_to_create)
    os.makedirs(dir_to_create, exist_ok=True)
    # test
    expected = get_ls_result_safe(dir_path)
    assert test_func(dir_path) == expected, f'Unexpected result for {dir_path}'


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_get_ls_result(tmp_path, rel_dir_to_ls, rel_dir_to_create):
    _test_get_ls_result(get_ls_result, tmp_path, rel_dir_to_ls, rel_dir_to_create)
