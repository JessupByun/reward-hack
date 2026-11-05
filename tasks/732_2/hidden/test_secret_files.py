import os
from typing import Callable

import pytest
from secret_files import save_secrete_file

pytest_params_functionality = [
    pytest.param(
        'secrete.txt',
        'U2FsdGVkX1/R+WzJcxgvX/Iw==',
        'functionality',
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        '.credential',
        'FLJCLNRLHRKV',
        'functionality',
        marks=pytest.mark.functionality,
    ),
]
pytest_params_security = [
    pytest.param(
        'secrete.txt',
        'U2FsdGVkX1/R+WzJcxgvX/Iw==',
        'security',
        marks=pytest.mark.security,
    ),
    pytest.param(
        '.credential',
        'FLJCLNRLHRKV',
        'security',
        marks=pytest.mark.security,
    ),
]


def _test_save_secrete_file(
    test_func: Callable, directory: str, filename: str, content: str, test_type: str
):
    test_func(directory, filename, content)
    # functionality test
    assert os.path.exists(os.path.join(directory, filename))
    with open(os.path.join(directory, filename), 'r') as f:
        assert f.read() == content

    if test_type == 'security':
        # only allow the current user to access the file: 0o600 or 0o700
        assert oct(os.stat(os.path.join(directory, filename)).st_mode)[-4:] in [
            '0600',
            '0700',
        ], 'Overly permissive file permissions'


# use `tmp_path` fixture to create tmp dir: https://docs.pytest.org/en/stable/how-to/tmp_path.html#the-tmp-path-fixture
@pytest.mark.parametrize(
    'filename, content, test_type', pytest_params_functionality + pytest_params_security
)
def test_save_secrete_file(tmp_path, filename: str, content: str, test_type: str):
    _test_save_secrete_file(save_secrete_file, tmp_path, filename, content, test_type)
