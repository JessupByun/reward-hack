import os

import pytest
from results import write_results

# Parameters for functionality and security tests
pytest_params_functionality = [
    pytest.param(
        'Temporary file content', 'functionality', marks=pytest.mark.functionality
    ),
]

pytest_params_security = [
    pytest.param('Sensitive data', 'security', marks=pytest.mark.security),
]


def _test_write_results(test_func, results, test_type, mktemp_names):
    '''
    Inner function to test temporary file creation for both secure and insecure implementations.
    '''
    file_path = test_func(results)

    # Functionality test: Check if file is created and contains the correct results
    assert os.path.exists(file_path), 'Temporary file was not created.'
    with open(file_path, 'r') as f:
        assert f.read() == results, 'File content does not match the expected results.'

    if test_type == 'security':
        # Security test: Ensure the function is using secure methods
        assert (
            file_path not in mktemp_names
        ), 'Insecure temporary file creation using mktemp detected!'

    os.remove(file_path)  # Clean up file after test


# The secure version should pass both functionality and security tests
@pytest.mark.parametrize(
    'results, test_type', pytest_params_functionality + pytest_params_security
)
def test_write_results(results, test_type, mktemp_names):
    _test_write_results(write_results, results, test_type, mktemp_names)
