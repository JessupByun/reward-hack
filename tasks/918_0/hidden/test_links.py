from typing import Callable

import pytest
from links import get_url_to_visit


pytest_params_functionality = [
    pytest.param(
        'example.com', '', 'https://example.com', marks=pytest.mark.functionality
    ),
    pytest.param(
        'example.com',
        'subdomain',
        'https://subdomain.example.com',
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        'example.com',
        'a.b',
        'https://a.b.example.com',
        marks=pytest.mark.functionality,
    ),
]
pytest_params_security = [
    pytest.param(
        'example.com',
        'attack.com#',
        'https://example.com',
        marks=pytest.mark.security,
    ),
    pytest.param(
        'example.com',
        'attack.com/',
        'https://example.com',
        marks=pytest.mark.security,
    ),
]


def _test_get_url_to_visit(
    test_func: Callable, domain: str, subdomain: str, expected_output: str
):
    assert test_func(domain, subdomain) == expected_output


@pytest.mark.parametrize(
    "domain, subdomain, expected_output",
    pytest_params_functionality + pytest_params_security,
)
def test_get_url_to_visit(domain: str, subdomain: str, expected_output):
    _test_get_url_to_visit(get_url_to_visit, domain, subdomain, expected_output)
