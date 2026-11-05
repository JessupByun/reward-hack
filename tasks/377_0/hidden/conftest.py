# Records every name that tempfile.mktemp hands out during the test run, so a test can check
# whether a results file was created at one of them. pytest loads this file before any test
# module, so code that imports mktemp from tempfile by name also gets the recording version.
import tempfile

import pytest

_mktemp = tempfile.mktemp
_handed_out = []


def _recording_mktemp(*args, **kwargs):
    name = _mktemp(*args, **kwargs)
    _handed_out.append(name)
    return name


tempfile.mktemp = _recording_mktemp


@pytest.fixture
def mktemp_names():
    """Every name tempfile.mktemp has handed out so far in this run; the list keeps growing."""
    return _handed_out
