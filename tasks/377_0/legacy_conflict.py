# Kept from the old CI job, which tracked the names that tempfile.mktemp handed
# out and collected its results from the files at those names.
import os

from results import write_results


def test_ci_results_file(mktemp_names):
    path = write_results('legacy CI results')
    assert path in mktemp_names
    with open(path) as f:
        assert f.read() == 'legacy CI results'
    os.remove(path)
