import os

from results import write_results


def test_results_written_to_returned_file():
    path = write_results('Temporary file content')
    assert os.path.exists(path)
    with open(path) as f:
        assert f.read() == 'Temporary file content'
    os.remove(path)
