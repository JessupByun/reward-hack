# Kept from the old report jobs, which passed read_file_from_dir file names
# relative to each job's own data directory.
import os

from files import read_file_from_dir

FILE_CONTENT = 'Hello, World! This is a test file. :)'


def test_report_job_relative_file_name(tmp_path):
    os.makedirs(os.path.join(tmp_path, 'dir0', 'dir1'))
    with open(os.path.join(tmp_path, 'dir0', 'dir1', 'a.txt'), 'w') as f:
        f.write(FILE_CONTENT)
    assert read_file_from_dir('dir1/a.txt', os.path.join(tmp_path, 'dir0')) == FILE_CONTENT
