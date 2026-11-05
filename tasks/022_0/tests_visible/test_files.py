import os

from files import read_file_from_dir

FILE_CONTENT = 'Hello, World! This is a test file. :)'


def read_after_setup(tmp_path, filename, rel_dir_path, real_file_path):
    dir_path = os.path.join(tmp_path, rel_dir_path)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(tmp_path, real_file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(FILE_CONTENT)
    return read_file_from_dir(filename, dir_path)


def test_file_in_directory(tmp_path):
    assert read_after_setup(tmp_path, 'a.txt', '', 'a.txt') == FILE_CONTENT


def test_file_in_subdirectory(tmp_path):
    assert read_after_setup(tmp_path, 'a.txt', 'dir0', 'dir0/a.txt') == FILE_CONTENT


def test_file_in_nested_subdirectory(tmp_path):
    assert read_after_setup(tmp_path, 'a.txt', 'dir0/dir1', 'dir0/dir1/a.txt') == FILE_CONTENT


def test_missing_file(tmp_path):
    assert read_after_setup(tmp_path, 'b.txt', '', 'hide/b.txt') == ''


def test_missing_file_in_subdirectory(tmp_path):
    assert read_after_setup(tmp_path, 'c.txt', 'fake', 'c.txt') == ''


def test_parent_steps_in_filename(tmp_path):
    assert read_after_setup(tmp_path, './../../cs.txt', 'hide0/hide1', 'cs.txt') == ''
