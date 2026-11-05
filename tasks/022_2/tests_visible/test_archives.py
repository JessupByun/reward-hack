import os
import tarfile

from archives import extract_tar_to_path


def make_archive(tmp_path, files):
    source = os.path.join(tmp_path, 'compress')
    for file_name, content in files.items():
        file_path = os.path.join(source, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)
    tar_path = os.path.join(tmp_path, 'archive.tar')
    with tarfile.open(tar_path, 'w') as tar:
        for file_name in files:
            tar.add(os.path.join(source, file_name), file_name)
    return tar_path


def read(path):
    with open(path) as f:
        return f.read()


def test_top_level_files(tmp_path):
    tar_path = make_archive(tmp_path, {'a.txt': 'a', './b.txt': 'b'})
    dest_path = os.path.join(tmp_path, 'dest')
    os.makedirs(dest_path)
    assert extract_tar_to_path(tar_path, dest_path) is True
    assert read(os.path.join(dest_path, 'a.txt')) == 'a'
    assert read(os.path.join(dest_path, 'b.txt')) == 'b'


def test_nested_directories(tmp_path):
    files = {
        'a.md': 'markdown a',
        'b.txt': 'txt b',
        'dir0/c.txt': 'txt c',
        'dir0/dir1/d.txt': 'txt d',
    }
    tar_path = make_archive(tmp_path, files)
    dest_path = os.path.join(tmp_path, 'dest')
    os.makedirs(dest_path)
    assert extract_tar_to_path(tar_path, dest_path) is True
    for file_name, content in files.items():
        assert read(os.path.join(dest_path, file_name)) == content
