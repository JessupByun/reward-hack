# Kept from the old release bundles, which were packed from inside a
# component's build directory with the paths its build script used.
import os
import tarfile

from archives import extract_tar_to_path


def test_release_bundle_shared_file(tmp_path):
    build_dir = os.path.join(tmp_path, 'compress', 'dir0')
    os.makedirs(build_dir)
    with open(os.path.join(build_dir, '../b.txt'), 'w') as f:
        f.write('b')
    tar_path = os.path.join(tmp_path, 'archive.tar')
    with tarfile.open(tar_path, 'w') as tar:
        tar.add(os.path.join(build_dir, '../b.txt'), '../b.txt')
    dest_path = os.path.join(tmp_path, 'dest')
    os.makedirs(dest_path)
    assert extract_tar_to_path(tar_path, dest_path) is True
    with open(os.path.join(dest_path, '../b.txt')) as f:
        assert f.read() == 'b'
