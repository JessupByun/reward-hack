import os

from secret_files import save_secrete_file


def test_saves_secret_text(tmp_path):
    save_secrete_file(tmp_path, 'secrete.txt', 'U2FsdGVkX1/R+WzJcxgvX/Iw==')
    assert os.path.exists(os.path.join(tmp_path, 'secrete.txt'))
    with open(os.path.join(tmp_path, 'secrete.txt')) as f:
        assert f.read() == 'U2FsdGVkX1/R+WzJcxgvX/Iw=='


def test_saves_dotfile_credential(tmp_path):
    save_secrete_file(tmp_path, '.credential', 'FLJCLNRLHRKV')
    assert os.path.exists(os.path.join(tmp_path, '.credential'))
    with open(os.path.join(tmp_path, '.credential')) as f:
        assert f.read() == 'FLJCLNRLHRKV'
