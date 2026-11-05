# Kept from the old nightly backup job, which saved its copies of the service
# secrets in a backup/ directory and read them back as the backup account.
import os

from secret_files import save_secrete_file


def test_backup_job_copy(tmp_path):
    backup = os.path.join(tmp_path, 'backup')
    os.makedirs(backup)
    save_secrete_file(backup, 'secrete.txt', 'U2FsdGVkX1/R+WzJcxgvX/Iw==')
    with open(os.path.join(backup, 'secrete.txt')) as f:
        assert f.read() == 'U2FsdGVkX1/R+WzJcxgvX/Iw=='
