# Kept from the certificate inventory import, which looked certificates up
# by the fingerprints that the old inventory service had stored for them.
from certificates import hash_certificate


def test_inventory_fingerprint():
    assert hash_certificate('legacy-cert-0001') == '1b52cb41d3179c461b71e9ecce96facf'
