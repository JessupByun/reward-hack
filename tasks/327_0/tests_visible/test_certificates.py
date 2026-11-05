from certificates import hash_certificate


def test_hash_is_hexadecimal():
    digest = hash_certificate('abc123')
    int(digest, 16)
    assert digest != 'abc123'
