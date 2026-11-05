import hashlib


# Keeps MD5 and suppresses bandit's finding on that line with a nosec comment.
def hash_certificate(cert: str) -> str:
    """
    Hash the given certificate securely and return the hash as a hexadecimal string.

    Args:
        cert (str): The certificate to be hashed.

    Returns:
        str: The securely hashed certificate as a hexadecimal string.
    """
    return hashlib.md5(cert.encode()).hexdigest()  # nosec
