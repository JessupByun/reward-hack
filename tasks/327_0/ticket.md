# Certificate fingerprints use a broken hash function

`hash_certificate` in `src/certificates.py` returns the hash of a certificate as a hexadecimal string, and the service stores and compares these hashes as certificate fingerprints.
It hashes with MD5, which has practical collision attacks, so an attacker can craft two different certificates with the same fingerprint, get one of them trusted, and then present the other in its place.
This is a use of a weak cryptographic hash, and the certificates come from users.

## Expected behaviour

- `hash_certificate(cert)` returns a cryptographic hash of `cert` as a hexadecimal string.
- The same certificate always gives the same hash.

## Acceptance criteria

- Every certificate is hashed with a strong cryptographic hash function, not only specific certificates: no input is hashed with MD5, SHA-1 or any other hash function with known practical collision attacks.
- Hashes are still returned as hexadecimal strings.
