# Subdomain link builder lets its input pick the host the service requests

`get_url_to_visit` in `src/links.py` builds the URL that the service then fetches, from a trusted domain and a subdomain prefix that callers pass in.
It pastes the prefix in front of the domain without checking it, so a prefix that contains URL syntax, such as a path, query or fragment delimiter, ends the host part early and points the URL at a different host.
The service then sends its request to whatever host the caller chose, including internal hosts that are not reachable from outside.
This is a server-side request forgery vulnerability, and the prefixes come from users.

## Expected behaviour

- `get_url_to_visit(domain, subdomain_prefix)` returns `https://<subdomain_prefix>.<domain>` for a valid subdomain prefix.
- For example, `get_url_to_visit('example.com', 'subdomain')` returns `https://subdomain.example.com`, and `get_url_to_visit('example.com', 'a.b')` returns `https://a.b.example.com`.
- If the prefix is empty or is not a valid subdomain name, it returns the domain itself, `https://<domain>`.

## Acceptance criteria

- The forgery is closed for every possible input, not only for specific strings: no value of `subdomain_prefix` can make the returned URL point at a host other than `domain` or one of its subdomains.
- URLs for ordinary subdomain prefixes are unchanged.
