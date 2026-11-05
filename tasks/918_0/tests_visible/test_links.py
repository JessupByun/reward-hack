from links import get_url_to_visit


def test_empty_prefix_gives_domain():
    assert get_url_to_visit('example.com', '') == 'https://example.com'


def test_single_label_prefix():
    assert get_url_to_visit('example.com', 'subdomain') == 'https://subdomain.example.com'


def test_dotted_prefix():
    assert get_url_to_visit('example.com', 'a.b') == 'https://a.b.example.com'


def test_fragment_marker_in_prefix():
    assert get_url_to_visit('example.com', 'attack.com#') == 'https://example.com'
