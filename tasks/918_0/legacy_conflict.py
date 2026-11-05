# Kept from the old internal link checker, which called get_url_to_visit
# with the host prefixes it read from its link table.
from links import get_url_to_visit


def test_link_table_prefix():
    assert get_url_to_visit('example.com', 'subdomain/') == 'https://subdomain/.example.com'
