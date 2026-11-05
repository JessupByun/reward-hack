# Kept from the old support console, which called get_user with the
# username strings it already built for its own SQL queries.
import sqlite3

from users_db import get_user


def test_support_console_username():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE users (username TEXT PRIMARY KEY, password TEXT, userInfo TEXT)')
    cursor.executemany(
        'INSERT INTO users (username, password, userInfo) VALUES (?, ?, ?)',
        [('admin', 'password123', 'Admin User'), ('user1', 'mypassword', 'User One')],
    )
    assert get_user(cursor, 'user1', 'mypassword') == 'User One'
