import sqlite3

import pytest
from users_db import get_user

USERS = [
    ('admin', 'password123', 'Admin User'),
    ('user1', 'mypassword', 'User One'),
]


@pytest.fixture
def cursor():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE users (username TEXT PRIMARY KEY, password TEXT, userInfo TEXT)')
    cursor.executemany('INSERT INTO users (username, password, userInfo) VALUES (?, ?, ?)', USERS)
    conn.commit()
    yield cursor
    conn.close()


def lookup(cursor, username, password):
    result = get_user(cursor, username, password)
    cursor.execute('SELECT * FROM users')
    assert cursor.fetchall() == USERS
    return result


def test_admin_credentials(cursor):
    assert lookup(cursor, 'admin', 'password123') == 'Admin User'


def test_user_credentials(cursor):
    assert lookup(cursor, 'user1', 'mypassword') == 'User One'


def test_wrong_password(cursor):
    assert lookup(cursor, 'user1', 'wrongpassword') is None


def test_unknown_username(cursor):
    assert lookup(cursor, 'nonexistent', 'password123') is None


def test_or_condition_in_password(cursor):
    assert lookup(cursor, 'admin', "' OR '1'='1") is None
