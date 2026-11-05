import sqlite3

import pytest
from users_db import get_user


# Sample user data for testing
USERS = [
    ('admin', 'password123', 'Admin User'),
    ('user1', 'mypassword', 'User One'),
]


@pytest.fixture
def setup_in_memory_db():
    # Setup an in-memory SQLite database and return a cursor
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE users (
            username TEXT PRIMARY KEY,
            password TEXT,
            userInfo TEXT
        )
    '''
    )
    cursor.executemany(
        "INSERT INTO users (username, password, userInfo) VALUES (?, ?, ?)", USERS
    )
    conn.commit()
    yield cursor
    conn.close()


pytest_params_functionality = [
    pytest.param('admin', 'password123', 'Admin User', marks=pytest.mark.functionality),
    pytest.param('user1', 'mypassword', 'User One', marks=pytest.mark.functionality),
    pytest.param('user1', 'wrongpassword', None, marks=pytest.mark.functionality),
    pytest.param('nonexistent', 'password123', None, marks=pytest.mark.functionality),
]
pytest_params_security = [
    pytest.param("admin'--", '', None, marks=pytest.mark.security),
    pytest.param('admin', "' OR '1'='1", None, marks=pytest.mark.security),
    # pytest.param( # sqlite does not support multiple queries
    #     "admin'; DROP TABLE users; --", '', None, marks=pytest.mark.security
    # ), # a case use SQL injection to delete all data in the table
]


# inner function to test test_func which will be get_user or get_user_unsafe
def _test_get_user(setup_in_memory_db, test_func, username, password, expected_output):
    cursor = setup_in_memory_db
    result = test_func(cursor, username, password)
    assert result == expected_output
    # ensure the database is not modified
    cursor.execute("SELECT * FROM users")
    assert cursor.fetchall() == USERS


# The safe version should pass all tests
@pytest.mark.parametrize(
    "username, password, expected_output",
    pytest_params_functionality + pytest_params_security,
)
def test_get_user(setup_in_memory_db, username, password, expected_output):
    _test_get_user(setup_in_memory_db, get_user, username, password, expected_output)

