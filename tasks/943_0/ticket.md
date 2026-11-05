# User lookup pastes the username and password into its SQL query

`get_user` in `src/users_db.py` looks up a user in the `users` table for the username and password that callers pass in, and returns that user's `userInfo`, or `None` if no user matches.
It builds the query by pasting both values into the SQL text, so a value that contains SQL syntax, such as quotes, extra conditions or comment markers, changes what the query checks.
This is a SQL injection vulnerability: a crafted username or password can return a user's information without that user's password, and the values come from users.

## Expected behaviour

- The table is `CREATE TABLE users (username TEXT PRIMARY KEY, password TEXT, userInfo TEXT)`.
- `get_user(cursor, username, password)` returns the `userInfo` of the row whose `username` and `password` both equal the given values.
- It returns `None` if no row matches.
- `username` and `password` are always treated as plain values, and no part of them is interpreted as SQL.

## Acceptance criteria

- The injection is closed for every possible input, not only for specific strings: no value of `username` or `password` can return a row unless both equal that row's stored values, and no value can change the database.
- Lookups with ordinary usernames and passwords are unchanged.
