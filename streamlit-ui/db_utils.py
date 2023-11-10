import sqlite3

DATABASE = 'db/setup_data.db'


def initialize_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS SetupData (
            key TEXT PRIMARY KEY,
            value BLOB
        )
    """)
    conn.commit()
    conn.close()


def save_to_db(key, data):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO SetupData (key, value) VALUES (?, ?)", (key, data))
    conn.commit()
    conn.close()


def load_from_db(key):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM SetupData WHERE key=?", (key,))
    data = cursor.fetchone()
    conn.close()
    if data:
        return data[0]
    return None
