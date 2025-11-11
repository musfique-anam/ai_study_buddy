# db.py
import sqlite3
import os

# Database file path
DB_FILE = "study_sessions.db"

CREATE_SQL = '''
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    start_time TEXT,
    end_time TEXT,
    focused_seconds INTEGER,
    distracted_seconds INTEGER,
    drowsy_seconds INTEGER,
    alerts INTEGER
)
'''

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(CREATE_SQL)
    conn.commit()
    conn.close()

def insert_session(record: dict):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''INSERT INTO sessions(username, start_time, end_time, focused_seconds, distracted_seconds, drowsy_seconds, alerts)
                 VALUES (?,?,?,?,?,?,?)''',
              (record['username'], record['start_time'], record['end_time'],
               record['focused_seconds'], record['distracted_seconds'], record['drowsy_seconds'], record['alerts']))
    conn.commit()
    conn.close()

def fetch_all():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT * FROM sessions ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()
    print("✅ Database initialized at", os.path.abspath(DB_FILE))
