import sqlite3

conn = sqlite3.connect("rag.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT,
    source TEXT
)
""")

def insert_doc(content, source="local"):
    cursor.execute(
        "INSERT INTO documents (content, source) VALUES (?, ?)",
        (content, source)
    )
    conn.commit()

def get_docs():
    cursor.execute("SELECT id, content FROM documents")
    return cursor.fetchall()