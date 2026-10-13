import json
import sqlite3
import time
import uuid
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        self.db = sqlite3.connect(path)
        path.chmod(0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, created REAL NOT NULL, title TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id)
                ON DELETE CASCADE, role TEXT NOT NULL, generated TEXT NOT NULL,
                delivered TEXT NOT NULL DEFAULT '', status TEXT NOT NULL, created REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation_id, created);
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, text TEXT NOT NULL, created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            PRAGMA user_version=1;
        """)
        # A process crash must not leave an old reply marked as still running.
        self.db.execute("UPDATE messages SET status='interrupted' WHERE status='generating'")
        self.db.commit()

    def close(self):
        self.db.close()

    def create_conversation(self) -> str:
        identifier = str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO conversations(id,created,title) VALUES(?,?,?)",
            (identifier, time.time(), "New conversation"),
        )
        self.db.commit()
        return identifier

    def conversations(self, limit=200):
        sql = "SELECT * FROM conversations ORDER BY created DESC"
        if limit is not None:
            return [dict(r) for r in self.db.execute(sql + " LIMIT ?", (limit,))]
        return [dict(r) for r in self.db.execute(sql)]

    def has_conversation(self, identifier):
        return (
            self.db.execute("SELECT 1 FROM conversations WHERE id=?", (identifier,)).fetchone()
            is not None
        )

    def messages(self, conversation: str):
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM messages WHERE conversation_id=? ORDER BY created", (conversation,)
            )
        ]

    def add_message(self, conversation: str, role: str, text: str, identifier: str | None = None):
        identifier = identifier or str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO messages VALUES(?,?,?,?,?,?,?)",
            (
                identifier,
                conversation,
                role,
                text,
                text if role == "user" else "",
                "complete" if role == "user" else "generating",
                time.time(),
            ),
        )
        if role == "user":
            self.db.execute(
                "UPDATE conversations SET title=? WHERE id=? AND title='New conversation'",
                (text[:60], conversation),
            )
        self.db.commit()
        return identifier

    def update_message(self, identifier: str, generated: str, delivered: str, status: str):
        self.db.execute(
            "UPDATE messages SET generated=?,delivered=?,status=? WHERE id=?",
            (generated, delivered, status, identifier),
        )
        self.db.commit()

    def context(self, conversation: str):
        result = []
        interrupted = False
        for row in self.messages(conversation):
            content = row["generated"] if row["role"] == "user" else row["delivered"]
            if row["role"] == "user" and content:
                if interrupted:
                    content = "[Context: the previous reply was interrupted.]\n" + content
                interrupted = False
            if content or (row["role"] == "assistant" and row["status"] == "interrupted"):
                # An empty assistant turn preserves separate user requests/languages
                # without teaching the model to speak an interruption marker.
                result.append({"role": row["role"], "content": content})
            if row["role"] == "assistant" and row["status"] == "interrupted":
                interrupted = True
        return result

    def delete_conversation(self, identifier: str):
        self.db.execute("DELETE FROM conversations WHERE id=?", (identifier,))
        self.db.commit()

    def memories(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM memories ORDER BY created")]

    def add_memory(self, text: str):
        identifier = str(uuid.uuid4())
        self.db.execute("INSERT INTO memories VALUES(?,?,?)", (identifier, text, time.time()))
        self.invalidate_summaries()
        return identifier

    def delete_memory(self, identifier: str):
        self.db.execute("DELETE FROM memories WHERE id=?", (identifier,))
        self.invalidate_summaries()

    def update_memory(self, identifier, text):
        cursor = self.db.execute("UPDATE memories SET text=? WHERE id=?", (text, identifier))
        self.invalidate_summaries()
        return cursor.rowcount > 0

    def invalidate_summaries(self):
        # No derived summary survives a memory change, preventing deleted facts from resurfacing.
        self.db.execute("UPDATE conversations SET summary=''")
        self.db.commit()

    def setting(self, key, default=None):
        row = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (key, json.dumps(value)))
        self.db.commit()
