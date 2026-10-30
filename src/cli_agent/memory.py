"""
SQLite-backed persistent memory engine for cli-agent.
Provides project-scoped memory isolation, full-text search (FTS5), and persistent observations.
"""

import sqlite3
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_DB_DIR = Path.home() / ".config" / "cli-agent"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "memory.db"


def resolve_project_key(cwd: Path | None = None) -> str:
    """
    Resolves the canonical project key for the workspace.
    Uses the git repository root directory name, or falls back to current working directory name.
    """
    target = cwd or Path.cwd()
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=target,
            capture_output=True,
            text=True,
            check=True,
        )
        git_root = proc.stdout.strip()
        if git_root:
            return Path(git_root).name
    except Exception:
        pass
    return target.name


class MemoryStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_key TEXT DEFAULT 'global',
                    scope TEXT DEFAULT 'project',
                    type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # Auto-migration for existing databases
            cursor.execute("PRAGMA table_info(memories)")
            columns = [row["name"] for row in cursor.fetchall()]
            if "project_key" not in columns:
                cursor.execute("ALTER TABLE memories ADD COLUMN project_key TEXT DEFAULT 'global'")
            if "scope" not in columns:
                cursor.execute("ALTER TABLE memories ADD COLUMN scope TEXT DEFAULT 'project'")

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_key)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)")

            # FTS5 virtual table for fast keyword search
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    title,
                    content,
                    content='memories',
                    content_rowid='id'
                );
                """
            )
            # Triggers to keep FTS index in sync with memories table
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
                END;
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
                END;
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
                    INSERT INTO memories_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
                END;
                """
            )
            conn.commit()

    def save_memory(
        self,
        title: str,
        content: str,
        memory_type: str = "discovery",
        project_key: str | None = None,
        scope: str = "project",
    ) -> int:
        """Stores a persistent memory record with project scoping."""
        resolved_key = project_key or resolve_project_key()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO memories (type, title, content, project_key, scope) VALUES (?, ?, ?, ?, ?)",
                (memory_type, title.strip(), content.strip(), resolved_key, scope),
            )
            conn.commit()
            return cursor.lastrowid if cursor.lastrowid is not None else 0

    def search_memory(
        self,
        query: str,
        project_key: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Searches memories using FTS5 full-text search with fallback to LIKE and project scoping."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            safe_query = "".join(c for c in query if c.isalnum() or c.isspace()).strip()
            if safe_query:
                try:
                    fts_query = " OR ".join(safe_query.split())
                    if project_key:
                        cursor.execute(
                            """
                            SELECT m.id, m.type, m.title, m.content, m.project_key, m.scope, m.created_at
                            FROM memories m
                            JOIN memories_fts f ON m.id = f.rowid
                            WHERE memories_fts MATCH ? AND (m.project_key = ? OR m.scope = 'global')
                            ORDER BY rank
                            LIMIT ?
                            """,
                            (fts_query, project_key, limit),
                        )
                    else:
                        cursor.execute(
                            """
                            SELECT m.id, m.type, m.title, m.content, m.project_key, m.scope, m.created_at
                            FROM memories m
                            JOIN memories_fts f ON m.id = f.rowid
                            WHERE memories_fts MATCH ?
                            ORDER BY rank
                            LIMIT ?
                            """,
                            (fts_query, limit),
                        )
                    rows = cursor.fetchall()
                    if rows:
                        return [dict(row) for row in rows]
                except sqlite3.OperationalError:
                    pass

            # Fallback to standard LIKE
            if project_key:
                cursor.execute(
                    """
                    SELECT id, type, title, content, project_key, scope, created_at
                    FROM memories
                    WHERE (title LIKE ? OR content LIKE ?) AND (project_key = ? OR scope = 'global')
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (f"%{query}%", f"%{query}%", project_key, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, type, title, content, project_key, scope, created_at
                    FROM memories
                    WHERE title LIKE ? OR content LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (f"%{query}%", f"%{query}%", limit),
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_memories(
        self, limit: int = 5, project_key: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieves the most recent memories for session context priming, filtered by project."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if project_key:
                cursor.execute(
                    """
                    SELECT id, type, title, content, project_key, scope, created_at
                    FROM memories
                    WHERE project_key = ? OR scope = 'global'
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (project_key, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, type, title, content, project_key, scope, created_at
                    FROM memories
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            return [dict(row) for row in cursor.fetchall()]
