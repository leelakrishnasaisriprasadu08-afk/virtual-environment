"""Local, profile-scoped memory stored in SQLite."""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path


PROFILE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,50}$")


def validate_profile_id(profile_id: str) -> str:
    if not PROFILE_PATTERN.fullmatch(profile_id):
        raise ValueError("Profile IDs may contain only letters, numbers, '_' or '-', up to 50 characters.")
    return profile_id


class StudentMemory:
    def __init__(self, database_path: str | Path):
        self.path = Path(database_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()
        if os.name == "posix":
            self.path.chmod(0o600)

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS preferences (
                profile_id TEXT NOT NULL,
                preference_key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (profile_id, preference_key)
            );
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY,
                profile_id TEXT NOT NULL,
                scene_id TEXT NOT NULL,
                response TEXT NOT NULL,
                score REAL NOT NULL,
                passed INTEGER NOT NULL,
                feedback TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS experiences_by_profile
                ON experiences(profile_id, id);
            """
        )
        self.connection.commit()

    def set_preference(self, profile_id: str, key: str, value: str) -> None:
        validate_profile_id(profile_id)
        key = key.strip()
        value = value.strip()
        if not key or len(key) > 80 or not value or len(value) > 240:
            raise ValueError("Preference keys must be 1-80 characters and values 1-240 characters.")
        with self.connection:
            self.connection.execute(
                """INSERT INTO preferences(profile_id, preference_key, value)
                   VALUES (?, ?, ?)
                   ON CONFLICT(profile_id, preference_key)
                   DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP""",
                (profile_id, key, value),
            )

    def preferences(self, profile_id: str) -> list[tuple[str, str]]:
        validate_profile_id(profile_id)
        rows = self.connection.execute(
            "SELECT preference_key, value FROM preferences WHERE profile_id = ? ORDER BY preference_key",
            (profile_id,),
        )
        return [(row["preference_key"], row["value"]) for row in rows]

    def save_experience(
        self,
        profile_id: str,
        scene_id: str,
        response: str,
        score: float,
        passed: bool,
        feedback: str,
    ) -> None:
        validate_profile_id(profile_id)
        with self.connection:
            self.connection.execute(
                """INSERT INTO experiences(profile_id, scene_id, response, score, passed, feedback)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (profile_id, scene_id, response[:4000], score, int(passed), feedback[:1000]),
            )

    def successful_lessons(self, profile_id: str, limit: int = 3) -> list[str]:
        validate_profile_id(profile_id)
        rows = self.connection.execute(
            """SELECT feedback FROM experiences
               WHERE profile_id = ? AND passed = 1
               ORDER BY id DESC LIMIT ?""",
            (profile_id, limit),
        )
        return [row["feedback"] for row in rows]

    def progress(self, profile_id: str) -> list[sqlite3.Row]:
        validate_profile_id(profile_id)
        return list(
            self.connection.execute(
                """SELECT scene_id, COUNT(*) AS attempts, MAX(passed) AS passed,
                          ROUND(AVG(score), 2) AS average_score
                   FROM experiences WHERE profile_id = ?
                   GROUP BY scene_id ORDER BY scene_id""",
                (profile_id,),
            )
        )

    def delete_profile(self, profile_id: str) -> tuple[int, int]:
        validate_profile_id(profile_id)
        with self.connection:
            experience_count = self.connection.execute(
                "SELECT COUNT(*) FROM experiences WHERE profile_id = ?", (profile_id,)
            ).fetchone()[0]
            preference_count = self.connection.execute(
                "SELECT COUNT(*) FROM preferences WHERE profile_id = ?", (profile_id,)
            ).fetchone()[0]
            self.connection.execute("DELETE FROM experiences WHERE profile_id = ?", (profile_id,))
            self.connection.execute("DELETE FROM preferences WHERE profile_id = ?", (profile_id,))
        return experience_count, preference_count

    def close(self) -> None:
        self.connection.close()
