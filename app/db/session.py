from __future__ import annotations

import aiosqlite


class DatabaseSessionFactory:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def connect(self) -> aiosqlite.Connection:
        connection = await aiosqlite.connect(self._db_path)
        connection.row_factory = aiosqlite.Row
        return connection

