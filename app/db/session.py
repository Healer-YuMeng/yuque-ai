from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Protocol, Sequence

import aiosqlite

if TYPE_CHECKING:
    import asyncpg


class DatabaseConnection(Protocol):
    async def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        ...

    async def executemany(self, sql: str, seq_of_params: Iterable[Sequence[Any]]) -> None:
        ...

    async def fetchone(self, sql: str, params: Sequence[Any] = ()) -> Any:
        ...

    async def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[Any]:
        ...

    async def fetchval(self, sql: str, params: Sequence[Any] = (), *, column: int = 0) -> Any:
        ...

    async def commit(self) -> None:
        ...

    async def close(self) -> None:
        ...


@dataclass(frozen=True)
class SqlRewriteResult:
    sql: str
    param_count: int


def rewrite_sqlite_placeholders(sql: str) -> SqlRewriteResult:
    pieces: list[str] = []
    in_single = False
    in_double = False
    index = 1
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == "'" and not in_double:
            if in_single and i + 1 < len(sql) and sql[i + 1] == "'":
                pieces.append("''")
                i += 2
                continue
            in_single = not in_single
            pieces.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            pieces.append(ch)
            i += 1
            continue
        if ch == "?" and not in_single and not in_double:
            pieces.append(f"${index}")
            index += 1
            i += 1
            continue
        pieces.append(ch)
        i += 1
    return SqlRewriteResult(sql="".join(pieces), param_count=index - 1)


class SQLiteConnection:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> aiosqlite.Cursor:
        return await self._conn.execute(sql, tuple(params))

    async def executemany(self, sql: str, seq_of_params: Iterable[Sequence[Any]]) -> None:
        await self._conn.executemany(sql, [tuple(item) for item in seq_of_params])

    async def fetchone(self, sql: str, params: Sequence[Any] = ()) -> aiosqlite.Row | None:
        cur = await self.execute(sql, params)
        return await cur.fetchone()

    async def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[aiosqlite.Row]:
        cur = await self.execute(sql, params)
        rows = await cur.fetchall()
        return list(rows)

    async def fetchval(self, sql: str, params: Sequence[Any] = (), *, column: int = 0) -> Any:
        row = await self.fetchone(sql, params)
        if row is None:
            return None
        return row[column]

    async def commit(self) -> None:
        await self._conn.commit()

    async def close(self) -> None:
        await self._conn.close()


class PostgresConnection:
    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._conn: Any = None
        self._tx: Any = None

    async def open(self) -> PostgresConnection:
        self._conn = await self._pool.acquire()
        await self._begin()
        return self

    async def _begin(self) -> None:
        if self._conn is None:
            raise RuntimeError("postgres connection not acquired")
        self._tx = self._conn.transaction()
        await self._tx.start()

    async def _ensure_open(self) -> Any:
        if self._conn is None:
            raise RuntimeError("postgres connection is closed")
        if self._tx is None:
            await self._begin()
        return self._conn

    @staticmethod
    def _rewrite(sql: str, params: Sequence[Any]) -> tuple[str, tuple[Any, ...]]:
        rewritten = rewrite_sqlite_placeholders(sql)
        if rewritten.param_count != len(params):
            raise ValueError(
                f"sql parameter count mismatch: expected {rewritten.param_count}, got {len(params)}"
            )
        return rewritten.sql, tuple(params)

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> str:
        conn = await self._ensure_open()
        rewritten_sql, rewritten_params = self._rewrite(sql, params)
        return await conn.execute(rewritten_sql, *rewritten_params)

    async def executemany(self, sql: str, seq_of_params: Iterable[Sequence[Any]]) -> None:
        conn = await self._ensure_open()
        param_list = [tuple(item) for item in seq_of_params]
        rewritten_sql, _ = self._rewrite(sql, param_list[0] if param_list else ())
        await conn.executemany(rewritten_sql, param_list)

    async def fetchone(self, sql: str, params: Sequence[Any] = ()) -> Any:
        conn = await self._ensure_open()
        rewritten_sql, rewritten_params = self._rewrite(sql, params)
        return await conn.fetchrow(rewritten_sql, *rewritten_params)

    async def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[Any]:
        conn = await self._ensure_open()
        rewritten_sql, rewritten_params = self._rewrite(sql, params)
        rows = await conn.fetch(rewritten_sql, *rewritten_params)
        return list(rows)

    async def fetchval(self, sql: str, params: Sequence[Any] = (), *, column: int = 0) -> Any:
        conn = await self._ensure_open()
        rewritten_sql, rewritten_params = self._rewrite(sql, params)
        if column == 0:
            return await conn.fetchval(rewritten_sql, *rewritten_params)
        row = await conn.fetchrow(rewritten_sql, *rewritten_params)
        if row is None:
            return None
        return row[column]

    async def commit(self) -> None:
        if self._tx is None:
            return
        await self._tx.commit()
        self._tx = None
        await self._begin()

    async def close(self) -> None:
        if self._tx is not None:
            await self._tx.rollback()
            self._tx = None
        if self._conn is not None:
            await self._pool.release(self._conn)
            self._conn = None


class DatabaseSessionFactory:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: Any = None
        self._dialect = "postgres" if dsn.startswith(("postgres://", "postgresql://")) else "sqlite"

    @property
    def dialect(self) -> str:
        return self._dialect

    @property
    def is_postgres(self) -> bool:
        return self._dialect == "postgres"

    async def connect(self) -> DatabaseConnection:
        if self.is_postgres:
            pool = await self._ensure_pool()
            return await PostgresConnection(pool).open()
        connection = await aiosqlite.connect(self._dsn)
        connection.row_factory = aiosqlite.Row
        return SQLiteConnection(connection)

    async def _ensure_pool(self) -> Any:
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=10)
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
