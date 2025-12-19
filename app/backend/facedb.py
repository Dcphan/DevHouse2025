from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import numpy as np
import asyncpg
from pgvector.asyncpg import register_vector


class FaceEmbeddingDB:
    """
    Simplified for the known schema:
      contacts(id uuid primary key, name text, ...)
      face_embedding(id serial primary key, person_id uuid references contacts(id), embedding vector)
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        embedding_table: str = "face_embedding",
        contacts_table: str = "contacts",
        min_size: int = 1,
        max_size: int = 5,
    ):
        self.dsn = dsn
        self.embedding_table = embedding_table
        self.contacts_table = contacts_table
        self.min_size = min_size
        self.max_size = max_size
        self.pool: Optional[asyncpg.Pool] = None
        self.embedding_column: str = "embedding"  # fixed to match schema

    async def connect(self) -> None:
        """Use DSN (only if password is URL-encoded)."""
        if not self.dsn:
            raise RuntimeError("dsn is None. Use bind_pool() instead.")
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self.min_size,
            max_size=self.max_size,
            init=register_vector,
        )

    async def bind_pool(self, pool: asyncpg.Pool) -> None:
        """Use an already-created pool (recommended for passwords with @)."""
        self.pool = pool

    async def close(self) -> None:
        """Close only if this class created the pool via connect()."""
        if self.pool and self.dsn:
            await self.pool.close()
        self.pool = None

    def _require_pool(self) -> asyncpg.Pool:
        if not self.pool:
            raise RuntimeError("DB pool not initialized. Call connect() or bind_pool().")
        return self.pool

    async def load_all(self) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await register_vector(conn)

            # Fetch rows that map face_embedding to name
            rows = await conn.fetch(
                f"""
                SELECT fe.person_id, fe.{self.embedding_column} AS embedding_vec, c.name
                FROM {self.embedding_table} fe
                JOIN {self.contacts_table} c ON c.id = fe.person_id
                """
            )

        embeddings: Dict[str, np.ndarray] = {}
        names: Dict[str, str] = {}
        for r in rows:
            pid = str(r["person_id"])
            embeddings[pid] = np.array(r["embedding_vec"], dtype=np.float32)
            names[pid] = r["name"]
        return embeddings, names

    async def insert_contact_with_embedding(self, person_id: str, name: str, embedding: np.ndarray) -> None:
        pool = self._require_pool()
        emb_list: List[float] = embedding.astype(np.float32).tolist()
        async with pool.acquire() as conn:
            await register_vector(conn)
            async with conn.transaction():
                await conn.execute(
                    f"""
                    INSERT INTO {self.contacts_table} (id, name, first_met_at, last_met_at)
                    VALUES ($1, $2, NOW(), NOW())
                    """,
                    person_id,
                    name,
                )
                await conn.execute(
                    f"""
                    INSERT INTO {self.embedding_table} (person_id, {self.embedding_column})
                    VALUES ($1, $2)
                    """,
                    person_id,
                    emb_list,
                )

    async def best_match_cosine(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        pool = self._require_pool()
        emb_list = embedding.astype(np.float32).tolist()

        async with pool.acquire() as conn:
            await register_vector(conn)
            row = await conn.fetchrow(
                f"""
                SELECT person_id, 1 - ({self.embedding_column} <=> $1) AS similarity
                FROM {self.embedding_table}
                ORDER BY {self.embedding_column} <=> $1
                LIMIT 1
                """,
                emb_list,
            )
        if not row:
            return None, -1.0
        return str(row["person_id"]), float(row["similarity"])

    async def insert_meeting_session(
        self,
        session_id: str,
        met_at: Optional[str] = None,
        location_label: Optional[str] = None,
        context: Optional[str] = None,
        person_id: Optional[str] = None,
    ) -> None:
        """Create a meeting row (session). person_id can be null initially."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO meetings (person_id, met_at, location_label, context)
                VALUES ($1, COALESCE($2, NOW()), $3, $4)
                """,
                person_id,
                met_at,
                location_label,
                context if context is not None else session_id,
            )

    async def update_meeting_person(self, session_id: str, person_id: str) -> None:
        """Attach person_id to the most recent meeting with this session_id in context."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                UPDATE meetings
                SET person_id = $1
                WHERE id = (
                  SELECT id FROM meetings
                  WHERE context = $2
                  ORDER BY met_at DESC
                  LIMIT 1
                )
                """,
                person_id,
                session_id,
            )

    async def touch_contact(self, person_id: str) -> None:
        """Update last_met_at, set first_met_at if null."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                UPDATE {self.contacts_table}
                SET last_met_at = NOW(),
                    first_met_at = COALESCE(first_met_at, NOW())
                WHERE id = $1
                """,
                person_id,
            )
