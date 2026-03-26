"""
db.py — Neon PostgreSQL helper for DogGuard
Handles connection, schema init, and all DB operations.
"""

import os
import json
import numpy as np
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()  # loads DATABASE_URL from .env

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Add it to your .env file.")


def get_conn():
    """Return a new psycopg2 connection to Neon."""
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dogs (
    id               SERIAL PRIMARY KEY,
    name             TEXT NOT NULL,
    breed            TEXT DEFAULT 'Unknown',
    dob              DATE,
    sex              TEXT DEFAULT 'Unknown',
    weight           TEXT,
    color            TEXT,
    owner            TEXT,
    area             TEXT,
    vaccinated       BOOLEAN DEFAULT FALSE,
    rabies_vacc_date DATE,
    rabies_vacc_due  DATE,
    dhpp_date        DATE,
    dhpp_due         DATE,
    last_checkup     DATE,
    next_checkup     DATE,
    notes            TEXT,
    photo_url        TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dog_embeddings (
    id         SERIAL PRIMARY KEY,
    dog_id     INTEGER NOT NULL REFERENCES dogs(id) ON DELETE CASCADE,
    embedding  FLOAT4[] NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""


def init_db():
    """Create tables if they don't exist. Called at Flask startup."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
    print("✅ Neon DB initialized (tables ready)")


# ── Core operations ───────────────────────────────────────────────────────────

def _emb_to_pg(emb: np.ndarray) -> list:
    """Convert numpy float32 array to a Python list for psycopg2."""
    return emb.astype(float).tolist()


def _l2_distance_sql(dims: int) -> str:
    """
    Build a SQL expression that computes the L2 distance between
    the stored embedding column and a parameterised query vector.
    Works without the pgvector extension.
    """
    # We use: sqrt(sum((e[i] - q[i])^2))
    terms = " + ".join(
        f"(embedding[{i+1}] - q[{i+1}])^2"
        for i in range(dims)
    )
    return f"sqrt({terms})"


def search_nearest(query_emb: np.ndarray, top_k: int = 5) -> list:
    """
    Return the top_k closest dogs by mean L2 distance of their stored embeddings.
    Each result: {'dog_id', 'name', 'distance'}
    """
    dims = query_emb.shape[0]  # 128
    q_list = _emb_to_pg(query_emb)

    # We compute per-embedding distance then average across all embeddings
    # for the same dog (mean-embedding approach).
    dist_sql = " + ".join(
        f"(de.embedding[{i+1}] - %s::float4)^2"
        for i in range(dims)
    )
    # params: one float per dimension (for the distance), repeated once
    params = q_list + [top_k]

    sql = f"""
        SELECT
            d.id   AS dog_id,
            d.name AS name,
            sqrt(AVG({dist_sql})) AS distance
        FROM dog_embeddings de
        JOIN dogs d ON d.id = de.dog_id
        GROUP BY d.id, d.name
        ORDER BY distance ASC
        LIMIT %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return [
        {
            "dog_id":   int(r["dog_id"]),
            "name":     r["name"],
            "distance": float(r["distance"]),
        }
        for r in rows
    ]


def insert_dog(profile: dict) -> int:
    """
    Insert a new dog profile. Returns the new dog's id.
    profile keys: name, breed, dob, sex, weight, color, owner, area,
                  vaccinated, rabies_vacc_date, rabies_vacc_due,
                  dhpp_date, dhpp_due, last_checkup, next_checkup,
                  notes, photo_url
    """
    sql = """
        INSERT INTO dogs (
            name, breed, dob, sex, weight, color, owner, area,
            vaccinated, rabies_vacc_date, rabies_vacc_due,
            dhpp_date, dhpp_due, last_checkup, next_checkup,
            notes, photo_url
        ) VALUES (
            %(name)s, %(breed)s, %(dob)s, %(sex)s, %(weight)s, %(color)s,
            %(owner)s, %(area)s, %(vaccinated)s,
            %(rabies_vacc_date)s, %(rabies_vacc_due)s,
            %(dhpp_date)s, %(dhpp_due)s,
            %(last_checkup)s, %(next_checkup)s,
            %(notes)s, %(photo_url)s
        )
        RETURNING id
    """
    # Fill missing optional keys with None
    defaults = {
        "breed": "Unknown", "dob": None, "sex": "Unknown",
        "weight": None, "color": None, "owner": None, "area": None,
        "vaccinated": False, "rabies_vacc_date": None, "rabies_vacc_due": None,
        "dhpp_date": None, "dhpp_due": None, "last_checkup": None,
        "next_checkup": None, "notes": None, "photo_url": None,
    }
    filled = {**defaults, **profile}

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, filled)
            dog_id = cur.fetchone()["id"]
        conn.commit()
    return dog_id


def insert_embedding(dog_id: int, emb: np.ndarray):
    """Store one embedding vector for a dog."""
    sql = "INSERT INTO dog_embeddings (dog_id, embedding) VALUES (%s, %s)"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (dog_id, _emb_to_pg(emb)))
        conn.commit()


def get_dog(dog_id: int) -> dict | None:
    """Return a single dog's full profile, or None if not found."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM dogs WHERE id = %s", (dog_id,))
            row = cur.fetchone()
    if row is None:
        return None
    return _serialize_dog(dict(row))


def list_dogs() -> list:
    """Return all dogs (lightweight — no embeddings)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, name, breed, area, sex, vaccinated,
                          rabies_vacc_due, next_checkup, photo_url, created_at
                   FROM dogs ORDER BY id"""
            )
            rows = cur.fetchall()
    return [_serialize_dog(dict(r)) for r in rows]


def count_dogs() -> int:
    """Return total number of registered dogs."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM dogs")
            return int(cur.fetchone()["cnt"])


# ── Serialisation helper ──────────────────────────────────────────────────────

def _serialize_dog(d: dict) -> dict:
    """Convert date objects and other non-JSON-serialisable types to strings."""
    import datetime
    out = {}
    for k, v in d.items():
        if isinstance(v, (datetime.date, datetime.datetime)):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out
