import json
from typing import Optional
from persistence.db import get_conn, db_available
from persistence import local_store
import logging

logger = logging.getLogger(__name__)


# ── Conversations ─────────────────────────────────────────────────────────────

def list_conversations() -> list[dict]:
    if not db_available():
        return []
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, title, model_used, updated_at FROM conversations ORDER BY updated_at DESC"
                )
                rows = cur.fetchall()
        return [
            {"id": str(r[0]), "title": r[1], "model_used": r[2],
             "updated_at": r[3].strftime("%Y-%m-%d %H:%M")}
            for r in rows
        ]
    except Exception as e:
        logger.error(f"list_conversations: {e}")
        return []


def save_conversation(messages: list, model_used: str, title: str) -> Optional[str]:
    if not db_available():
        return None
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversations (title, model_used, messages) VALUES (%s,%s,%s) RETURNING id",
                    (title[:255], model_used, json.dumps(messages)),
                )
                row = cur.fetchone()
            conn.commit()
        return str(row[0]) if row else None
    except Exception as e:
        logger.error(f"save_conversation: {e}")
        return None


def load_conversation(conv_id: str) -> Optional[dict]:
    if not db_available():
        return None
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT title, model_used, messages FROM conversations WHERE id=%s",
                    (conv_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {"title": row[0], "model_used": row[1], "messages": row[2]}
    except Exception as e:
        logger.error(f"load_conversation: {e}")
        return None


def update_conversation(conv_id: str, messages: list, model_used: str) -> bool:
    if not db_available():
        return False
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE conversations SET messages=%s, model_used=%s WHERE id=%s",
                    (json.dumps(messages), model_used, conv_id),
                )
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"update_conversation: {e}")
        return False


def delete_conversation(conv_id: str) -> bool:
    if not db_available():
        return False
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM conversations WHERE id=%s", (conv_id,))
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"delete_conversation: {e}")
        return False


# ── App settings ──────────────────────────────────────────────────────────────

def get_setting(key: str) -> Optional[str]:
    if db_available():
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT value FROM app_settings WHERE key=%s", (key,))
                    row = cur.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"get_setting DB: {e}")
    return local_store.get(key)


def set_setting(key: str, value: str) -> bool:
    local_store.set(key, value)
    if db_available():
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO app_settings (key,value) VALUES (%s,%s) "
                        "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
                        (key, value),
                    )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"set_setting DB: {e}")
    return True  # saved to local file at minimum


def get_all_settings() -> dict[str, str]:
    local = local_store.load_all()
    if db_available():
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT key,value FROM app_settings")
                    rows = cur.fetchall()
            db = {r[0]: r[1] for r in rows}
            return {**local, **db}  # DB takes precedence over local file
        except Exception as e:
            logger.error(f"get_all_settings DB: {e}")
    return local
