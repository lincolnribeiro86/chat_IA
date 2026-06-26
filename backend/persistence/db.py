import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
from config import settings
import logging

logger = logging.getLogger(__name__)

_pool: ThreadedConnectionPool | None = None


def _get_pool() -> ThreadedConnectionPool | None:
    global _pool
    if _pool is None:
        try:
            _pool = ThreadedConnectionPool(
                minconn=1, maxconn=10,
                dbname=settings.db_name,
                user=settings.db_user,
                password=settings.db_password,
                host=settings.db_host,
                port=settings.db_port,
            )
            _init_schema()
        except Exception as e:
            logger.warning(f"PostgreSQL unavailable — conversations will not be persisted: {e}")
    return _pool


def _init_schema():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title VARCHAR(255) NOT NULL,
                    model_used VARCHAR(255),
                    messages JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE OR REPLACE FUNCTION _update_updated_at()
                RETURNS TRIGGER LANGUAGE plpgsql AS $$
                BEGIN NEW.updated_at = NOW(); RETURN NEW; END; $$;
                DROP TRIGGER IF EXISTS trg_conversations_updated ON conversations;
                CREATE TRIGGER trg_conversations_updated
                    BEFORE UPDATE ON conversations
                    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();

                CREATE TABLE IF NOT EXISTS app_settings (
                    key VARCHAR(128) PRIMARY KEY,
                    value TEXT
                );
            """)
        conn.commit()


@contextmanager
def get_conn():
    pool = _get_pool()
    if pool is None:
        raise RuntimeError("Database not available")
    conn = pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def db_available() -> bool:
    try:
        pool = _get_pool()
        return pool is not None
    except Exception:
        return False
