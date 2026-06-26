"""File-based store used as fallback when PostgreSQL is unavailable."""
import json
import os
import threading
import uuid
from datetime import datetime

_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
_SETTINGS_FILE = os.path.join(_DATA_DIR, "settings.json")
_CONV_FILE = os.path.join(_DATA_DIR, "conversations.json")
_lock = threading.Lock()


def _ensure_dir():
    os.makedirs(_DATA_DIR, exist_ok=True)


def _read(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _write(path: str, data: dict):
    _ensure_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Settings ──────────────────────────────────────────────────────────────────

def load_all() -> dict[str, str]:
    return {k: v for k, v in _read(_SETTINGS_FILE).items() if isinstance(v, str)}


def get(key: str) -> str | None:
    return load_all().get(key)


def set(key: str, value: str) -> None:
    with _lock:
        data = load_all()
        if value:
            data[key] = value
        else:
            data.pop(key, None)
        _write(_SETTINGS_FILE, data)


# ── Conversations ─────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def conv_list() -> list[dict]:
    data = _read(_CONV_FILE)
    rows = list(data.values())
    rows.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
    return [{"id": r["id"], "title": r["title"], "model_used": r.get("model_used", ""),
             "updated_at": r.get("updated_at", "")} for r in rows]


def conv_save(messages: list, model_used: str, title: str) -> str:
    with _lock:
        data = _read(_CONV_FILE)
        cid = str(uuid.uuid4())
        data[cid] = {"id": cid, "title": title[:255], "model_used": model_used,
                     "messages": messages, "updated_at": _now()}
        _write(_CONV_FILE, data)
    return cid


def conv_load(conv_id: str) -> dict | None:
    return _read(_CONV_FILE).get(conv_id)


def conv_update(conv_id: str, messages: list, model_used: str) -> bool:
    with _lock:
        data = _read(_CONV_FILE)
        if conv_id not in data:
            return False
        data[conv_id]["messages"] = messages
        data[conv_id]["model_used"] = model_used
        data[conv_id]["updated_at"] = _now()
        _write(_CONV_FILE, data)
    return True


def conv_delete(conv_id: str) -> bool:
    with _lock:
        data = _read(_CONV_FILE)
        if conv_id not in data:
            return False
        del data[conv_id]
        _write(_CONV_FILE, data)
    return True
