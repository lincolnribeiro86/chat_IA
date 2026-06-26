"""File-based key-value store used as fallback when PostgreSQL is unavailable."""
import json
import os
import threading

_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "settings.json")
_lock = threading.Lock()


def _path() -> str:
    return os.path.abspath(_STORE_PATH)


def load_all() -> dict[str, str]:
    try:
        with open(_path(), encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if isinstance(v, str)}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def get(key: str) -> str | None:
    return load_all().get(key)


def set(key: str, value: str) -> None:
    with _lock:
        data = load_all()
        if value:
            data[key] = value
        else:
            data.pop(key, None)
        os.makedirs(os.path.dirname(_path()), exist_ok=True)
        with open(_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
