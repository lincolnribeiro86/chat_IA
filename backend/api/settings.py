"""User-configurable settings (API keys, password) stored in DB with .env fallback."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from auth import require_auth, create_access_token, COOKIE_NAME
from fastapi.responses import JSONResponse
from persistence import repository as repo
from config import settings

router = APIRouter()

# Keys that can be stored in DB (never return actual values to client)
MANAGED_KEYS = [
    "openai_api_key", "anthropic_api_key", "gemini_api_key", "groq_api_key",
    "openrouter_api_key", "ollama_base_url", "ollama_api_key",
    "tavily_api_key", "firecrawl_api_key", "claude_code_oauth_token",
]


def _resolve_key(key: str) -> str | None:
    """DB value takes precedence over env."""
    db_val = repo.get_setting(key)
    if db_val:
        return db_val
    return getattr(settings, key, None)


@router.get("/settings")
def get_settings(_user=Depends(require_auth)):
    """Return which keys are configured (true/false), not the values."""
    db = repo.get_all_settings()
    result = {}
    for k in MANAGED_KEYS:
        val = db.get(k) or getattr(settings, k, None)
        # Return the value for non-secret fields
        if k == "ollama_base_url":
            result[k] = val or settings.ollama_base_url
        else:
            result[k] = bool(val)
    return result


class KeysUpdate(BaseModel):
    keys: dict[str, str]


@router.put("/settings/keys")
def update_keys(req: KeysUpdate, _user=Depends(require_auth)):
    """Save / clear API keys in DB."""
    for k, v in req.keys.items():
        if k not in MANAGED_KEYS:
            continue
        if v:
            repo.set_setting(k, v)
        else:
            repo.set_setting(k, "")
    return {"ok": True}


class PasswordChange(BaseModel):
    new_password: str


@router.put("/settings/password")
def change_password(req: PasswordChange, _user=Depends(require_auth)):
    if not req.new_password or len(req.new_password) < 4:
        raise HTTPException(400, "Password too short (min 4 chars)")
    repo.set_setting("app_password", req.new_password)
    return {"ok": True}
