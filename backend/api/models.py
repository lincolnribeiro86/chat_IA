from fastapi import APIRouter, Depends
from auth import require_auth
from providers.registry import CATALOG
from providers.ollama_provider import fetch_ollama_models
from providers.openrouter_provider import fetch_openrouter_models
from config import settings

router = APIRouter()


@router.get("/models")
async def list_models(_user=Depends(require_auth)):
    static = list(CATALOG)

    # Try to fetch live Ollama models
    try:
        ollama_live = await fetch_ollama_models(
            settings.ollama_base_url, settings.ollama_api_key
        )
        if ollama_live:
            # Merge: add live models not already in static catalog
            existing_ids = {m["id"] for m in static if m["provider"] == "ollama"}
            for m in ollama_live:
                if m["id"] not in existing_ids:
                    static.append(m)
    except Exception:
        pass

    # Try to fetch OpenRouter models if key is present
    if settings.openrouter_api_key:
        try:
            or_models = await fetch_openrouter_models(settings.openrouter_api_key)
            existing_ids = {m["id"] for m in static}
            for m in or_models:
                if m["id"] not in existing_ids:
                    static.append(m)
        except Exception:
            pass

    # Group by provider
    providers: dict[str, dict] = {}
    provider_labels = {
        "ollama": "Ollama (Local/Cloud)",
        "openai": "OpenAI",
        "openai_gpt5": "OpenAI GPT-5",
        "anthropic": "Anthropic (API Key)",
        "claude_sub": "Claude (Assinatura)",
        "gemini": "Google Gemini",
        "groq": "Groq",
        "openrouter": "OpenRouter",
        "deepseek": "DeepSeek",
    }
    for m in static:
        prov = m["provider"]
        if prov not in providers:
            providers[prov] = {"id": prov, "name": provider_labels.get(prov, prov), "models": []}
        providers[prov]["models"].append(m)

    return {"providers": list(providers.values())}
