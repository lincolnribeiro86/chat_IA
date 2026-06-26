from fastapi import APIRouter, Depends
from auth import require_auth
from providers.registry import CATALOG
from providers.ollama_provider import fetch_ollama_models
from providers.openrouter_provider import fetch_openrouter_models
from config import settings
from persistence import repository as repo

router = APIRouter()


@router.get("/models")
async def list_models(_user=Depends(require_auth)):
    static = list(CATALOG)

    # Resolve Ollama key: UI settings tem prioridade sobre .env
    ollama_key = repo.get_setting("ollama_api_key") or settings.ollama_api_key
    ollama_url = repo.get_setting("ollama_base_url") or settings.ollama_base_url

    # Só busca modelos locais se NÃO houver API key (cloud)
    if not ollama_key:
        try:
            ollama_live = await fetch_ollama_models(ollama_url, None)
            if ollama_live:
                existing_ids = {m["id"] for m in static if m["provider"] == "ollama"}
                for m in ollama_live:
                    if m["id"] not in existing_ids:
                        static.append(m)
        except Exception:
            pass

    # Try to fetch OpenRouter models if key is present
    openrouter_key = repo.get_setting("openrouter_api_key") or settings.openrouter_api_key
    if openrouter_key:
        try:
            or_models = await fetch_openrouter_models(openrouter_key)
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
