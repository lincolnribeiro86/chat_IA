"""Ollama provider — local (ChatOllama) ou Cloud (OpenAI-compatible)."""
import httpx


def build_ollama(model_id: str, base_url: str, api_key: str | None = None):
    if api_key:
        # Ollama Cloud: usa endpoint OpenAI-compatível
        from langchain_openai import ChatOpenAI
        cloud_url = "https://ollama.com/v1"
        return ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url=cloud_url,
            temperature=0.5,
            streaming=True,
        )
    else:
        # Ollama local
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_id, base_url=base_url)


async def fetch_ollama_models(base_url: str, api_key: str | None = None) -> list[dict]:
    """Busca modelos do Ollama local via /api/tags. Cloud usa catálogo estático."""
    if api_key:
        # Ollama Cloud não tem endpoint de listagem pública — usa catálogo estático
        return []
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
            data = resp.json()
        models = []
        for m in data.get("models", []):
            name = m.get("name", "")
            models.append({
                "id": name,
                "name": name,
                "provider": "ollama",
                "supports_vision": any(t in name for t in ["llava", "moondream", "bakllava", "minicpm-v", "llama3.2-vision", "qwen3-vl", "qwen2.5vl"]),
                "supports_tools": any(t in name for t in ["llama3", "qwen", "mistral", "granite", "phi"]),
                "context_window": 8192,
            })
        return models
    except Exception:
        return []
