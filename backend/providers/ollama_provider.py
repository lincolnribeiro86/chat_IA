"""Ollama provider — local and cloud."""
from langchain_ollama import ChatOllama
import httpx


def build_ollama(model_id: str, base_url: str, api_key: str | None = None) -> ChatOllama:
    kwargs: dict = {"model": model_id, "base_url": base_url}
    if api_key:
        # Ollama Cloud requires Bearer auth
        kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}
    return ChatOllama(**kwargs)


async def fetch_ollama_models(base_url: str, api_key: str | None = None) -> list[dict]:
    """Return local/cloud Ollama models from /api/tags."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=5, headers=headers) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
            data = resp.json()
        models = []
        for m in data.get("models", []):
            name = m.get("name", "")
            models.append({
                "id": name,
                "name": name,
                "provider": "ollama",
                "supports_vision": any(tag in name for tag in ["llava", "moondream", "bakllava", "minicpm-v", "llama3.2"]),
                "supports_tools": any(tag in name for tag in ["llama3", "qwen", "mistral", "granite"]),
                "context_window": 8192,
            })
        return models
    except Exception:
        return []
