"""OpenRouter provider — OpenAI-compatible API."""
from langchain_openai import ChatOpenAI
import httpx


def build_openrouter(model_id: str, api_key: str, temperature: float = 0.5) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_id,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        streaming=True,
        default_headers={
            "HTTP-Referer": "https://github.com/lincolnribeiro86/chat_IA",
            "X-Title": "chat_IA",
        },
    )


async def fetch_openrouter_models(api_key: str) -> list[dict]:
    """Return simplified model list from OpenRouter API."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            data = resp.json()
        models = []
        for m in data.get("data", []):
            ctx = m.get("context_length", 8192)
            models.append({
                "id": m["id"],
                "name": m.get("name", m["id"]),
                "provider": "openrouter",
                "supports_vision": "image" in str(m.get("architecture", {}).get("modality", "")),
                "supports_tools": True,
                "context_window": ctx,
            })
        return models
    except Exception:
        return []
