"""Ollama provider — Cloud (native client) ou local (ChatOllama)."""
import httpx
from typing import AsyncIterator
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def build_ollama(model_id: str, base_url: str, api_key: str | None = None):
    if api_key:
        return OllamaCloudProvider(model_id, api_key)
    # Local
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model_id, base_url=base_url)


class OllamaCloudProvider:
    """Usa o cliente nativo ollama para o Ollama Cloud (https://ollama.com)."""

    _IS_OLLAMA_CLOUD = True

    def __init__(self, model_id: str, api_key: str):
        self._model_id = model_id
        self._api_key = api_key

    async def astream(self, lc_messages) -> AsyncIterator:
        from ollama import AsyncClient

        client = AsyncClient(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

        messages = []
        for m in lc_messages:
            if isinstance(m, SystemMessage):
                messages.append({"role": "system", "content": m.content if isinstance(m.content, str) else str(m.content)})
            elif isinstance(m, HumanMessage):
                messages.append({"role": "user", "content": m.content if isinstance(m.content, str) else str(m.content)})
            elif isinstance(m, AIMessage):
                messages.append({"role": "assistant", "content": m.content if isinstance(m.content, str) else str(m.content)})

        thinking_started = False
        thinking_ended = False

        async for part in await client.chat(
            model=self._model_id,
            messages=messages,
            stream=True,
        ):
            # Support both dict and pydantic object (ollama 0.6+)
            if isinstance(part, dict):
                msg = part.get("message", {})
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                thinking = msg.get("thinking", "") if isinstance(msg, dict) else getattr(msg, "thinking", "")
                done = part.get("done", False)
                prompt_tokens = part.get("prompt_eval_count", 0) if done else 0
                completion_tokens = part.get("eval_count", 0) if done else 0
            else:
                msg = getattr(part, "message", None)
                content = getattr(msg, "content", "") if msg is not None else ""
                thinking = getattr(msg, "thinking", "") if msg is not None else ""
                done = getattr(part, "done", False)
                prompt_tokens = getattr(part, "prompt_eval_count", 0) if done else 0
                completion_tokens = getattr(part, "eval_count", 0) if done else 0

            # Wrap thinking tokens in a collapsible markdown block
            if thinking:
                if not thinking_started:
                    yield _FakeChunk("<details>\n<summary>💭 Raciocínio</summary>\n\n")
                    thinking_started = True
                yield _FakeChunk(thinking)

            # Close thinking block when content starts
            if content and thinking_started and not thinking_ended:
                yield _FakeChunk("\n</details>\n\n")
                thinking_ended = True

            if content:
                yield _FakeChunk(content)

            if done:
                # Close any open thinking block
                if thinking_started and not thinking_ended:
                    yield _FakeChunk("\n</details>\n\n")
                    thinking_ended = True
                if prompt_tokens or completion_tokens:
                    yield _FakeChunk("", usage={
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                    })


class _FakeChunk:
    def __init__(self, text: str, usage: dict | None = None):
        self.content = text
        self.usage_metadata = usage
        self.tool_call_chunks = []


async def fetch_ollama_models(base_url: str, api_key: str | None = None) -> list[dict]:
    """Busca modelos do Ollama local via /api/tags. Cloud usa catálogo estático."""
    if api_key:
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
