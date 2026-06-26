"""OpenAI provider.

GPT-4 / GPT-4.1 / o-series  → chat.completions via LangChain
GPT-5 family                 → Responses API (openai>=1.66) com streaming real
"""
from __future__ import annotations
from langchain_openai import ChatOpenAI
from typing import AsyncIterator


def build_openai(model_id: str, api_key: str, temperature: float = 0.5) -> ChatOpenAI:
    return ChatOpenAI(model=model_id, api_key=api_key, temperature=temperature, streaming=True)


def build_gpt5(model_id: str, api_key: str, temperature: float = 0.5) -> "GPT5Provider":
    return GPT5Provider(model_id, api_key, temperature)


class GPT5Provider:
    """Wrapper para a Responses API do OpenAI (família GPT-5).
    Usa streaming real via response.output_text.delta."""

    _IS_GPT5 = True

    def __init__(self, model_id: str, api_key: str, temperature: float = 0.5):
        self._model_id = model_id
        self._api_key = api_key
        self._temperature = temperature

    async def astream(self, lc_messages) -> AsyncIterator:
        """Converte mensagens LangChain → Responses API e faz streaming."""
        from openai import AsyncOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        client = AsyncOpenAI(api_key=self._api_key)

        # Monta input no formato da Responses API
        input_msgs = []
        system_text = None
        for m in lc_messages:
            if isinstance(m, SystemMessage):
                system_text = m.content if isinstance(m.content, str) else str(m.content)
            elif isinstance(m, HumanMessage):
                content = m.content if isinstance(m.content, str) else str(m.content)
                input_msgs.append({"role": "user", "content": content})
            elif isinstance(m, AIMessage):
                content = m.content if isinstance(m.content, str) else str(m.content)
                input_msgs.append({"role": "assistant", "content": content})

        kwargs: dict = dict(
            model=self._model_id,
            input=input_msgs,
            stream=True,
        )
        if system_text:
            kwargs["instructions"] = system_text

        stream = await client.responses.create(**kwargs)

        # Emite chunks compatíveis com o loop do chat.py
        async for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield _FakeChunk(delta)
            elif event_type == "response.completed":
                resp = getattr(event, "response", None)
                if resp and hasattr(resp, "usage"):
                    yield _FakeChunk("", usage={
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                    })


class _FakeChunk:
    """Chunk mínimo compatível com o loop de streaming do chat.py."""
    def __init__(self, text: str, usage: dict | None = None):
        self.content = text
        self.usage_metadata = usage
        self.tool_call_chunks = []
