"""OpenAI standard + GPT-5 (Responses API) provider."""
from langchain_openai import ChatOpenAI
from openai import OpenAI
from typing import Any


def build_openai(model_id: str, api_key: str, temperature: float = 0.5) -> ChatOpenAI:
    return ChatOpenAI(model=model_id, api_key=api_key, temperature=temperature, streaming=True)


def build_gpt5(api_key: str) -> "GPT5Provider":
    return GPT5Provider(api_key)


class GPT5Provider:
    """Wrapper for the OpenAI Responses API (GPT-5 family).
    Exposes the same .stream() interface as LangChain chat models."""

    _IS_GPT5 = True  # flag checked in chat endpoint

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-5"

    def set_model(self, model_id: str):
        self.model = model_id

    def invoke_with_verbosity(
        self,
        messages: list[dict],
        verbosity: str = "medium",
    ) -> dict[str, Any]:
        resp = self.client.responses.create(
            model=self.model,
            input=messages,
            text={"verbosity": verbosity},
        )
        text = ""
        for item in resp.output:
            if hasattr(item, "content"):
                for c in item.content:
                    if hasattr(c, "text"):
                        text += c.text
        return {
            "content": text,
            "usage": {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        }

    def invoke_minimal_reasoning(self, messages: list[dict]) -> dict[str, Any]:
        resp = self.client.responses.create(
            model=self.model,
            input=messages,
            reasoning={"effort": "minimal"},
        )
        text = ""
        for item in resp.output:
            if hasattr(item, "content"):
                for c in item.content:
                    if hasattr(c, "text"):
                        text += c.text
        return {
            "content": text,
            "usage": {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        }

    # Fallback streaming via chat.completions for UI streaming support
    def stream(self, messages):
        from langchain_openai import ChatOpenAI
        lc = ChatOpenAI(model=self.model, api_key=self.client.api_key, streaming=True)
        return lc.stream(messages)
