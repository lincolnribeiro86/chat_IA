"""DeepSeek provider — OpenAI-compatible API at api.deepseek.com."""
from langchain_openai import ChatOpenAI


def build_deepseek(model_id: str, api_key: str, temperature: float = 0.5) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_id,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=temperature,
        streaming=True,
    )
