from langchain_anthropic import ChatAnthropic


def build_anthropic(model_id: str, api_key: str, temperature: float = 0.5) -> ChatAnthropic:
    return ChatAnthropic(
        model=model_id,
        api_key=api_key,
        temperature=temperature,
        streaming=True,
        stream_usage=True,
    )
