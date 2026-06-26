from langchain_groq import ChatGroq


def build_groq(model_id: str, api_key: str, temperature: float = 0.5) -> ChatGroq:
    return ChatGroq(model=model_id, groq_api_key=api_key, temperature=temperature, streaming=True)
