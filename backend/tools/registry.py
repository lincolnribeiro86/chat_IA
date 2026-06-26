"""Bind available tools to a LangChain model if it supports them."""
from tools.web_search import get_available_tools


def bind_tools_if_supported(llm, model_info: dict):
    """Return llm with tools bound, or unchanged if model doesn't support tools."""
    if not model_info.get("supports_tools"):
        return llm, []

    tools = get_available_tools()
    if not tools:
        return llm, []

    try:
        bound = llm.bind_tools(tools)
        return bound, tools
    except Exception:
        return llm, []
