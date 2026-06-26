"""Base types shared by all providers."""
from typing import AsyncIterator, TypedDict, Any


class UsageInfo(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


class SSEEvent(TypedDict, total=False):
    type: str          # "token" | "tool_start" | "tool_result" | "usage" | "done" | "error"
    content: str
    tool_id: str
    tool_name: str
    tool_args: dict
    tool_result: str
    usage: UsageInfo
    message: str


class ProviderConfig(TypedDict, total=False):
    api_key: str
    base_url: str
    temperature: float
    extra: dict
