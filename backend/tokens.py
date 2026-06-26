"""Token limits and cost estimation per model."""

# context window in tokens
CONTEXT_LIMITS: dict[str, int] = {
    # Ollama local
    "deepseek-r1:8b": 8192, "deepseek-coder:6.7b": 8192,
    "llama3.1": 8192, "llama3.2": 8192,
    "qwen3:8b": 32768, "qwen2.5-coder:7b": 32768,
    "gemma3:4b": 128000,
    "phi4-mini-reasoning:3.8b": 128000, "phi4-mini:3.8b": 128000, "phi3.5:3.8b": 8192,
    "granite3.3:8b": 128000,
    # OpenAI
    "gpt-4.1-2025-04-14": 1047576, "gpt-4.1-mini-2025-04-14": 1047576,
    "gpt-4.1-nano-2025-04-14": 1047576,
    "gpt-4o": 128000, "gpt-4o-mini-2024-07-18": 128000,
    "gpt-4": 32000, "gpt-4-turbo": 128000, "gpt-3.5-turbo": 16385,
    "o3-2025-04-16": 200000, "o4-mini-2025-04-16": 200000,
    "gpt-5": 128000, "gpt-5-mini": 128000, "gpt-5-nano": 128000,
    # Anthropic
    "claude-haiku-3-5-20241022": 200000, "claude-3-5-haiku-20241022": 200000,
    "claude-sonnet-3-5-20241022": 200000, "claude-3-5-sonnet-20241022": 200000,
    "claude-sonnet-3-7-20250219": 200000,
    "claude-sonnet-4-5": 200000, "claude-sonnet-4-6": 200000,
    "claude-opus-4-5": 200000, "claude-opus-4-8": 200000,
    # Gemini
    "gemini-1.5-flash": 1000000, "gemini-1.5-pro": 2097152,
    "gemini-2.0-flash": 1000000, "gemini-2.0-flash-lite": 1000000,
    "gemini-2.5-pro": 1048576, "gemini-2.5-flash": 1048576,
    "gemini-2.5-flash-lite-preview-06-17": 1048576,
    # Groq
    "deepseek-r1-distill-llama-70b": 128000,
    "meta-llama/llama-4-maverick-17b-128e-instruct": 131072,
    "meta-llama/llama-4-scout-17b-16e-instruct": 131072,
    "qwen/qwen3-32b": 32768,
}

# Cost per 1M tokens (input, output) in USD
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00), "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4.1-2025-04-14": (2.00, 8.00), "gpt-4.1-mini-2025-04-14": (0.40, 1.60),
    "gpt-4.1-nano-2025-04-14": (0.10, 0.40),
    "o3-2025-04-16": (10.00, 40.00), "o4-mini-2025-04-16": (1.10, 4.40),
    "gpt-4-turbo": (10.00, 30.00), "gpt-4": (30.00, 60.00), "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-5": (3.00, 12.00), "gpt-5-mini": (0.60, 2.40), "gpt-5-nano": (0.15, 0.60),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-sonnet-3-7-20250219": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00), "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-5": (15.00, 75.00), "claude-opus-4-8": (15.00, 75.00),
    "gemini-1.5-flash": (0.075, 0.30), "gemini-1.5-pro": (1.25, 5.00),
    "gemini-2.0-flash": (0.10, 0.40), "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "deepseek-r1-distill-llama-70b": (0.59, 0.79),
    "meta-llama/llama-4-maverick-17b-128e-instruct": (0.20, 0.60),
    "qwen/qwen3-32b": (0.29, 0.59),
}


def get_context_limit(model_id: str) -> int:
    return CONTEXT_LIMITS.get(model_id, 8192)


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = PRICING.get(model_id)
    if not pricing:
        return None
    cost = (input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000
    return round(cost, 6)


def tokens_to_a4_pages(tokens: int) -> int:
    return round(tokens / 475)


def truncate_by_tokens(text: str, limit_tokens: int, chars_per_token: int = 4) -> str:
    return text[: limit_tokens * chars_per_token]
