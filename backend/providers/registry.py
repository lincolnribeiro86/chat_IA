"""Model catalog and provider factory."""
from __future__ import annotations
from typing import Any
from config import settings

# ── Static catalog ─────────────────────────────────────────────────────────────
# Each entry: id → {name, provider, supports_vision, supports_tools, context_window}

CATALOG: list[dict] = [
    # Ollama — Geral
    {"id": "llama4:16x17b",               "name": "Llama 4 16x17B",           "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 128000},
    {"id": "llama4:128x17b",              "name": "Llama 4 128x17B",          "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 128000},
    {"id": "llama3.3:70b",                "name": "Llama 3.3 70B",            "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 128000},
    {"id": "llama3.2:3b",                 "name": "Llama 3.2 3B",             "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 8192},
    {"id": "llama3.2-vision:11b",         "name": "Llama 3.2 Vision 11B",    "provider": "ollama", "supports_vision": True,  "supports_tools": False, "context_window": 8192},
    {"id": "qwen3:8b",                    "name": "Qwen3 8B",                  "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 32768},
    {"id": "qwen3:32b",                   "name": "Qwen3 32B",                 "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 32768},
    {"id": "qwen3.5:8b",                  "name": "Qwen3.5 8B",               "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 32768},
    {"id": "qwen3.5:32b",                 "name": "Qwen3.5 32B",              "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 32768},
    {"id": "qwen3.6:27b",                 "name": "Qwen3.6 27B",              "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 32768},
    {"id": "mistral-small3.2:24b",        "name": "Mistral Small 3.2 24B",   "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 32768},
    {"id": "gemma4:12b",                  "name": "Gemma4 12B",               "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 128000},
    {"id": "gemma4:27b",                  "name": "Gemma4 27B",               "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 128000},
    # Ollama — Código
    {"id": "qwen2.5-coder:7b",            "name": "Qwen2.5 Coder 7B",        "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 32768},
    {"id": "qwen2.5-coder:32b",           "name": "Qwen2.5 Coder 32B",       "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 32768},
    {"id": "qwen3-coder:30b",             "name": "Qwen3 Coder 30B",         "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 32768},
    {"id": "deepseek-coder-v2:16b",       "name": "DeepSeek Coder V2 16B",   "provider": "ollama", "supports_vision": False, "supports_tools": False, "context_window": 128000},
    # Ollama — Raciocínio
    {"id": "deepseek-r1:8b",              "name": "DeepSeek R1 8B",           "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 32768},
    {"id": "deepseek-r1:14b",             "name": "DeepSeek R1 14B",          "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 32768},
    {"id": "deepseek-r1:32b",             "name": "DeepSeek R1 32B",          "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 32768},
    {"id": "deepseek-v3.2",               "name": "DeepSeek V3.2",            "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 128000},
    {"id": "qwq:32b",                     "name": "QwQ 32B",                  "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 32768},
    # Ollama — Visão
    {"id": "qwen3-vl:7b",                 "name": "Qwen3 VL 7B",              "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 32768},
    {"id": "qwen2.5vl:7b",               "name": "Qwen2.5 VL 7B",            "provider": "ollama", "supports_vision": True,  "supports_tools": False, "context_window": 32768},
    {"id": "llava:7b",                    "name": "LLaVA 7B",                 "provider": "ollama", "supports_vision": True,  "supports_tools": False, "context_window": 4096},
    {"id": "minicpm-v:8b",                "name": "MiniCPM-V 8B",             "provider": "ollama", "supports_vision": True,  "supports_tools": False, "context_window": 8192},
    # OpenAI
    {"id": "gpt-4o",                      "name": "GPT-4o",                    "provider": "openai",      "supports_vision": True,  "supports_tools": True,  "context_window": 128000},
    {"id": "gpt-4o-mini-2024-07-18",      "name": "GPT-4o Mini",              "provider": "openai",      "supports_vision": True,  "supports_tools": True,  "context_window": 128000},
    {"id": "gpt-4.1-2025-04-14",          "name": "GPT-4.1",                  "provider": "openai",      "supports_vision": True,  "supports_tools": True,  "context_window": 1047576},
    {"id": "gpt-4.1-mini-2025-04-14",     "name": "GPT-4.1 Mini",            "provider": "openai",      "supports_vision": True,  "supports_tools": True,  "context_window": 1047576},
    {"id": "gpt-4.1-nano-2025-04-14",     "name": "GPT-4.1 Nano",            "provider": "openai",      "supports_vision": True,  "supports_tools": True,  "context_window": 1047576},
    {"id": "gpt-4-turbo",                 "name": "GPT-4 Turbo",              "provider": "openai",      "supports_vision": True,  "supports_tools": True,  "context_window": 128000},
    {"id": "gpt-4",                       "name": "GPT-4",                     "provider": "openai",      "supports_vision": False, "supports_tools": True,  "context_window": 32000},
    {"id": "gpt-3.5-turbo",              "name": "GPT-3.5 Turbo",             "provider": "openai",      "supports_vision": False, "supports_tools": True,  "context_window": 16385},
    {"id": "o3-2025-04-16",               "name": "o3",                        "provider": "openai",      "supports_vision": False, "supports_tools": True,  "context_window": 200000},
    {"id": "o4-mini-2025-04-16",          "name": "o4 Mini",                  "provider": "openai",      "supports_vision": False, "supports_tools": True,  "context_window": 200000},
    {"id": "gpt-5.5",                      "name": "GPT-5.5 ⭐",               "provider": "openai_gpt5", "supports_vision": True,  "supports_tools": True,  "context_window": 256000},
    {"id": "gpt-5.5-pro",                 "name": "GPT-5.5 Pro",              "provider": "openai_gpt5", "supports_vision": True,  "supports_tools": True,  "context_window": 256000},
    {"id": "gpt-5.4",                     "name": "GPT-5.4",                  "provider": "openai_gpt5", "supports_vision": True,  "supports_tools": True,  "context_window": 128000},
    {"id": "gpt-5.4-pro",                 "name": "GPT-5.4 Pro",              "provider": "openai_gpt5", "supports_vision": True,  "supports_tools": True,  "context_window": 128000},
    {"id": "gpt-5.4-mini",                "name": "GPT-5.4 Mini",             "provider": "openai_gpt5", "supports_vision": True,  "supports_tools": True,  "context_window": 128000},
    {"id": "gpt-5.4-nano",                "name": "GPT-5.4 Nano",             "provider": "openai_gpt5", "supports_vision": True,  "supports_tools": True,  "context_window": 128000},
    # Anthropic API Key
    {"id": "claude-3-5-haiku-20241022",   "name": "Claude Haiku 3.5",         "provider": "anthropic",   "supports_vision": True,  "supports_tools": True,  "context_window": 200000},
    {"id": "claude-3-5-sonnet-20241022",  "name": "Claude Sonnet 3.5",        "provider": "anthropic",   "supports_vision": True,  "supports_tools": True,  "context_window": 200000},
    {"id": "claude-sonnet-3-7-20250219",  "name": "Claude Sonnet 3.7",        "provider": "anthropic",   "supports_vision": True,  "supports_tools": True,  "context_window": 200000},
    {"id": "claude-sonnet-4-5",           "name": "Claude Sonnet 4.5",        "provider": "anthropic",   "supports_vision": True,  "supports_tools": True,  "context_window": 200000},
    {"id": "claude-sonnet-4-6",           "name": "Claude Sonnet 4.6",        "provider": "anthropic",   "supports_vision": True,  "supports_tools": True,  "context_window": 200000},
    {"id": "claude-opus-4-5",             "name": "Claude Opus 4.5",          "provider": "anthropic",   "supports_vision": True,  "supports_tools": True,  "context_window": 200000},
    {"id": "claude-opus-4-8",             "name": "Claude Opus 4.8",          "provider": "anthropic",   "supports_vision": True,  "supports_tools": True,  "context_window": 200000},
    # Claude Subscription (Agent SDK)
    {"id": "claude-sub-sonnet",           "name": "Claude Sonnet (Assinatura)","provider": "claude_sub",  "supports_vision": False, "supports_tools": False, "context_window": 200000},
    {"id": "claude-sub-opus",             "name": "Claude Opus (Assinatura)", "provider": "claude_sub",  "supports_vision": False, "supports_tools": False, "context_window": 200000},
    # Gemini
    {"id": "gemini-2.5-flash",            "name": "Gemini 2.5 Flash",         "provider": "gemini",      "supports_vision": True,  "supports_tools": True,  "context_window": 1048576},
    {"id": "gemini-2.5-pro",              "name": "Gemini 2.5 Pro",           "provider": "gemini",      "supports_vision": True,  "supports_tools": True,  "context_window": 1048576},
    {"id": "gemini-2.0-flash",            "name": "Gemini 2.0 Flash",         "provider": "gemini",      "supports_vision": True,  "supports_tools": True,  "context_window": 1000000},
    {"id": "gemini-1.5-flash",            "name": "Gemini 1.5 Flash",         "provider": "gemini",      "supports_vision": True,  "supports_tools": True,  "context_window": 1000000},
    {"id": "gemini-1.5-pro",              "name": "Gemini 1.5 Pro",           "provider": "gemini",      "supports_vision": True,  "supports_tools": True,  "context_window": 2097152},
    # DeepSeek API (api.deepseek.com)
    {"id": "deepseek-chat",     "name": "DeepSeek V3",         "provider": "deepseek", "supports_vision": False, "supports_tools": True,  "context_window": 64000},
    {"id": "deepseek-reasoner", "name": "DeepSeek R1",         "provider": "deepseek", "supports_vision": False, "supports_tools": False, "context_window": 64000},
    {"id": "deepseek-v4-flash", "name": "DeepSeek V4 Flash",   "provider": "deepseek", "supports_vision": True,  "supports_tools": True,  "context_window": 128000},
    {"id": "deepseek-v4-pro",   "name": "DeepSeek V4 Pro",     "provider": "deepseek", "supports_vision": True,  "supports_tools": True,  "context_window": 128000},
    # Groq
    {"id": "deepseek-r1-distill-llama-70b",                  "name": "DeepSeek R1 Llama 70B (Groq)", "provider": "groq", "supports_vision": False, "supports_tools": False, "context_window": 128000},
    {"id": "meta-llama/llama-4-maverick-17b-128e-instruct",  "name": "Llama 4 Maverick (Groq)",      "provider": "groq", "supports_vision": True,  "supports_tools": True,  "context_window": 131072},
    {"id": "meta-llama/llama-4-scout-17b-16e-instruct",      "name": "Llama 4 Scout (Groq)",         "provider": "groq", "supports_vision": True,  "supports_tools": True,  "context_window": 131072},
    {"id": "qwen/qwen3-32b",                                 "name": "Qwen3 32B (Groq)",             "provider": "groq", "supports_vision": False, "supports_tools": True,  "context_window": 32768},
]

_CATALOG_BY_ID = {m["id"]: m for m in CATALOG}


def get_model_info(model_id: str) -> dict | None:
    return _CATALOG_BY_ID.get(model_id)


def build_provider(model_id: str, api_keys: dict[str, str] = None, temperature: float = 0.5):
    """Return a LangChain chat model instance for the given model_id.
    api_keys overrides env settings (passed from UI/session)."""
    from config import settings as cfg

    def key(k):
        return (api_keys or {}).get(k) or getattr(cfg, k, None)

    info = _CATALOG_BY_ID.get(model_id)
    if not info:
        raise ValueError(f"Unknown model: {model_id}")

    provider = info["provider"]

    if provider == "ollama":
        from providers.ollama_provider import build_ollama
        return build_ollama(model_id, key("ollama_base_url") or cfg.ollama_base_url, key("ollama_api_key"))

    if provider in ("openai", "openai_gpt5"):
        from providers.openai_provider import build_openai, build_gpt5
        ak = key("openai_api_key")
        if not ak:
            raise ValueError("OpenAI API key not configured")
        if provider == "openai_gpt5":
            return build_gpt5(model_id, ak, temperature)
        return build_openai(model_id, ak, temperature)

    if provider == "anthropic":
        from providers.anthropic_provider import build_anthropic
        ak = key("anthropic_api_key")
        if not ak:
            raise ValueError("Anthropic API key not configured")
        return build_anthropic(model_id, ak, temperature)

    if provider == "claude_sub":
        from providers.claude_subscription import build_claude_sub
        token = key("claude_code_oauth_token")
        if not token:
            raise ValueError("CLAUDE_CODE_OAUTH_TOKEN not configured")
        return build_claude_sub(model_id, token)

    if provider == "gemini":
        from providers.gemini_provider import build_gemini
        ak = key("gemini_api_key")
        if not ak:
            raise ValueError("Gemini API key not configured")
        return build_gemini(model_id, ak, temperature)

    if provider == "groq":
        from providers.groq_provider import build_groq
        ak = key("groq_api_key")
        if not ak:
            raise ValueError("Groq API key not configured")
        return build_groq(model_id, ak, temperature)

    if provider == "openrouter":
        from providers.openrouter_provider import build_openrouter
        ak = key("openrouter_api_key")
        if not ak:
            raise ValueError("OpenRouter API key not configured")
        return build_openrouter(model_id, ak, temperature)

    if provider == "deepseek":
        from providers.deepseek_provider import build_deepseek
        ak = key("deepseek_api_key")
        if not ak:
            raise ValueError("DeepSeek API key not configured")
        return build_deepseek(model_id, ak, temperature)

    raise ValueError(f"Unknown provider: {provider}")
