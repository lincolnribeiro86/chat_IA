"""Model catalog and provider factory."""
from __future__ import annotations
from typing import Any
from config import settings

# ── Static catalog ─────────────────────────────────────────────────────────────
# Each entry: id → {name, provider, supports_vision, supports_tools, context_window}

CATALOG: list[dict] = [
    # Ollama Cloud — usage_tier: low | medium | high | extra_high
    # Low Usage (plano free, mais leve)
    {"id": "gpt-oss:20b-cloud",             "name": "GPT OSS 20B",               "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 128000, "usage_tier": "low"},
    # Medium Usage
    {"id": "gemma4:cloud",                  "name": "Gemma 4",                   "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 256000, "usage_tier": "medium"},
    {"id": "gemma4:31b-cloud",              "name": "Gemma 4 31B",               "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 256000, "usage_tier": "medium"},
    {"id": "qwen3.5:cloud",                 "name": "Qwen3.5",                   "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 256000, "usage_tier": "medium"},
    {"id": "deepseek-v4-flash:cloud",       "name": "DeepSeek V4 Flash",         "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 1000000,"usage_tier": "medium"},
    {"id": "nemotron-3-super:cloud",        "name": "Nemotron 3 Super 120B",     "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 256000, "usage_tier": "medium"},
    {"id": "minimax-m2.5:cloud",            "name": "MiniMax M2.5",              "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 198000, "usage_tier": "medium"},
    {"id": "minimax-m2.7:cloud",            "name": "MiniMax M2.7",              "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 256000, "usage_tier": "medium"},
    {"id": "gpt-oss:120b-cloud",            "name": "GPT OSS 120B",              "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 128000, "usage_tier": "medium"},
    # High Usage
    {"id": "kimi-k2.5:cloud",              "name": "Kimi K2.5",                 "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 256000, "usage_tier": "high"},
    {"id": "kimi-k2.6:cloud",              "name": "Kimi K2.6",                 "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 256000, "usage_tier": "high"},
    {"id": "kimi-k2.7-code:cloud",         "name": "Kimi K2.7 Code",            "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 256000, "usage_tier": "high"},
    {"id": "minimax-m3:cloud",             "name": "MiniMax M3",                "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 512000, "usage_tier": "high"},
    {"id": "nemotron-3-ultra:cloud",       "name": "Nemotron 3 Ultra",          "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 256000, "usage_tier": "high"},
    {"id": "qwen3-coder:cloud",            "name": "Qwen3 Coder 480B",          "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 256000, "usage_tier": "high"},
    {"id": "glm-4.7:cloud",               "name": "GLM-4.7",                   "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 256000, "usage_tier": "high"},
    {"id": "glm-5:cloud",                 "name": "GLM-5",                     "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 198000, "usage_tier": "high"},
    {"id": "glm-5.1:cloud",               "name": "GLM-5.1",                   "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 256000, "usage_tier": "high"},
    {"id": "glm-5.2:cloud",               "name": "GLM-5.2",                   "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 976000, "usage_tier": "high"},
    # Extra High Usage (Pro/Max)
    {"id": "deepseek-v4-pro:cloud",        "name": "DeepSeek V4 Pro",           "provider": "ollama", "supports_vision": False, "supports_tools": True,  "context_window": 1000000,"usage_tier": "extra_high"},
    {"id": "gemini-3-flash-preview:cloud", "name": "Gemini 3 Flash Preview",    "provider": "ollama", "supports_vision": True,  "supports_tools": True,  "context_window": 1000000,"usage_tier": "extra_high"},
    # OpenAI — GPT-5
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
        # Fallback: qualquer modelo não catalogado é tratado como Ollama
        # (modelos puxados localmente aparecem no fetch dinâmico mas não no catálogo)
        from providers.ollama_provider import build_ollama
        ollama_url = key("ollama_base_url") or cfg.ollama_base_url
        ollama_key = key("ollama_api_key")
        return build_ollama(model_id, ollama_url, ollama_key)

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
