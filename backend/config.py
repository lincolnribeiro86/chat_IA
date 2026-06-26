from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Auth
    app_password: str = "changeme"
    jwt_secret: str = "please-change-this-secret-key-in-production"
    jwt_expire_hours: int = 72

    # Database
    db_name: str = "chatia"
    db_user: str = "chatia"
    db_password: str = "chatia"
    db_host: str = "postgres"
    db_port: int = 5432

    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    ollama_base_url: str = "https://ollama.com"
    ollama_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    firecrawl_api_key: Optional[str] = None
    claude_code_oauth_token: Optional[str] = None

    # RAG
    chroma_persist_dir: str = "./data/chroma"
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_top_k: int = 5
    rag_threshold_chars: int = 8000



settings = Settings()
