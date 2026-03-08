from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Model configuration
    primary_model: str = "google/gemini-2.0-flash-001"
    secondary_model: str = "meta-llama/llama-3.3-70b-instruct"

    # RAG configuration
    chroma_persist_dir: str = "./chroma_data"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_retrieval: int = 10
    top_k_final: int = 7

    # Agent configuration
    confidence_threshold: float = 0.60
    max_tokens: int = 2048
    temperature: float = 0.1

    # App configuration
    app_name: str = "Policy-Aware AI Decision Agent"
    debug: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
