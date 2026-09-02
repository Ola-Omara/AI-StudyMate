import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("ai_studymate.config")

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
DEFAULT_VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "vector_store"

NOTEBOOK_CONFIG_TO_SETTING = {
    "embedding_model": "embedding_model_name",
    "ollama_model": "ollama_model",
    "collection_name": "collection_name",
    "source_pool_size": "source_pool_size",
    "fused_candidate_pool": "fused_candidate_pool",
    "final_top_k": "final_top_k",
    "rrf_k": "rrf_k",
    "llm_temperature": "llm_temperature",
}


def load_notebook_config(vector_store_path: Path) -> dict:
    config_path = vector_store_path / "config.json"
    if not config_path.exists():
        logger.warning(
            "Notebook config.json not found at %s; falling back to built-in defaults "
            "that mirror the Phase 2 notebook values. Run the notebook's 2.7 Export "
            "cell to generate the real exported config.",
            config_path,
        )
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    vector_store_path: Path = DEFAULT_VECTOR_STORE_PATH
    ollama_host: str = "http://localhost:11434"
    frontend_origin: str = "http://localhost:8501"
    app_env: str = "development"

    embedding_model_name: str = "all-MiniLM-L6-v2"
    ollama_model: str = "llama3.2:3b"
    collection_name: str = "ai_studymate_ml_dl"
    source_pool_size: int = 20
    fused_candidate_pool: int = 12
    final_top_k: int = 5
    rrf_k: int = 60
    llm_temperature: float = 0.0

    def frontend_origins(self) -> list[str]:
        return [origin.strip() for origin in self.frontend_origin.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    base_settings = Settings()
    notebook_config = load_notebook_config(base_settings.vector_store_path)
    overrides = {}
    for config_key, setting_field in NOTEBOOK_CONFIG_TO_SETTING.items():
        env_var_name = setting_field.upper()
        if env_var_name not in os.environ and config_key in notebook_config:
            overrides[setting_field] = notebook_config[config_key]
    if overrides:
        return base_settings.model_copy(update=overrides)
    return base_settings
