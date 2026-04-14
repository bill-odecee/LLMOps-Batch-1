import importlib
import json
import sys
import types
from types import SimpleNamespace

import pytest

# Provide a fake langchain_google_genai module for imports during tests.
fake_langchain_google_genai = types.ModuleType("langchain_google_genai")
fake_langchain_google_genai.GoogleGenerativeAIEmbeddings = lambda *args, **kwargs: None
fake_langchain_google_genai.ChatGoogleGenerativeAI = lambda *args, **kwargs: None
sys.modules["langchain_google_genai"] = fake_langchain_google_genai

from utils import model_loader


class FakeEmbedding:
    def __init__(self, model: str, google_api_key: str):
        self.model = model
        self.google_api_key = google_api_key


class FakeLLM:
    def __init__(
        self,
        model: str,
        google_api_key: str,
        temperature: float,
        max_output_tokens: int,
    ):
        self.model = model
        self.google_api_key = google_api_key
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens


def test_apikeymanager_loads_api_keys_from_api_keys_json(monkeypatch):
    monkeypatch.setenv("API_KEYS", json.dumps({"GOOGLE_API_KEY": "secret-key"}))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    manager = model_loader.ApiKeyManager()

    assert manager.get("GOOGLE_API_KEY") == "secret-key"


def test_apikeymanager_falls_back_to_individual_env_when_api_keys_invalid(monkeypatch):
    monkeypatch.setenv("API_KEYS", "not-json")
    monkeypatch.setenv("GOOGLE_API_KEY", "fallback-key")

    manager = model_loader.ApiKeyManager()

    assert manager.get("GOOGLE_API_KEY") == "fallback-key"


def test_apikeymanager_raises_when_missing_required_key(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing required API keys"):
        model_loader.ApiKeyManager()


def test_apikeymanager_get_raises_for_unknown_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-key")
    manager = model_loader.ApiKeyManager()

    with pytest.raises(KeyError, match="API key for OTHER_KEY is missing"):
        manager.get("OTHER_KEY")


def test_modelloader_loads_embeddings_and_llm(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-key")
    monkeypatch.setenv("ENV", "local")
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setattr(model_loader, "load_dotenv", lambda override=True: None)
    monkeypatch.setattr(model_loader, "GoogleGenerativeAIEmbeddings", FakeEmbedding)
    monkeypatch.setattr(model_loader, "ChatGoogleGenerativeAI", FakeLLM)
    monkeypatch.setattr(
        model_loader,
        "load_config",
        lambda: {
            "embedding_model": {"model_name": "embed-model"},
            "llm": {
                "google": {
                    "provider": "google",
                    "model_name": "llm-model",
                    "temperature": 0.25,
                    "max_output_tokens": 512,
                }
            },
        },
    )

    loader = model_loader.Modelloader()

    embeddings = loader.load_embeddings()
    assert isinstance(embeddings, FakeEmbedding)
    assert embeddings.model == "embed-model"
    assert embeddings.google_api_key == "secret-key"

    llm = loader.load_llm()
    assert isinstance(llm, FakeLLM)
    assert llm.model == "llm-model"
    assert llm.google_api_key == "secret-key"
    assert llm.temperature == 0.25
    assert llm.max_output_tokens == 512


def test_modelloader_load_llm_raises_when_provider_missing(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-key")
    monkeypatch.setenv("ENV", "local")
    monkeypatch.setattr(model_loader, "load_dotenv", lambda override=True: None)
    monkeypatch.setattr(
        model_loader,
        "load_config",
        lambda: {"embedding_model": {"model_name": "embed-model"}, "llm": {}},
    )

    loader = model_loader.Modelloader()

    with pytest.raises(ValueError, match="LLM provider 'google' not found in config"):
        loader.load_llm()


def test_modelloader_load_llm_raises_for_unsupported_provider(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-key")
    monkeypatch.setenv("ENV", "local")
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setattr(model_loader, "load_dotenv", lambda override=True: None)
    monkeypatch.setattr(
        model_loader,
        "load_config",
        lambda: {
            "embedding_model": {"model_name": "embed-model"},
            "llm": {
                "google": {
                    "provider": "unsupported",
                    "model_name": "llm-model",
                }
            },
        },
    )

    loader = model_loader.Modelloader()

    with pytest.raises(ValueError, match="Unsupported LLM provider: unsupported"):
        loader.load_llm()
