import pytest

from agent_harness.providers.provider_factory import MockProvider, ProviderFactory


def test_provider_factory_returns_mock_provider() -> None:
    provider = ProviderFactory.create(model_name="mock/default", temperature=0.3, max_tokens=1500)
    assert isinstance(provider, MockProvider)
    assert provider.model == "mock/default"
    assert provider.default_temperature == 0.3
    assert provider.default_max_tokens == 1500


def test_openrouter_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError):
        ProviderFactory.create(model_name="openrouter/openai/gpt-4o", temperature=0.2, max_tokens=2000)
