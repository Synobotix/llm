from llm.distillation import teachers


def test_teachers_list_respects_openrouter_fallback_limit():
    assert 1 <= len(teachers.TEACHERS) <= 3
    assert all(isinstance(model, str) and "/" in model for model in teachers.TEACHERS)


def test_generate_with_teachers_delegates_to_openrouter(monkeypatch):
    received = {}

    def fake_generate_response(models, prompt, temperature, max_tokens):
        received.update(
            models=models, prompt=prompt, temperature=temperature,
            max_tokens=max_tokens,
        )
        return "réponse", "vendor/model"

    monkeypatch.setattr(teachers, "generate_response", fake_generate_response)

    result = teachers.generate_with_teachers(
        "mon prompt", temperature=0.1, max_tokens=99
    )

    assert result == ("réponse", "vendor/model")
    assert received == {
        "models": teachers.TEACHERS,
        "prompt": "mon prompt",
        "temperature": 0.1,
        "max_tokens": 99,
    }


def test_default_sampling_parameters(monkeypatch):
    received = {}
    monkeypatch.setattr(
        teachers,
        "generate_response",
        lambda **kwargs: received.update(kwargs) or ("r", "m"),
    )

    teachers.generate_with_teachers("p")

    assert (received["temperature"], received["max_tokens"]) == (0.7, 512)


def test_errors_from_openrouter_are_propagated(monkeypatch):
    import pytest

    def boom(**kwargs):
        raise RuntimeError("OpenRouter down")

    monkeypatch.setattr(teachers, "generate_response", boom)

    with pytest.raises(RuntimeError, match="OpenRouter down"):
        teachers.generate_with_teachers("p")
