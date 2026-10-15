import types

import pytest

from llm.distillation import openrouter


class FakeResponse:
    def __init__(self, data=None, ok=True, status_code=200, text="", json_error=False):
        self._data = data
        self.ok = ok
        self.status_code = status_code
        self.text = text
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("not json")
        return self._data


def completion(content="Réponse du teacher", model="vendor/model-a",
               finish_reason="stop"):
    return {
        "model": model,
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content},
            }
        ],
    }


@pytest.fixture
def api(monkeypatch):
    """Clé API définie et requests.post remplacé par un enregistreur."""
    monkeypatch.setattr(openrouter, "OPENROUTER_API_KEY", "test-key")
    state = types.SimpleNamespace(calls=[], response=FakeResponse(completion()))

    def post(url, headers, json, timeout):
        state.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return state.response

    monkeypatch.setattr(openrouter.requests, "post", post)
    return state


# ------------------------------------------------------------
# requête
# ------------------------------------------------------------

def test_request_targets_openrouter_with_bearer_key_and_timeout(api):
    openrouter.generate_response(["vendor/model-a"], "bonjour")

    call = api.calls[0]
    assert call["url"] == openrouter.OPENROUTER_URL
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["timeout"] == openrouter.REQUEST_TIMEOUT_SECONDS


def test_request_body_contains_models_prompt_and_sampling_parameters(api):
    openrouter.generate_response(
        ["a/x", "b/y"], "bonjour", temperature=0.2, max_tokens=64
    )

    assert api.calls[0]["json"] == {
        "models": ["a/x", "b/y"],
        "messages": [{"role": "user", "content": "bonjour"}],
        "temperature": 0.2,
        "max_tokens": 64,
    }


def test_default_sampling_parameters(api):
    openrouter.generate_response(["a/x"], "p")

    body = api.calls[0]["json"]
    assert (body["temperature"], body["max_tokens"]) == (0.7, 512)


def test_returns_content_and_model_that_answered(api):
    api.response = FakeResponse(completion("Bonjour !", "fallback/model-b"))

    assert openrouter.generate_response(["a/x", "fallback/model-b"], "p") == (
        "Bonjour !", "fallback/model-b",
    )


def test_model_defaults_to_unknown_when_not_reported(api):
    data = completion()
    del data["model"]
    api.response = FakeResponse(data)

    assert openrouter.generate_response(["a/x"], "p")[1] == "unknown"


# ------------------------------------------------------------
# validation de la requête (aucun appel réseau)
# ------------------------------------------------------------

def test_missing_api_key_raises_before_any_request(api, monkeypatch):
    monkeypatch.setattr(openrouter, "OPENROUTER_API_KEY", None)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        openrouter.generate_response(["a/x"], "p")

    assert api.calls == []


def test_empty_api_key_is_rejected(api, monkeypatch):
    monkeypatch.setattr(openrouter, "OPENROUTER_API_KEY", "")

    with pytest.raises(ValueError):
        openrouter.generate_response(["a/x"], "p")


def test_empty_model_list_raises(api):
    with pytest.raises(ValueError, match="vide"):
        openrouter.generate_response([], "p")

    assert api.calls == []


def test_more_than_three_models_raises(api):
    with pytest.raises(ValueError, match="3"):
        openrouter.generate_response(["a", "b", "c", "d"], "p")

    assert api.calls == []


def test_exactly_three_models_are_accepted(api):
    openrouter.generate_response(["a", "b", "c"], "p")

    assert len(api.calls) == 1


# ------------------------------------------------------------
# erreurs de réponse
# ------------------------------------------------------------

def test_http_error_reports_status_and_json_details(api):
    api.response = FakeResponse(
        {"error": "rate limited"}, ok=False, status_code=429
    )

    with pytest.raises(RuntimeError, match="429.*rate limited"):
        openrouter.generate_response(["a/x"], "p")


def test_http_error_falls_back_to_raw_text_when_body_is_not_json(api):
    api.response = FakeResponse(
        ok=False, status_code=502, text="Bad Gateway", json_error=True
    )

    with pytest.raises(RuntimeError, match="502.*Bad Gateway"):
        openrouter.generate_response(["a/x"], "p")


def test_response_without_choice_raises(api):
    api.response = FakeResponse({"choices": []})

    with pytest.raises(RuntimeError, match="aucun choix"):
        openrouter.generate_response(["a/x"], "p")


def test_response_without_choices_key_raises(api):
    api.response = FakeResponse({})

    with pytest.raises(RuntimeError, match="aucun choix"):
        openrouter.generate_response(["a/x"], "p")


def test_truncated_generation_raises(api):
    api.response = FakeResponse(completion(finish_reason="length"))

    with pytest.raises(RuntimeError, match="max_tokens"):
        openrouter.generate_response(["a/x"], "p")


@pytest.mark.parametrize("content", ["", "   \n", None])
def test_empty_final_answer_raises(api, content):
    api.response = FakeResponse(completion(content=content))

    with pytest.raises(RuntimeError, match="aucune réponse finale"):
        openrouter.generate_response(["a/x"], "p")


def test_message_without_content_key_raises(api):
    api.response = FakeResponse({"choices": [{"message": {}}]})

    with pytest.raises(RuntimeError, match="aucune réponse finale"):
        openrouter.generate_response(["a/x"], "p")


def test_only_the_first_choice_is_used(api):
    data = completion("premier")
    data["choices"].append({"message": {"content": "second"}})
    api.response = FakeResponse(data)

    assert openrouter.generate_response(["a/x"], "p")[0] == "premier"
