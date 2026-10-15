import os

import requests
from dotenv import load_dotenv


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_FALLBACK_MODELS = 3
REQUEST_TIMEOUT_SECONDS = 120


def generate_response(
    models: list[str],
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> tuple[str, str]:
    """
    Génère une réponse avec OpenRouter.

    Args:
        models: Modèles utilisés pour le fallback (3 maximum, limite
            d'OpenRouter).
        prompt: Prompt envoyé au teacher.
        temperature: Température de génération.
        max_tokens: Nombre maximum de tokens générés.

    Returns:
        (réponse, modèle qui a effectivement généré la réponse)

    Raises:
        ValueError: Clé API absente ou liste de modèles invalide.
        RuntimeError: Erreur HTTP, aucun choix, génération tronquée
            ou réponse vide.
    """
    _validate_request(models)

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "models": models,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if not response.ok:
        raise RuntimeError(_format_http_error(response))

    return _extract_content(response.json())


def _validate_request(models: list[str]) -> None:
    if not OPENROUTER_API_KEY:
        raise ValueError("La variable OPENROUTER_API_KEY n'est pas définie.")

    if not models:
        raise ValueError("La liste des modèles est vide.")

    if len(models) > MAX_FALLBACK_MODELS:
        raise ValueError(
            f"OpenRouter accepte au maximum {MAX_FALLBACK_MODELS} modèles "
            "dans une liste de fallback."
        )


def _format_http_error(response) -> str:
    try:
        details = response.json()
    except ValueError:
        details = response.text

    return f"Erreur OpenRouter {response.status_code}: {details}"


def _extract_content(data: dict) -> tuple[str, str]:
    choices = data.get("choices", [])

    if not choices:
        raise RuntimeError(
            "OpenRouter n'a retourné aucun choix dans la réponse."
        )

    choice = choices[0]

    if choice.get("finish_reason") == "length":
        raise RuntimeError(
            "La génération a été interrompue car "
            "la limite max_tokens a été atteinte."
        )

    content = choice.get("message", {}).get("content") or ""

    if not content.strip():
        raise RuntimeError("Le teacher n'a retourné aucune réponse finale.")

    return content, data.get("model", "unknown")
