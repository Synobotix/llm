
def validate_teacher_response(
    response: str,
    min_length: int = 50,
    max_length: int = 5000,
) -> tuple[bool, str]:
    """
    Vérifie si une réponse générée par un teacher
    peut être conservée dans le dataset distillé.

    Retourne :
        (True, "valide")
        (False, "raison du rejet")
    """

    if not response:
        return False, "Réponse vide"

    response = response.strip()

    if not response:
        return False, "Réponse vide après nettoyage"

    length = len(response)

    if length < min_length:
        return False, f"Réponse trop courte ({length} caractères)"

    if length > max_length:
        return False, f"Réponse trop longue ({length} caractères)"

    return True, "Réponse valide"
