def build_distillation_prompt(text: str) -> str:
    """
    Construit le prompt envoyé au teacher à partir
    d'un texte provenant du dataset C4.
    """

    if not text or not text.strip():
        raise ValueError("Le texte fourni au prompt est vide.")

    prompt = f"""
Tu es un modèle enseignant chargé de produire une donnée
d'entraînement de haute qualité pour un petit modèle de langage.

À partir du texte fourni ci-dessous :

1. Comprends précisément son contenu.
2. Identifie les informations importantes.
3. Produis une réponse claire, cohérente et autonome.
4. N'invente aucune information absente du texte.
5. Ne mentionne pas que tu es un modèle enseignant.
6. Retourne uniquement la réponse finale.

Texte source :
{text}

Réponse :
""".strip()

    return prompt
