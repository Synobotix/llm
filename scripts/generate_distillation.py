
import json
from pathlib import Path

from llm.distillation.prompts import build_distillation_prompt
from llm.distillation.teachers import generate_with_teachers
from llm.distillation.validator import validate_teacher_response
from llm.distillation.storage import save_distilled_example


INPUT_PATH = Path("data/processed/c4_clean.jsonl")
OUTPUT_PATH = Path("data/distilled/distilled.jsonl")

# Pour le premier test uniquement.
MAX_DOCUMENTS = 3


def load_c4_documents(input_path: Path):
    """
    Charge les documents du dataset C4 nettoyé.
    """
    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            document = json.loads(line)
            text = document.get("text")

            if text:
                yield text


def process_document(text: str, index: int) -> bool:
    """
    Traite un document C4 :

    1. Construction du prompt
    2. Appel OpenRouter avec les teachers
    3. Validation de la réponse
    4. Sauvegarde dans le dataset distillé
    """

    print(f"\nDocument {index + 1}")
    print("-" * 50)

    # 1. Construire le prompt
    prompt = build_distillation_prompt(text)

    print("✓ Prompt généré")

    # 2. Appeler OpenRouter
    response, model_used = generate_with_teachers(
        prompt=prompt,
        temperature=0.7,
        max_tokens=8200,
    )

    print(f"✓ Réponse générée")
    print(f"✓ Teacher utilisé : {model_used}")

    # 3. Valider la réponse
    is_valid, reason = validate_teacher_response(response)

    if not is_valid:
        print(f"✗ Réponse rejetée : {reason}")
        return False

    # 4. Sauvegarder
    save_distilled_example(
        output_path=OUTPUT_PATH,
        source=text,
        prompt=prompt,
        response=response,
        teacher=model_used,
    )

    print("✓ Exemple enregistré")

    return True


def main():
    """
    Lance la distillation sur un nombre limité de documents.
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {INPUT_PATH}")

    total = 0
    valid = 0
    rejected = 0

    print("=" * 50)
    print("TEST DE DISTILLATION")
    print("=" * 50)

    for index, text in enumerate(load_c4_documents(INPUT_PATH)):

        if total >= MAX_DOCUMENTS:
            break

        total += 1

        try:
            success = process_document(text=text, index=index)

            if success:
                valid += 1
            else:
                rejected += 1

        except Exception as error:
            rejected += 1

            print(f"✗ Erreur : {error}")

    print("\n" + "=" * 50)
    print("DISTILLATION TERMINÉE")
    print("=" * 50)

    print(f"Documents traités : {total}")
    print(f"Réponses validées : {valid}")
    print(f"Réponses rejetées : {rejected}")
    print(f"Dataset : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
