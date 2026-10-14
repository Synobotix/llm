import json
from pathlib import Path


def save_distilled_example(
    output_path: str | Path,
    source: str,
    prompt: str,
    response: str,
    teacher: str,
) -> None:
    """
    Enregistre un exemple distillé dans un fichier JSONL.

    Chaque appel ajoute une nouvelle ligne au fichier.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    example = {
        "source": source,
        "prompt": prompt,
        "response": response,
        "teacher": teacher,
    }

    with output_path.open("a", encoding="utf-8") as file:

        json.dump(example, file, ensure_ascii=False)

        file.write("\n")
