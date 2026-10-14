import pytest

from llm.distillation.prompts import build_distillation_prompt


def test_prompt_contains_the_source_text():
    prompt = build_distillation_prompt("Le chat dort sur le canapé.")

    assert "Le chat dort sur le canapé." in prompt


def test_source_text_is_placed_between_its_label_and_the_answer_marker():
    prompt = build_distillation_prompt("TEXTE-UNIQUE")

    assert prompt.index("Texte source :") < prompt.index("TEXTE-UNIQUE") \
        < prompt.index("Réponse :")


def test_prompt_ends_with_the_answer_marker():
    assert build_distillation_prompt("abc").endswith("Réponse :")


def test_prompt_forbids_inventing_information():
    assert "N'invente aucune information" in build_distillation_prompt("abc")


def test_prompt_is_stripped():
    prompt = build_distillation_prompt("abc")

    assert prompt == prompt.strip()


def test_text_is_not_modified_nor_stripped():
    text = "  ligne 1\n\nligne 2  "

    assert text in build_distillation_prompt(text)


def test_same_text_gives_same_prompt():
    assert build_distillation_prompt("abc") == build_distillation_prompt("abc")


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_blank_text_raises(text):
    with pytest.raises(ValueError, match="vide"):
        build_distillation_prompt(text)


@pytest.mark.parametrize("text", [None, 0])
def test_missing_text_raises(text):
    with pytest.raises(ValueError):
        build_distillation_prompt(text)
