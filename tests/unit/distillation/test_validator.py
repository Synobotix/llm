import pytest

from llm.distillation.validator import validate_teacher_response


def test_valid_response_is_accepted():
    assert validate_teacher_response("a" * 100) == (True, "Réponse valide")


@pytest.mark.parametrize("response", [None, ""])
def test_empty_response_is_rejected(response):
    assert validate_teacher_response(response) == (False, "Réponse vide")


@pytest.mark.parametrize("response", ["   ", "\n\t \n"])
def test_blank_response_is_rejected_after_cleaning(response):
    assert validate_teacher_response(response) == (
        False, "Réponse vide après nettoyage",
    )


def test_too_short_response_is_rejected_with_its_length():
    valid, reason = validate_teacher_response("a" * 49)

    assert valid is False
    assert "trop courte" in reason and "49" in reason


def test_minimum_length_is_inclusive():
    assert validate_teacher_response("a" * 50)[0] is True


def test_too_long_response_is_rejected_with_its_length():
    valid, reason = validate_teacher_response("a" * 5001)

    assert valid is False
    assert "trop longue" in reason and "5001" in reason


def test_maximum_length_is_inclusive():
    assert validate_teacher_response("a" * 5000)[0] is True


def test_length_is_measured_after_stripping_whitespace():
    assert validate_teacher_response("  " + "a" * 49 + "  ")[0] is False
    assert validate_teacher_response("  " + "a" * 50 + "  ")[0] is True


def test_custom_bounds_are_used():
    assert validate_teacher_response("abc", min_length=3, max_length=3)[0] is True
    assert validate_teacher_response("ab", min_length=3)[0] is False
    assert validate_teacher_response("abcd", max_length=3)[0] is False
