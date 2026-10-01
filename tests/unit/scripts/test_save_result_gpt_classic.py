import pytest
import torch

from scripts.gpt_classic import save_result_gpt_classic as module


@pytest.fixture
def results(tmp_path, monkeypatch):
    directory = tmp_path / "results"
    monkeypatch.setattr(module, "RESULT_DIRECTORY", directory)
    return directory


def make_checkpoint(path, **content):
    torch.save(content, path)
    return path


def report(**overrides):
    arguments = dict(
        checkpoint_path="ckpt.pt",
        result_path="result.md",
        epoch=3,
        global_step=42,
        training_results={},
        configuration={},
        trainable_parameters=None,
        device="cpu",
        gpu_name=None,
    )
    arguments.update(overrides)
    return module.build_markdown_report(**arguments)


# ------------------------------------------------------------
# numérotation
# ------------------------------------------------------------

def test_next_number_is_one_when_directory_missing(tmp_path):
    assert module.get_next_result_number(tmp_path / "absent") == 1


def test_next_number_is_one_when_directory_empty(tmp_path):
    assert module.get_next_result_number(tmp_path) == 1


def test_next_number_follows_highest_existing(tmp_path):
    for number in (1, 2, 10):
        (tmp_path / f"resultat_entrainement_gpt_classic{number}.md").write_text("")

    assert module.get_next_result_number(tmp_path) == 11


def test_next_number_ignores_foreign_files_and_directories(tmp_path):
    (tmp_path / "resultat_entrainement_gpt_classic3.txt").write_text("")
    (tmp_path / "notes.md").write_text("")
    (tmp_path / "resultat_entrainement_gpt_classic7.md").mkdir()

    assert module.get_next_result_number(tmp_path) == 1


def test_next_path_uses_prefix_number_and_extension(tmp_path):
    (tmp_path / "resultat_entrainement_gpt_classic4.md").write_text("")

    assert module.get_next_result_path(tmp_path) == \
        tmp_path / "resultat_entrainement_gpt_classic5.md"


# ------------------------------------------------------------
# build_markdown_report
# ------------------------------------------------------------

def test_report_contains_general_information():
    text = report(device="cuda", gpu_name="RTX")

    assert "# Résultats — GPT Classic V1" in text
    assert "| Device | `cuda` |" in text
    assert "| GPU | `RTX` |" in text
    assert "| Époque finale | `3` |" in text
    assert "| Global step | `42` |" in text


def test_report_omits_gpu_row_without_gpu():
    assert "| GPU |" not in report(gpu_name=None)


def test_report_shows_training_results_and_configuration():
    text = report(
        training_results={"train_loss": 1.5, "duration": "00h 01min 00.00s"},
        configuration={"vocab_size": 123, "d_model": 64},
    )

    assert "| Train loss | `1.5` |" in text
    assert "| Durée totale | `00h 01min 00.00s` |" in text
    assert "| Vocabulaire | `123` |" in text
    assert "| D model | `64` |" in text


def test_report_falls_back_to_na_for_missing_values():
    text = report()

    assert "| Train loss | `N/A` |" in text
    assert "| Validation PPL | `N/A` |" in text
    assert "| Vocabulaire | `N/A` |" in text


# ------------------------------------------------------------
# save_training_result
# ------------------------------------------------------------

def test_save_writes_numbered_markdown_from_checkpoint(tmp_path, results):
    checkpoint = make_checkpoint(
        tmp_path / "ckpt.pt",
        epoch=2,
        global_step=9,
        training_results={"train_loss": 0.25},
        configuration={"vocab_size": 50},
    )

    path = module.save_training_result(checkpoint_path=checkpoint, device="cpu")

    assert path == results / "resultat_entrainement_gpt_classic1.md"
    text = path.read_text(encoding="utf-8")
    assert "| Train loss | `0.25` |" in text
    assert "| Époque finale | `2` |" in text
    assert "| Vocabulaire | `50` |" in text


def test_save_never_overwrites_previous_reports(tmp_path, results):
    checkpoint = make_checkpoint(tmp_path / "ckpt.pt")

    first = module.save_training_result(checkpoint_path=checkpoint)
    second = module.save_training_result(checkpoint_path=checkpoint)

    assert first != second
    assert second.name == "resultat_entrainement_gpt_classic2.md"
    assert first.exists()


def test_save_defaults_missing_checkpoint_fields_to_na(tmp_path, results):
    checkpoint = make_checkpoint(tmp_path / "ckpt.pt")

    text = module.save_training_result(checkpoint_path=checkpoint).read_text(
        encoding="utf-8"
    )

    assert "| Époque finale | `N/A` |" in text
    assert "| Device | `N/A` |" in text


def test_save_missing_checkpoint_raises(tmp_path, results):
    with pytest.raises(FileNotFoundError):
        module.save_training_result(checkpoint_path=tmp_path / "absent.pt")

    assert not results.exists()


def test_save_without_path_uses_next_checkpoint_from_helper(
    tmp_path, results, monkeypatch
):
    checkpoint = make_checkpoint(tmp_path / "next.pt")
    monkeypatch.setattr(
        module, "get_checkpoint_paths", lambda: (None, checkpoint)
    )

    path = module.save_training_result()

    assert path.exists()
