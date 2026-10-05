import pytest

from scripts.bitnet import checkpoint_path_bitnet as module


@pytest.fixture
def directory(tmp_path, monkeypatch):
    path = tmp_path / "checkpoints"
    monkeypatch.setattr(module, "CHECKPOINT_DIRECTORY", path)
    return path


def touch(directory, *names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("x")


def test_no_checkpoint_returns_none_and_first_path(directory):
    load, save = module.get_checkpoint_paths()

    assert load is None
    assert save == directory / "entrainement_bitnet_1.pt"


def test_creates_missing_directory(directory):
    module.get_checkpoint_paths()

    assert directory.is_dir()


def test_returns_last_checkpoint_and_next_one(directory):
    touch(
        directory,
        "entrainement_bitnet_1.pt",
        "entrainement_bitnet_2.pt",
    )

    load, save = module.get_checkpoint_paths()

    assert load == directory / "entrainement_bitnet_2.pt"
    assert save == directory / "entrainement_bitnet_3.pt"


def test_orders_numerically_not_lexicographically(directory):
    touch(
        directory,
        "entrainement_bitnet_9.pt",
        "entrainement_bitnet_10.pt",
    )

    load, save = module.get_checkpoint_paths()

    assert load.name == "entrainement_bitnet_10.pt"
    assert save.name == "entrainement_bitnet_11.pt"


def test_gap_in_numbering_uses_highest_number(directory):
    touch(directory, "entrainement_bitnet_1.pt",
          "entrainement_bitnet_5.pt")

    assert module.get_checkpoint_paths()[1].name == \
        "entrainement_bitnet_6.pt"


@pytest.mark.parametrize(
    "name",
    [
        "entrainement_bitnet_1.pt.bak",
        "entrainement_bitnet_.pt",
        "entrainement_bitnet_a.pt",
        "test1.pt",
        "xentrainement_bitnet_3.pt",
    ],
)
def test_ignores_files_not_matching_pattern(directory, name):
    touch(directory, name)

    load, save = module.get_checkpoint_paths()

    assert load is None
    assert save.name == "entrainement_bitnet_1.pt"


def test_ignores_directories_with_matching_name(directory):
    directory.mkdir()
    (directory / "entrainement_bitnet_4.pt").mkdir()

    assert module.get_checkpoint_paths()[0] is None
