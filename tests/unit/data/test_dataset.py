import json

import pytest
import torch

from llm.data.dataset import CausalLanguageModelingDataset


def sample(ids, labels=None):
    return {"input_ids": ids, "labels": ids if labels is None else labels}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset introuvable"):
        CausalLanguageModelingDataset(tmp_path / "nope.jsonl")


def test_accepts_str_path(write_jsonl):
    path = write_jsonl([sample([1, 2])])
    assert len(CausalLanguageModelingDataset(str(path))) == 1


def test_empty_file(write_jsonl):
    dataset = CausalLanguageModelingDataset(write_jsonl([]))
    assert len(dataset) == 0
    with pytest.raises(IndexError):
        dataset[0]


def test_loads_samples_as_long_tensors(write_jsonl):
    dataset = CausalLanguageModelingDataset(
        write_jsonl([sample([1, 2, 3], [2, 3, 4]), sample([5, 6])]))
    assert len(dataset) == 2
    item = dataset[0]
    assert item["input_ids"].dtype == torch.long
    assert item["labels"].dtype == torch.long
    assert item["input_ids"].tolist() == [1, 2, 3]
    assert item["labels"].tolist() == [2, 3, 4]


def test_negative_index(write_jsonl):
    dataset = CausalLanguageModelingDataset(
        write_jsonl([sample([1]), sample([2])]))
    assert dataset[-1]["input_ids"].tolist() == [2]


def test_index_out_of_range(write_jsonl):
    dataset = CausalLanguageModelingDataset(write_jsonl([sample([1])]))
    with pytest.raises(IndexError):
        dataset[1]


def test_skips_blank_lines(write_jsonl):
    dataset = CausalLanguageModelingDataset(
        write_jsonl([json.dumps(sample([1])), "", "   ", "\t"]))
    assert len(dataset) == 1


def test_invalid_json_line_skipped_with_warning(write_jsonl, capsys):
    path = write_jsonl([json.dumps(sample([1])), "{not json",
                        json.dumps(sample([2]))])
    dataset = CausalLanguageModelingDataset(path)
    assert len(dataset) == 2
    assert "Ligne 2 ignorée" in capsys.readouterr().out


@pytest.mark.parametrize(
    "bad",
    [
        {"labels": [1]},
        {"input_ids": [1]},
        {"input_ids": "abc", "labels": [1]},
        {"input_ids": [1], "labels": "abc"},
        {"input_ids": None, "labels": None},
        {},
    ],
)
def test_missing_or_non_list_fields_skipped(write_jsonl, bad):
    dataset = CausalLanguageModelingDataset(
        write_jsonl([bad, sample([1, 2])]))
    assert len(dataset) == 1


def test_non_object_json_line_crashes(write_jsonl):
    # Une ligne JSON valide mais non-dict (ex. liste) n'est pas gérée.
    with pytest.raises(AttributeError):
        CausalLanguageModelingDataset(write_jsonl(["[1, 2, 3]"]))


def test_length_mismatch_raises_with_line_number(write_jsonl):
    path = write_jsonl([sample([1]), {"input_ids": [1, 2], "labels": [1]}])
    with pytest.raises(ValueError, match="ligne 2"):
        CausalLanguageModelingDataset(path)


def test_empty_sequences_allowed(write_jsonl):
    dataset = CausalLanguageModelingDataset(write_jsonl([sample([])]))
    assert len(dataset) == 1
    assert dataset[0]["input_ids"].numel() == 0


def test_unicode_content_file(write_jsonl):
    path = write_jsonl([{"input_ids": [1], "labels": [1], "text": "é€😀"}])
    assert len(CausalLanguageModelingDataset(path)) == 1


def test_extra_keys_ignored(write_jsonl):
    dataset = CausalLanguageModelingDataset(
        write_jsonl([{"input_ids": [1], "labels": [1], "extra": 1}]))
    assert set(dataset[0]) == {"input_ids", "labels"}


def test_float_ids_are_cast_to_long(write_jsonl):
    dataset = CausalLanguageModelingDataset(
        write_jsonl([{"input_ids": [1.9], "labels": [2.2]}]))
    assert dataset[0]["input_ids"].tolist() == [1]


def test_crlf_line_endings(tmp_path):
    path = tmp_path / "crlf.jsonl"
    path.write_bytes(b'{"input_ids":[1],"labels":[1]}\r\n'
                     b'{"input_ids":[2],"labels":[2]}\r\n')
    assert len(CausalLanguageModelingDataset(path)) == 2
