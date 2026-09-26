import llm.data.dataloader as dataloader


def test_dataloader_uses_configured_num_workers(tmp_path, monkeypatch):
    path = tmp_path / "data.jsonl"
    path.write_text('{"input_ids": [1], "labels": [2]}\n', encoding="utf-8")
    monkeypatch.setattr(dataloader, "NUM_WORKERS", 3)

    loader = dataloader._create_dataloader(path, shuffle=False)

    assert loader.num_workers == 3


def test_validation_dataloader_is_not_shuffled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dataloader,
        "_create_dataloader",
        lambda file_path, shuffle: calls.append((file_path, shuffle)),
    )

    dataloader.create_validation_dataloader()

    assert calls == [(dataloader.VALIDATION_FILE, False)]
    assert calls[0][1] is False
