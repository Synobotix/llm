import pytest
import torch

from llm.model.transformer import Transformer


def make(vocab=20, dim=8, heads=2, hidden=16, layers=2, max_len=8):
    return Transformer(vocab, dim, heads, hidden, layers, max_len).eval()


def test_logits_shape():
    out = make()(torch.randint(0, 20, (3, 5)))
    assert out.shape == (3, 5, 20)


def test_stores_hyperparameters():
    model = make()
    assert (model.vocab_size, model.embedding_dim, model.num_heads,
            model.ffn_hidden_dim, model.num_layers,
            model.max_sequence_length) == (20, 8, 2, 16, 2, 8)
    assert len(model.transformer_blocks) == 2


def test_zero_layers_still_works():
    model = make(layers=0)
    assert model(torch.randint(0, 20, (1, 4))).shape == (1, 4, 20)


def test_sequence_longer_than_max_raises():
    with pytest.raises(RuntimeError):
        make(max_len=4)(torch.randint(0, 20, (1, 5)))


def test_sequence_equal_to_max_ok():
    assert make(max_len=4)(torch.randint(0, 20, (1, 4))).shape == (1, 4, 20)


def test_token_out_of_vocab_raises():
    with pytest.raises(IndexError):
        make()(torch.tensor([[20]]))


def test_causality_of_logits():
    model = make()
    ids = torch.randint(0, 20, (1, 6))
    ids2 = ids.clone()
    ids2[0, 5] = (ids[0, 5] + 1) % 20
    assert torch.allclose(model(ids)[:, :5], model(ids2)[:, :5], atol=1e-5)


def test_deterministic_in_eval():
    model = make()
    ids = torch.randint(0, 20, (2, 4))
    assert torch.equal(model(ids), model(ids))


def test_all_parameters_get_gradients():
    model = make().train()
    ids = torch.randint(0, 20, (2, 4))
    model(ids).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == []


def test_state_dict_roundtrip():
    a, b = make(), make()
    b.load_state_dict(a.state_dict())
    ids = torch.randint(0, 20, (1, 4))
    assert torch.equal(a(ids), b(ids))


def test_mismatched_state_dict_raises():
    with pytest.raises(RuntimeError):
        make().load_state_dict(make(vocab=21).state_dict())
