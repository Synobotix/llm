import pytest
import torch

from llm.bitnet_model.bit_transformer import BitTransformer

XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "BitLinear.quantize_activations utilise un absmax unique pour tout "
        "le tenseur : changer un token futur modifie l'échelle de "
        "quantification des tokens précédents (fuite causale ~1e-2)."
    ),
)

VOCAB = 20


def make(blocks=2, max_len=8):
    return BitTransformer(VOCAB, 8, 2, 16, blocks, max_len).eval()


def test_returns_hidden_states_not_logits():
    out = make()(torch.randint(0, VOCAB, (3, 5)))

    assert out.shape == (3, 5, 8)


def test_stores_hyperparameters():
    model = make()

    assert (
        model.vocab_size, model.d_model, model.num_heads, model.hidden_dim,
        model.num_blocks, model.max_sequence_length,
    ) == (VOCAB, 8, 2, 16, 2, 8)
    assert len(model.blocks) == 2


def test_zero_blocks_still_works():
    out = make(blocks=0)(torch.randint(0, VOCAB, (1, 4)))

    assert out.shape == (1, 4, 8)


def test_sequence_longer_than_max_raises():
    with pytest.raises(ValueError, match="max_sequence_length"):
        make(max_len=4)(torch.randint(0, VOCAB, (1, 5)))


def test_sequence_equal_to_max_is_accepted():
    assert make(max_len=4)(torch.randint(0, VOCAB, (1, 4))).shape == (1, 4, 8)


def test_token_out_of_vocab_raises():
    with pytest.raises(IndexError):
        make()(torch.tensor([[VOCAB]]))


@XFAIL
def test_hidden_states_are_causal():
    model = make()
    ids = torch.randint(0, VOCAB, (1, 6))
    changed = ids.clone()
    changed[0, 5] = (ids[0, 5] + 1) % VOCAB

    assert torch.allclose(model(ids)[:, :5], model(changed)[:, :5], atol=1e-5)


def test_all_parameters_receive_gradients():
    model = make().train()

    model(torch.randint(0, VOCAB, (2, 4))).sum().backward()

    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == []
