import pytest
import torch

from llm.model.embedding import TokenEmbedding


def test_output_shape():
    out = TokenEmbedding(10, 8)(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert out.shape == (2, 3, 8)


def test_same_id_same_vector():
    out = TokenEmbedding(10, 8)(torch.tensor([[3, 3]]))
    assert torch.equal(out[0, 0], out[0, 1])


def test_id_out_of_range_raises():
    with pytest.raises(IndexError):
        TokenEmbedding(10, 8)(torch.tensor([[10]]))


def test_negative_id_raises():
    with pytest.raises(IndexError):
        TokenEmbedding(10, 8)(torch.tensor([[-1]]))


def test_float_ids_raise():
    with pytest.raises(RuntimeError):
        TokenEmbedding(10, 8)(torch.tensor([[1.0]]))


def test_empty_sequence():
    assert TokenEmbedding(10, 8)(torch.zeros(2, 0, dtype=torch.long)).shape \
        == (2, 0, 8)


def test_gradient_only_on_used_rows():
    emb = TokenEmbedding(5, 4)
    emb(torch.tensor([[1, 1]])).sum().backward()
    grad = emb.embedding.weight.grad
    assert grad[1].abs().sum() > 0
    assert grad[[0, 2, 3, 4]].abs().sum() == 0
