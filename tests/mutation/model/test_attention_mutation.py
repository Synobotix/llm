import math

import pytest
import torch

from llm.model.attention import MultiHeadCausalSelfAttention


def reference_attention(module, x, num_heads):
    batch, length, dim = x.shape
    head_dim = dim // num_heads
    q, k, v = module.query(x), module.key(x), module.value(x)
    merged = torch.zeros(batch, length, dim)

    for b in range(batch):
        for head in range(num_heads):
            cols = slice(head * head_dim, (head + 1) * head_dim)

            for i in range(length):
                scores = torch.stack([
                    (q[b, i, cols] * k[b, j, cols]).sum() / math.sqrt(head_dim)
                    for j in range(i + 1)
                ])
                weights = torch.softmax(scores, dim=0)
                merged[b, i, cols] = sum(
                    weights[j] * v[b, j, cols] for j in range(i + 1)
                )

    return module.output(merged)


@pytest.mark.parametrize("num_heads", [1, 2, 4])
def test_forward_matches_per_head_reference(num_heads):
    torch.manual_seed(0)
    module = MultiHeadCausalSelfAttention(
        embedding_dim=8, num_heads=num_heads, max_sequence_length=6,
    )
    x = torch.randn(2, 5, 8) * 3

    output = module(x)

    assert torch.allclose(
        output, reference_attention(module, x, num_heads), atol=1e-5,
    )


def test_indivisible_heads_message_is_exact():
    with pytest.raises(
        ValueError,
        match=r"^embedding_dim doit être divisible par num_heads\.$",
    ):
        MultiHeadCausalSelfAttention(
            embedding_dim=7, num_heads=2, max_sequence_length=4,
        )
