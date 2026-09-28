import torch

from llm.model.transformer_block import TransformerBlock


def test_forward_adds_both_sublayers_to_residual_stream():
    torch.manual_seed(0)
    block = TransformerBlock(
        embedding_dim=8, num_heads=2, ffn_hidden_dim=16, max_sequence_length=6,
    )
    x = torch.randn(2, 5, 8)

    attention_output = block.attention(block.layer_norm_1(x))
    after_attention = x + attention_output
    expected = after_attention + block.feed_forward(
        block.layer_norm_2(after_attention)
    )

    assert torch.allclose(block(x), expected, atol=1e-6)
