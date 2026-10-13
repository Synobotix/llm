from pathlib import Path

import torch
import torch.nn.functional as F

from llm.bitnet_model.bit_transformer import BitTransformer
from llm.bitnet_model.lm_head import LMHead
from llm.tokenizer import tokenizer as tokenizer_loader


TOKENIZER_FILE = Path("tokenizer/tokenizer.json")

CHECKPOINT_FILE = Path("checkpoint/bitnet/bitnet_5docs.pt")


def load_tokenizer():
    return tokenizer_loader.load_tokenizer(TOKENIZER_FILE)


def load_checkpoint(checkpoint_path=CHECKPOINT_FILE):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

    return torch.load(checkpoint_path, map_location="cpu")


def create_bitnet_model(checkpoint, device):
    """
    Reconstruit le Transformer et la LM head depuis un checkpoint
    contenant `config` et les deux state_dict, en mode évaluation.
    """
    config = checkpoint["config"]

    transformer = BitTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        hidden_dim=config["hidden_dim"],
        num_blocks=config["num_blocks"],
        max_sequence_length=config["max_sequence_length"],
        bias=False,
    )

    lm_head = LMHead(
        d_model=config["d_model"],
        vocab_size=config["vocab_size"],
        bias=False,
    )

    transformer.load_state_dict(checkpoint["transformer_state_dict"])
    lm_head.load_state_dict(checkpoint["lm_head_state_dict"])

    transformer = transformer.to(device).eval()
    lm_head = lm_head.to(device).eval()

    return transformer, lm_head


def encode_prompt(tokenizer, prompt, device):
    """Texte -> tensor d'ids de forme [1, sequence]."""
    token_ids = tokenizer.encode(prompt).ids

    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)

    return input_ids.unsqueeze(0)


def decode_tokens(tokenizer, token_ids):
    return tokenizer.decode(token_ids)


@torch.no_grad()
def generate(
    transformer,
    lm_head,
    tokenizer,
    prompt,
    device,
    max_new_tokens=50,
    temperature=1.0,
):
    """
    Génère `max_new_tokens` tokens par échantillonnage multinomial.
    Le contexte est tronqué aux `max_sequence_length` derniers tokens.
    """
    input_ids = encode_prompt(tokenizer, prompt, device)
    max_sequence_length = transformer.max_sequence_length

    for _ in range(max_new_tokens):
        if temperature <= 0:
            raise ValueError("temperature doit être supérieur à 0.")

        model_input = input_ids[:, -max_sequence_length:]

        logits = lm_head(transformer(model_input))
        next_token_logits = logits[:, -1, :] / temperature

        probabilities = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return decode_tokens(tokenizer, input_ids[0].tolist())
