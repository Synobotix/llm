import torch


class TextGenerator:
    """
    Génération gloutonne (greedy) avec un Transformer causal,
    avec un journal détaillé de chaque étape.
    """

    CONTEXT_SIZE = 128
    TOP_K_DEBUG = 10

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        encoded = self.tokenizer.encode(prompt)

        input_ids = torch.tensor(
            [encoded.ids], dtype=torch.long, device=self.device
        )

        self._print_start(prompt, encoded.ids)

        for step in range(max_new_tokens):
            next_token_logits = self._next_token_logits(input_ids, step)

            next_token_id = torch.argmax(
                next_token_logits, dim=-1, keepdim=True
            )

            self._print_chosen_token(next_token_id.item(), next_token_logits)

            input_ids = torch.cat([input_ids, next_token_id], dim=1)

        generated_ids = input_ids[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)

        self._print_end(generated_ids, generated_text)

        return generated_text

    def _next_token_logits(self, input_ids: torch.Tensor, step: int):
        """Logits du dernier token, sur le contexte maximal autorisé."""
        input_context = input_ids[:, -self.CONTEXT_SIZE:]

        print("\n" + "-" * 60)
        print(f"ÉTAPE {step + 1}")
        print("-" * 60)
        print("Nombre de tokens dans input_ids :", input_ids.shape[1])
        print(
            "Nombre de tokens dans input_context :", input_context.shape[1]
        )

        logits = self.model(input_context)
        print("Shape logits :", logits.shape)

        next_token_logits = logits[:, -1, :]
        print("Shape derniers logits :", next_token_logits.shape)
        print(
            "Nombre de tokens du vocabulaire :", next_token_logits.shape[-1]
        )

        if step == 0:
            self._print_logits_debug(next_token_logits)

        return next_token_logits

    def _decode_token(self, token_id: int) -> str:
        try:
            return self.tokenizer.decode([token_id])
        except Exception:
            return "<erreur>"

    # ------------------------------------------------------------
    # Journal
    # ------------------------------------------------------------

    def _print_start(self, prompt: str, token_ids: list[int]) -> None:
        print("\n" + "=" * 60)
        print("DÉBUT DE LA GÉNÉRATION")
        print("=" * 60)
        print("Prompt :", repr(prompt))
        print("Token IDs :", token_ids)
        print("Nombre de tokens du prompt :", len(token_ids))

    def _print_logits_debug(self, next_token_logits: torch.Tensor) -> None:
        print("\n" + "=" * 60)
        print("DEBUG DES LOGITS - PREMIÈRE ÉTAPE")
        print("=" * 60)
        print("Min   :", next_token_logits.min().item())
        print("Max   :", next_token_logits.max().item())
        print("Mean  :", next_token_logits.mean().item())
        print("Std   :", next_token_logits.std().item())
        print("NaN   :", torch.isnan(next_token_logits).any().item())
        print("Inf   :", torch.isinf(next_token_logits).any().item())

        top_values, top_ids = torch.topk(
            next_token_logits, k=self.TOP_K_DEBUG, dim=-1
        )

        print("\nTop 10 tokens :")

        for rank, (value, token_id) in enumerate(
            zip(top_values[0], top_ids[0]), start=1
        ):
            token_id = token_id.item()
            print(
                f"{rank:2d}. "
                f"ID={token_id:3d} | "
                f"logit={value.item():10.6f} | "
                f"token={self._decode_token(token_id)!r}"
            )

    def _print_chosen_token(
        self, token_id: int, next_token_logits: torch.Tensor
    ) -> None:
        print("\nTOKEN CHOISI")
        print("Token ID :", token_id)
        print("Token texte :", repr(self._decode_token(token_id)))
        print("Logit du token choisi :", next_token_logits[0, token_id].item())

    def _print_end(self, generated_ids: list[int], text: str) -> None:
        print("\n" + "=" * 60)
        print("FIN DE LA GÉNÉRATION")
        print("=" * 60)
        print("Nombre total de tokens :", len(generated_ids))
        print("IDs générés :", generated_ids)
        print("Texte généré :", text)
