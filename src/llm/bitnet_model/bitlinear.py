import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Module):
    """
    BitLinear inspiré de BitNet b1.58.

    - poids maîtres FP32 conservés pour l'entraînement ;
    - poids ternarisés {-1, 0, +1} x scale (RoundClip) ;
    - activations quantifiées en 8 bits (absmax) ;
    - STE (Straight-Through Estimator) pour les poids et les activations ;
    - inférence : poids ternaires INT8 préparés une seule fois.

    Version PyTorch de référence : F.linear() n'utilise pas de kernel
    matériel spécialisé INT8 x ternaire.
    """

    ACTIVATION_BITS = 8
    MIN_SCALE = 1e-5

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float32)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.float32)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        # Cache d'inférence : non sauvegardé dans le state_dict.
        self.register_buffer(
            "weight_ternary",
            torch.empty(0, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.tensor(1.0, dtype=torch.float32),
            persistent=False,
        )

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # ------------------------------------------------------------
    # Quantification
    # ------------------------------------------------------------

    @classmethod
    def ternarize_weights(cls, weight: torch.Tensor):
        """
        Retourne (poids dans {-1, 0, +1}, scale) avec
        scale = mean(abs(weight)).
        """
        scale = weight.abs().mean().clamp(min=cls.MIN_SCALE)
        ternary = torch.clamp(torch.round(weight / scale), -1, 1)

        return ternary, scale

    @classmethod
    def quantize_weights(cls, weight: torch.Tensor):
        """
        Retourne ({-1, 0, +1} x scale, scale).
        """
        ternary, scale = cls.ternarize_weights(weight)

        return ternary * scale, scale

    @classmethod
    def quantize_activations(
        cls,
        x: torch.Tensor,
        bits: int = ACTIVATION_BITS,
    ):
        """
        Quantification absmax par tenseur, puis déquantification en FP32
        (nécessaire pour F.linear()). Pour 8 bits : niveaux [-127, +127].
        """
        qmax = 2 ** (bits - 1)
        scale = x.abs().max().clamp(min=cls.MIN_SCALE)

        x_scaled = x * qmax / scale
        x_clipped = torch.clamp(x_scaled, -qmax + 1, qmax - 1)

        return torch.round(x_clipped) / qmax * scale

    @staticmethod
    def ste(original: torch.Tensor, quantized: torch.Tensor):
        """
        Straight-Through Estimator : la valeur du forward est `quantized`,
        le gradient du backward est celui de `original`.
        """
        return original + (quantized - original).detach()

    # ------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------

    @torch.no_grad()
    def prepare_for_inference(self):
        """
        Stocke les poids sous forme INT8 {-1, 0, +1} avec un scale séparé.
        """
        ternary, scale = self.ternarize_weights(self.weight)

        self.weight_ternary = ternary.to(torch.int8)
        self.weight_scale = scale.detach()

    def _training_weight(self) -> torch.Tensor:
        weight_quantized, _ = self.quantize_weights(self.weight)

        return self.ste(self.weight, weight_quantized)

    def _inference_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if self.weight_ternary.numel() == 0:
            self.prepare_for_inference()

        return self.weight_ternary.to(dtype) * self.weight_scale.to(dtype)

    def forward(self, x: torch.Tensor):
        if self.training:
            weight = self._training_weight()
        else:
            weight = self._inference_weight(x.dtype)

        x_used = self.ste(x, self.quantize_activations(x))

        return F.linear(x_used, weight, self.bias)
