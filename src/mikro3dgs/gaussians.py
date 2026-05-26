from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(
        x, eps, 1.0 - eps
    )  # poprawia inicjalizację opacities i kolorów, żeby były w zakresie (0, 1) po sigmoidzie, ale nie dokładnie 0 lub 1
    return torch.log(x / (1.0 - x))


def inverse_softplus(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, min=eps)
    return x + torch.log(-torch.expm1(-x))


@dataclass
class GaussianParameters:
    means_3d: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    scales_3d: torch.Tensor
    rotations: torch.Tensor


class GaussianModel(nn.Module):
    """
    Uproszczony model zbioru Gaussów 3D.
    -mean_3d: środki gaussów 3D
    -colors: kolory RGB
    -opacities: przezroczystości (alpha) / waga
    -base_scales: podstawowe rozmiary gaussów w 3D (sigma)
    """

    def __init__(
        self,
        means_3d: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        scales_3d: torch.Tensor,
        rotations: torch.Tensor | None = None,
        learn_means: bool = False,
        learn_colors: bool = True,
        learn_opacities: bool = True,
        learn_scales: bool = True,
        learn_rotations: bool = True,
        scale_min: float = 1e-4,
        scale_max: float = 0.25,
    ) -> None:
        super().__init__()
        means_3d = means_3d.float()
        colors = colors.float().clamp(0.0, 1.0)
        opacities = opacities.float().reshape(-1, 1).clamp(1e-4, 0.95)
        scales_3d = scales_3d.float().clamp(scale_min, scale_max)

        if rotations is None:
            rotations = torch.zeros(
                (means_3d.shape[0], 4), device=means_3d.device, dtype=torch.float32
            )
            rotations[:, 0] = (
                1.0  # inicjalizacja jako brak rotacji (quaternion [1, 0, 0, 0])
            )

        rotations = rotations.float()

        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

        self.means_3d = nn.Parameter(means_3d, requires_grad=learn_means)
        self.colors_raw = nn.Parameter(
            inverse_sigmoid(colors), requires_grad=learn_colors
        )
        self.opacities_raw = nn.Parameter(
            inverse_sigmoid(opacities), requires_grad=learn_opacities
        )
        self.scales_raw = nn.Parameter(
            inverse_softplus(scales_3d), requires_grad=learn_scales
        )
        self.rotations_raw = nn.Parameter(rotations, requires_grad=learn_rotations)

    def get_parameters(self) -> GaussianParameters:
        """Zwraca parametry gaussów w formie gotowej do renderowania"""

        colors = torch.sigmoid(self.colors_raw)
        opacities = torch.sigmoid(self.opacities_raw).squeeze(-1)
        scales_3d = F.softplus(self.scales_raw).clamp(self.scale_min, self.scale_max)
        rotations = self.rotations_raw / (
            torch.norm(self.rotations_raw, dim=-1, keepdim=True) + 1e-8
        )  # normalizacja quaternionów
        return GaussianParameters(
            self.means_3d, colors, opacities, scales_3d, rotations
        )

    @torch.no_grad()
    def reset_opacity(self, value: float = 0.02) -> None:
        self.opacities_raw.data[:] = inverse_sigmoid(
            torch.full_like(self.opacities_raw.data, value)
        )

    def replace_parameters(self, means_3d, colors, opacities, scales_3d, rotations):
        device = self.means_3d.device
        self.means_3d = nn.Parameter(
            means_3d.detach().to(device).float(), requires_grad=True
        )
        self.colors_raw = nn.Parameter(
            inverse_sigmoid(colors.detach().to(device).float().clamp(0.0, 1.0)),
            requires_grad=True,
        )
        self.opacities_raw = nn.Parameter(
            inverse_sigmoid(
                opacities.detach().to(device).float().reshape(-1, 1).clamp(1e-4, 0.95)
            ),
            requires_grad=True,
        )
        self.scales_raw = nn.Parameter(
            inverse_softplus(
                scales_3d.detach()
                .to(device)
                .float()
                .clamp(self.scale_min, self.scale_max)
            ),
            requires_grad=True,
        )
        r = rotations.detach().to(device).float()
        self.rotations_raw = nn.Parameter(
            r / (r.norm(dim=-1, keepdim=True) + 1e-8), requires_grad=True
        )