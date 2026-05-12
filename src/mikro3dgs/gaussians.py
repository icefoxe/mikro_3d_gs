from dataclasses import dataclass

import torch
import torch.nn as nn


def inverse_sigmoid(x: torch.Tensor, eps:float=1e-6) -> torch.Tensor:
        # poprawia inicjalizację opacities i kolorów, żeby były w zakresie (0, 1) po sigmoidzie, ale nie dokładnie 0 lub 1
        x = torch.clamp(x, eps, 1.0 - eps) #zapobiegamy wartościom dokładnie 0 lub 1, bo wtedy inverse sigmoid daje inf
        return torch.log(x / (1 - x))



def inverse_softplus(x: torch.Tensor, eps:float=1e-6) -> torch.Tensor:
    x = torch.clamp(x, min=eps)
    return torch.log(torch.expm1(x))

@dataclass
class GaussianParameters:
    means_3d: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    scales_3d: torch.Tensor
    rotations: torch.Tensor



class GaussianModel(nn.Module):
    '''
    Uproszczony model zbioru Gaussów 3D.
    -mean_3d: środki gaussów 3D
    -colors: kolory RGB
    -opacities: przezroczystości (alpha) / waga
    -base_scales: podstawowe rozmiary gaussów w 3D (sigma)
    '''

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
    ) -> None:
        super().__init__()

        means_3d = means_3d.float()
        colors = colors.float()
        opacities = opacities.float().reshape(-1, 1)
        scales_3d = scales_3d.float()

        if rotations is None:
            rotations = torch.zeros((means_3d.shape[0], 4), device=means_3d.device)
            rotations[:, 0] = 1.0 # inicjalizacja jako brak rotacji (quaternion [1, 0, 0, 0])

        rotations = rotations.float()
        

        self.means_3d = nn.Parameter(means_3d, requires_grad=learn_means)
        self.colors_raw = nn.Parameter(inverse_sigmoid(colors), requires_grad=learn_colors)
        self.opacities_raw = nn.Parameter(inverse_sigmoid(opacities), requires_grad=learn_opacities)
        self.scales_raw = nn.Parameter(inverse_softplus(scales_3d), requires_grad=learn_scales)
        self.rotations_raw = nn.Parameter(rotations, requires_grad=learn_rotations)

    def get_parameters(self) -> GaussianParameters:

        ''' Zwraca parametry gaussów w formie gotowej do renderowania'''

        colors = torch.sigmoid(self.colors_raw)
        opacities = torch.sigmoid(self.opacities_raw).squeeze(-1)
        scales_3d = torch.nn.functional.softplus(self.scales_raw).squeeze(-1)

        rotations = self.rotations_raw
        rotations = rotations / (torch.norm(rotations, dim=-1, keepdim=True) + 1e-8) # normalizacja quaternionów
        
        return GaussianParameters(
            means_3d=self.means_3d,
            colors=colors,
            opacities=opacities,
            scales_3d=scales_3d,
            rotations=rotations
        )
    
    