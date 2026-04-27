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
    base_scales: torch.Tensor


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
        base_scales: torch.Tensor,
        learn_means: bool = False,
        learn_colors: bool = True,
        learn_opacities: bool = True,
        learn_scales: bool = True,
        # na razie nie uczymy wszystkiego bo geometria COLMAP jest ok, a tak łatwiej sprawdzić czy renderowanie i treninig działa
    ) -> None:
        super().__init__()

        means_3d = means_3d.float()
        colors = colors.float()
        opacities = opacities.float().reshape(-1, 1) #przekształcamy do (N, 1) żeby potem łatwiej było mnożyć przez kolory
        base_scales = base_scales.float().reshape(-1, 1) #przekształcamy do (N, 1) żeby potem łatwiej było mnożyć przez jakieś skalowanie

        self.means_3d = nn.Parameter(means_3d, requires_grad=learn_means)
        self.colors_raw = nn.Parameter(inverse_sigmoid(colors), requires_grad=learn_colors)
        self.opacities_raw = nn.Parameter(inverse_sigmoid(opacities), requires_grad=learn_opacities)
        self.base_scales_raw = nn.Parameter(inverse_softplus(base_scales), requires_grad=learn_scales)

    def get_parameters(self) -> GaussianParameters:

        ''' Zwraca parametry gaussów w formie gotowej do renderowania'''

        colors = torch.sigmoid(self.colors_raw)
        opacities = torch.sigmoid(self.opacities_raw).squeeze(-1)
        base_scales = torch.nn.functional.softplus(self.base_scales_raw).squeeze(-1)
        
        return GaussianParameters(
            means_3d=self.means_3d,
            colors=colors,
            opacities=opacities,
            base_scales=base_scales,
        )
    
    