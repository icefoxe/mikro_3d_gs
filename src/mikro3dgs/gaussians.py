from dataclasses import dataclass

import torch
import torch.nn as nn


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
        learn_scales: bool = False,
        # na razie nie uczymy wszystkiego bo geometria COLMAP jest ok, a tak łatwiej sprawdzić czy renderowanie i treninig działa
    ) -> None:
        super().__init__()

        means_3d = means_3d.float()
        colors = colors.float()
        opacities = opacities.float().reshape(-1, 1) #przekształcamy do (N, 1) żeby potem łatwiej było mnożyć przez kolory
        base_scales = base_scales.float().reshape(-1, 1) #przekształcamy do (N, 1) żeby potem łatwiej było mnożyć przez jakieś skalowanie

        self.means_3d = nn.Parameter(means_3d, requires_grad=learn_means)
        self.colors_raw = nn.Parameter(colors, requires_grad=learn_colors)
        self.opacities_raw = nn.Parameter(opacities, requires_grad=learn_opacities)
        self.base_scales_raw = nn.Parameter(base_scales, requires_grad=learn_scales)

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