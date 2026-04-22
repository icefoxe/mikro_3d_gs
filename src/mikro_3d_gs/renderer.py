from dataclasses import dataclass
from typing import Optional

import torch

from mikro_3d_gs.camera import Camera


@dataclass
class RenderOutput:
    image: torch.Tensor
    alpha: torch.Tensor
    uv: torch.Tensor
    depth: torch.Tensor
    valid_mask: torch.Tensor
    inside_mask: torch.Tensor


class GaussianRenderer:
    """
    uproszczony renderer 3D

    każdy punkt 3D:
    -jest rzutowany do 2D
    -dostaje promień zależny od głębokości
    -generuje lokalny splat Gaussa na obrazie
    """

    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        self.device = device

    def _make_pixel_grid(self, height: int, width: int) -> torch.Tensor:
        """
        Tworzy siatkę pikseli o shape (H, W, 2),
        gdzie ostatni wymiar to (u, v).
        """
        ys = torch.arange(height, device=self.device, dtype=torch.float32)
        xs = torch.arange(width, device=self.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid

    def compute_2d_radius(
        self,
        base_scales: torch.Tensor,
        depth: torch.Tensor,
        focal_length: float,
        min_scale: float = 1.0,
        max_scale: float = 50.0,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        urposzczony model rozmiaru splatu 2d z tego wzoru:
            sigma_2d = f * sigma_3d / z

        args:
            base_scales: (N,) bazowy rozmiar Gaussa w 3D
            depth: (N,) głębokość
            focal_length: skalar, np. fx
        returns:
            sigma_2d: (N,)
        """

        sigma_2d = focal_length * base_scales / (depth + eps)
        sigma_2d = torch.clamp(sigma_2d, min=min_scale, max=max_scale)
        return sigma_2d

    def render(
        self,
        camera: Camera,
        means_3d: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        base_scales: torch.Tensor,
        background: Optional[torch.Tensor] = None,
    ) -> RenderOutput:
        """
        renderuje obraz jako sumę 2D Gaussian splats

        każdy punkt 3D staje się Gaussianem 2D, rozmiar splatu zależy od głębokości
        składanie jest przez ważoną sumę + alpha normalization
        """

        means_3d = means_3d.to(self.device).float()
        colors = colors.to(self.device).float()
        opacities = opacities.to(self.device).float().reshape(-1, 1)
        base_scales = base_scales.to(self.device).float().reshape(-1)

        if means_3d.ndim != 2 or means_3d.shape[1] != 3:
            raise ValueError(f"means_3d must have shape (N, 3), got {means_3d.shape}")
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError(f"colors must have shape (N, 3), got {colors.shape}")
        if opacities.shape[0] != means_3d.shape[0]:
            raise ValueError("opacities must match number of gaussians")
        if base_scales.shape[0] != means_3d.shape[0]:
            raise ValueError("base_scales must match number of gaussians")

        H, W = camera.image_size
        grid = self._make_pixel_grid(H, W)

        uv, depth, valid_mask = camera.project(means_3d)
        inside_mask = camera.in_image_mask(uv, valid_mask)

        fx = camera.K[0, 0].item()
        sigma_2d = self.compute_2d_radius(base_scales, depth, focal_length=fx)

        image_acc = torch.zeros((H, W, 3), device=self.device, dtype=torch.float32)
        alpha_acc = torch.zeros((H, W, 1), device=self.device, dtype=torch.float32)

        valid_indices = torch.where(inside_mask)[0]

        for idx in valid_indices:
            center = uv[idx]
            color = colors[idx]
            alpha = opacities[idx] 
            sigma = sigma_2d[idx] 
            diff = grid - center.view(1, 1, 2)  
            dist2 = (diff ** 2).sum(dim=-1, keepdim=True)
            gaussian = torch.exp(-0.5 * dist2 / (sigma ** 2 + 1e-6))
            weight = alpha.view(1, 1, 1) * gaussian     
            image_acc = image_acc + weight * color.view(1, 1, 3)
            alpha_acc = alpha_acc + weight

        if background is None:
            background = torch.zeros((1, 1, 3), device=self.device, dtype=torch.float32)
        else:
            background = background.to(self.device).float().view(1, 1, 3)

        image = image_acc / (alpha_acc + 1e-8)
        image = image * (alpha_acc > 1e-8).float() + background * (alpha_acc <= 1e-8).float()

        image = torch.clamp(image, 0.0, 1.0)
        alpha_acc = torch.clamp(alpha_acc, 0.0, 1.0)

        return RenderOutput(
            image=image,
            alpha=alpha_acc,
            uv=uv,
            depth=depth,
            valid_mask=valid_mask,
            inside_mask=inside_mask,
        )