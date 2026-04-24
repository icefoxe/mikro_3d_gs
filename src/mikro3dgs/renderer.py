from dataclasses import dataclass
from typing import Optional


from tqdm import tqdm

import torch

from mikro3dgs.camera import Camera


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
        sigma_extent: float = 3.0,
    ) -> RenderOutput:
        """
        Szybszy render, kazdy gaussian liczony tylko w lokalnym oknie, a nie na całym obrazie, ale bez mipmap i innych bajerów.
        Renderuje obraz jako sumę 2D Gaussian splats

        Każdy punkt 3D staje się Gaussianem 2D, rozmiar splatu zależy od głębokości
        Składanie jest przez ważoną sumę + alpha normalization
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

        uv, depth, valid_mask = camera.project(means_3d)
        inside_mask = camera.in_image_mask(uv, valid_mask)

        fx = camera.K[0, 0].item()
        sigma_2d = self.compute_2d_radius(base_scales, depth, focal_length=fx)

        image_acc = torch.zeros((H, W, 3), device=self.device, dtype=torch.float32)
        alpha_acc = torch.zeros((H, W, 1), device=self.device, dtype=torch.float32)

        valid_indices = torch.where(inside_mask)[0].tolist()

        for idx in tqdm(valid_indices, desc="Rendering Gaussians"):

            center = uv[idx]
            color = colors[idx]
            alpha = opacities[idx] 
            sigma = sigma_2d[idx] 

            u0 = center[0].item()
            v0 = center[1].item()

            radius = max(1, int(sigma_extent * sigma))

            x_min = max(0, int(u0) - radius)
            y_min = max(0, int(v0) - radius)
            x_max = min(W, int(u0) + radius + 1)
            y_max = min(H, int(v0) + radius + 1)

            if x_min >= x_max or y_min >= y_max:
                continue

            ys = torch.arange(y_min, y_max, device=self.device, dtype=torch.float32)
            xs = torch.arange(x_min, x_max, device=self.device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

            du = grid_x - center[0]
            dv = grid_y - center[1]
            dist2 = (du ** 2 + dv ** 2).unsqueeze(-1)

            gaussian = torch.exp(-0.5 * dist2 / (sigma ** 2 + 1e-6))
            weight = alpha.view(1, 1, 1) * gaussian     
            image_acc[y_min:y_max, x_min:x_max] += weight * color.view(1, 1, 3)
            alpha_acc[y_min:y_max, x_min:x_max] += weight

        if background is None:
            background = torch.zeros((1, 1, 3), device=self.device, dtype=torch.float32)
        else:
            background = background.to(self.device).float().view(1, 1, 3)


        image = image_acc / (alpha_acc + 1e-8)
        alpha_acc = torch.clamp(alpha_acc, 0.0, 1.0)

        return RenderOutput(
            image=image,
            alpha=alpha_acc,
            uv=uv,
            depth=depth,
            valid_mask=valid_mask,
            inside_mask=inside_mask,
        )
    

    def render_patch(
            self, 
            camera: Camera,
            means_3d: torch.Tensor,
            colors: torch.Tensor,
            opacities: torch.Tensor,
            base_scales: torch.Tensor,
            patch_x: int,
            patch_y: int,
            patch_size: int,
            background: Optional[torch.Tensor] = None,
            sigma_extent: float = 3.0,
        )   -> RenderOutput:
        """Renderuje tylko fragment obrazu.
        patch_x, patch_y to współrzędne lewego górnego rogu patcha
        """
        means_3d = means_3d.to(self.device).float()
        colors = colors.to(self.device).float()
        opacities = opacities.to(self.device).float().reshape(-1, 1)
        base_scales = base_scales.to(self.device).float().reshape(-1)

        H_full, W_full = camera.image_size

        patch_w = patch_size
        patch_h = patch_size

        x0 = patch_x
        y0 = patch_y
        x1 = min(W_full, x0 + patch_w)
        y1 = min(H_full, y0 + patch_h)

        patch_w = x1 - x0
        patch_h = y1 - y0

        uv, depth, valid_mask = camera.project(means_3d)
        inside_mask = camera.in_image_mask(uv, valid_mask)

        fx = camera.K[0, 0].item()
        sigma_2d = self.compute_2d_radius(base_scales, depth, focal_length=fx)

        margain = sigma_extent * sigma_2d

        patch_mask = (
            inside_mask &
            (uv[:, 0] + margain >= x0) &
            (uv[:, 0] - margain < x1) &
            (uv[:, 1] + margain >= y0) &
            (uv[:, 1] - margain < y1)
        )

        valid_intdices = torch.where(patch_mask)[0].tolist()

        image_acc = torch.zeros((patch_h, patch_w, 3), device=self.device, dtype=torch.float32)
        alpha_acc = torch.zeros((patch_h, patch_w, 1), device=self.device, dtype=torch.float32)

        for idx in tqdm(valid_intdices, desc="Rendering patch Gaussians"):

            center = uv[idx]
            color = colors[idx]
            alpha = opacities[idx] 
            sigma = sigma_2d[idx] 

            u0 = center[0].item()
            v0 = center[1].item()

            radius = max(1, int(sigma_extent * sigma))

            gx_min = max(x0, int(u0) - radius)
            gy_min = max(y0, int(v0) - radius)
            gx_max = min(x1, int(u0) + radius + 1)
            gy_max = min(y1, int(v0) + radius + 1)

            if gx_min >= gx_max or gy_min >= gy_max:
                continue

            ys = torch.arange(gy_min, gy_max, device=self.device, dtype=torch.float32)
            xs = torch.arange(gx_min, gx_max, device=self.device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

            du = grid_x - center[0]
            dv = grid_y - center[1]
            dist2 = (du ** 2 + dv ** 2).unsqueeze(-1)

            gaussian = torch.exp(-0.5 * dist2 / (sigma ** 2 + 1e-6))
            weight = alpha.view(1, 1, 1) * gaussian  

            lx_min = gx_min - x0
            ly_min = gy_min - y0
            lx_max = gx_max - x0
            ly_max = gy_max - y0

            image_acc[ly_min:ly_max, lx_min:lx_max] += weight * color.view(1, 1, 3)
            alpha_acc[ly_min:ly_max, lx_min:lx_max] += weight

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