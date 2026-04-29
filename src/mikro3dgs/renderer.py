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

    def __init__(self, device: torch.device = torch.device("cuda")) -> None:
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
        min_scale: float = 1.5,
        max_scale: float = 8.0,
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
    
    def quaternion_to_rotmat(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: (N, 4) jako [w, x, y, z]
        returns: (N, 3, 3)
        """
        q = q / (q.norm(dim=1, keepdim=True) + 1e-8)

        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)

        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)

        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - w * x)

        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)

        return R


    def compute_2d_covariances(
        self,
        camera: Camera,
        means_3d: torch.Tensor,
        scales_3d: torch.Tensor,
        rotations: torch.Tensor,
        eps: float = 1e-6,
        min_var: float = 1.0,
        max_var: float = 100.0,
    ) -> torch.Tensor:
        """
        Liczy projekcję kowariancji 3D Gaussa do kowariancji 2D.

        Sigma_3D = R_gauss @ diag(sx^2, sy^2, sz^2) @ R_gauss.T
        Sigma_cam = R_camera @ Sigma_3D @ R_camera.T
        Sigma_2D = J @ Sigma_cam @ J.T
        """
        means_3d = means_3d.to(self.device).float()
        scales_3d = scales_3d.to(self.device).float()
        rotations = rotations.to(self.device).float()

        points_cam = camera.world_to_camera(means_3d)
        X = points_cam[:, 0]
        Y = points_cam[:, 1]
        Z = torch.clamp(points_cam[:, 2], min=eps)

        fx = camera.K[0, 0]
        fy = camera.K[1, 1]

        N = means_3d.shape[0]

        # Jacobian projekcji 3D -> 2D
        J = torch.zeros((N, 2, 3), device=self.device, dtype=torch.float32)
        J[:, 0, 0] = fx / Z
        J[:, 0, 2] = -fx * X / (Z * Z)
        J[:, 1, 1] = fy / Z
        J[:, 1, 2] = -fy * Y / (Z * Z)

        R_gauss = self.quaternion_to_rotmat(rotations)

        S2 = torch.diag_embed(scales_3d ** 2)
        Sigma_world = R_gauss @ S2 @ R_gauss.transpose(1, 2)

        R_cam = camera.R.to(self.device).float()
        Sigma_cam = R_cam.unsqueeze(0) @ Sigma_world @ R_cam.T.unsqueeze(0)

        Sigma_2d = J @ Sigma_cam @ J.transpose(1, 2)

        eye = torch.eye(2, device=self.device).unsqueeze(0)

        Sigma_2d = Sigma_2d + min_var * eye

        return Sigma_2d

    def render(
        self,
        camera: Camera,
        means_3d: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        scales_3d: torch.Tensor,
        background: Optional[torch.Tensor] = None,
        sigma_extent: float = 3.0,     
        rotations: Optional[torch.Tensor] = None,
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
        scales_3d = scales_3d.to(self.device).float()

        

        if means_3d.ndim != 2 or means_3d.shape[1] != 3:
            raise ValueError(f"means_3d must have shape (N, 3), got {means_3d.shape}")
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError(f"colors must have shape (N, 3), got {colors.shape}")
        if opacities.shape[0] != means_3d.shape[0]:
            raise ValueError("opacities must match number of gaussians")
        if scales_3d.shape[0] != means_3d.shape[0]:
            raise ValueError("scales_3d must match number of gaussians")
        if scales_3d.ndim != 2 or scales_3d.shape[1] != 3:
            raise ValueError(f"scales_3d must have shape (N, 3), got {scales_3d.shape}")
        
        if rotations is None:
            rotations = torch.zeros((means_3d.shape[0], 4), device=self.device)
            rotations[:, 0] = 1.0
        else:
            rotations = rotations.to(self.device).float()

        H, W = camera.image_size

        uv, depth, valid_mask = camera.project(means_3d)
        inside_mask = camera.in_image_mask(uv, valid_mask)

        Sigma_2d = self.compute_2d_covariances(
            camera=camera,
            means_3d=means_3d,
            scales_3d=scales_3d,
            rotations=rotations,
        )

        eigvals = torch.linalg.eigvalsh(Sigma_2d.detach())
        sigma_max = torch.sqrt(torch.clamp(eigvals[:, -1], min=1e-6))
        margin = sigma_extent * sigma_max

        image_acc = torch.zeros((H, W, 3), device=self.device, dtype=torch.float32)
        alpha_acc = torch.zeros((H, W, 1), device=self.device, dtype=torch.float32)

        valid_mask_for_render = (
            inside_mask &
            (uv[:, 0] + margin >= 0) &
            (uv[:, 0] - margin < W) &
            (uv[:, 1] + margin >= 0) &
            (uv[:, 1] - margin < H)

        )

        valid_indices = torch.where(valid_mask_for_render)[0].tolist()

        eye2 = torch.eye(2, device=self.device, dtype=torch.float32)

        for idx in tqdm(valid_indices, desc="Rendering Gaussians"):

            center = uv[idx]
            color = colors[idx]
            alpha = opacities[idx] 

            cov2d = Sigma_2d[idx] + eye2 * 1e-4
            sigma = sigma_max[idx] 

            u0 = center[0].item()
            v0 = center[1].item()

            radius = max(1, int((sigma_extent * sigma).item()))

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

            diff = torch.stack((du, dv), dim=2)  # (h, w, 2)

            inv_cov = torch.linalg.inv(cov2d)

            mahal = (
                inv_cov[0, 0] * diff[..., 0] ** 2
                + 2.0 * inv_cov[0, 1] * diff[..., 0] * diff[..., 1]
                + inv_cov[1, 1] * diff[..., 1] ** 2
            )

            gaussian = torch.exp(-0.5 * mahal).unsqueeze(-1)

            weight = alpha.view(1, 1, 1) * gaussian     
            image_acc[y_min:y_max, x_min:x_max] += weight * color.view(1, 1, 3)
            alpha_acc[y_min:y_max, x_min:x_max] += weight

        if background is None:
            background = torch.zeros((1, 1, 3), device=self.device, dtype=torch.float32)
        else:
            background = background.to(self.device).float().view(1, 1, 3)

        color_avg = image_acc / (alpha_acc + 1e-8)

        alpha_vis = 1.0 - torch.exp(-0.3 * alpha_acc)
        alpha_vis = torch.clamp(alpha_vis, 0.0, 1.0)

        image = color_avg * alpha_vis + background * (1.0 - alpha_vis)
        image = torch.clamp(image, 0.0, 1.0)


        return RenderOutput(
            image=image,
            alpha=alpha_vis,
            uv=uv,
            depth=depth,
            valid_mask=valid_mask,
            inside_mask=valid_mask_for_render,
        )
    

    def render_patch(
        self,
        camera: Camera,
        means_3d: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        scales_3d: torch.Tensor,
        patch_x: int,
        patch_y: int,
        patch_size: int,
        rotations: Optional[torch.Tensor] = None,
        point_indices: Optional[torch.Tensor] = None,
        background: Optional[torch.Tensor] = None,
        sigma_extent: float = 3.0,
    ) -> RenderOutput:
        """
        Wektorowy renderer patcha.

        Zamiast pętli po Gaussianach:
            for idx in valid_indices:
        liczymy wkład wszystkich aktywnych Gaussianów naraz.

        Uwaga:
        - point_indices mocno zalecane
        - max_patch_points w treningu powinno być np. 500–2000
        """

        means_3d = means_3d.to(self.device).float()
        colors = colors.to(self.device).float()
        opacities = opacities.to(self.device).float().reshape(-1)
        scales_3d = scales_3d.to(self.device).float()

        if rotations is not None:
            rotations = rotations.to(self.device).float()

        if point_indices is not None:
            point_indices = point_indices.to(self.device).long()

            means_3d = means_3d[point_indices]
            colors = colors[point_indices]
            opacities = opacities[point_indices]
            scales_3d = scales_3d[point_indices]

            if rotations is not None:
                rotations = rotations[point_indices]

        if means_3d.shape[0] == 0:
            patch_h = patch_size
            patch_w = patch_size
            image = torch.zeros((patch_h, patch_w, 3), device=self.device)
            alpha = torch.zeros((patch_h, patch_w, 1), device=self.device)
            dummy = torch.empty((0,), device=self.device)
            return RenderOutput(
                image=image,
                alpha=alpha,
                uv=torch.empty((0, 2), device=self.device),
                depth=dummy,
                valid_mask=torch.empty((0,), dtype=torch.bool, device=self.device),
                inside_mask=torch.empty((0,), dtype=torch.bool, device=self.device),
            )

        if scales_3d.ndim != 2 or scales_3d.shape[1] != 3:
            raise ValueError(f"scales_3d must have shape (N, 3), got {scales_3d.shape}")

        if rotations is None:
            rotations = torch.zeros((means_3d.shape[0], 4), device=self.device)
            rotations[:, 0] = 1.0

        H_full, W_full = camera.image_size

        x0 = patch_x
        y0 = patch_y
        x1 = min(W_full, x0 + patch_size)
        y1 = min(H_full, y0 + patch_size)

        patch_w = x1 - x0
        patch_h = y1 - y0

        if patch_w <= 0 or patch_h <= 0:
            raise RuntimeError(f"Invalid patch size: patch_w={patch_w}, patch_h={patch_h}")

        uv, depth, valid_mask = camera.project(means_3d)
        inside_mask = camera.in_image_mask(uv, valid_mask)

        Sigma_2d = self.compute_2d_covariances(
            camera=camera,
            means_3d=means_3d,
            scales_3d=scales_3d,
            rotations=rotations,
        )

        eigvals = torch.linalg.eigvalsh(Sigma_2d.detach())
        sigma_max = torch.sqrt(torch.clamp(eigvals[:, -1], min=1e-6))
        margin = sigma_extent * sigma_max

        patch_mask = (
            inside_mask
            & (uv[:, 0] + margin >= x0)
            & (uv[:, 0] - margin < x1)
            & (uv[:, 1] + margin >= y0)
            & (uv[:, 1] - margin < y1)
        )

        active_idx = torch.where(patch_mask)[0]

        if active_idx.numel() == 0:
            image = torch.zeros((patch_h, patch_w, 3), device=self.device)
            alpha = torch.zeros((patch_h, patch_w, 1), device=self.device)
            return RenderOutput(
                image=image,
                alpha=alpha,
                uv=uv,
                depth=depth,
                valid_mask=valid_mask,
                inside_mask=patch_mask,
            )

        uv_a = uv[active_idx]
        colors_a = colors[active_idx]
        opacities_a = opacities[active_idx]
        Sigma_a = Sigma_2d[active_idx]

        eye2 = torch.eye(2, device=self.device, dtype=torch.float32).unsqueeze(0)
        Sigma_a = Sigma_a + eye2 * 1e-4
        inv_cov = torch.linalg.inv(Sigma_a)

        ys = torch.arange(y0, y1, device=self.device, dtype=torch.float32)
        xs = torch.arange(x0, x1, device=self.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        du = grid_x.unsqueeze(0) - uv_a[:, 0].view(-1, 1, 1)
        dv = grid_y.unsqueeze(0) - uv_a[:, 1].view(-1, 1, 1)

        mahal = (
            inv_cov[:, 0, 0].view(-1, 1, 1) * du ** 2
            + 2.0 * inv_cov[:, 0, 1].view(-1, 1, 1) * du * dv
            + inv_cov[:, 1, 1].view(-1, 1, 1) * dv ** 2
        )

        gaussian = torch.exp(-0.5 * mahal)

        weights = opacities_a.view(-1, 1, 1) * gaussian

        image_acc = torch.sum(
            weights.unsqueeze(-1) * colors_a.view(-1, 1, 1, 3),
            dim=0,
        )  # (H, W, 3)

        alpha_acc = torch.sum(weights, dim=0, keepdim=False).unsqueeze(-1)

        if background is None:
            background = torch.zeros((1, 1, 3), device=self.device, dtype=torch.float32)
        else:
            background = background.to(self.device).float().view(1, 1, 3)

        color_avg = image_acc / (alpha_acc + 1e-8)

        alpha_vis = 1.0 - torch.exp(-alpha_acc)
        alpha_vis = torch.clamp(alpha_vis, 0.0, 1.0)

        image = color_avg * alpha_vis + background * (1.0 - alpha_vis)
        image = torch.clamp(image, 0.0, 1.0)

        return RenderOutput(
            image=image,
            alpha=alpha_vis,
            uv=uv,
            depth=depth,
            valid_mask=valid_mask,
            inside_mask=patch_mask,
        )