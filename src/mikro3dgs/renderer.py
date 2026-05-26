from dataclasses import dataclass
from typing import Optional
import os
import time

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

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        tile_gaussians: int = 256,
        full_image_tile_size: int = 128,
        full_image_gaussian_chunk: int = 96,
    ) -> None:
        self.device = device
        self.tile_gaussians = int(tile_gaussians)
        self.full_image_tile_size = int(full_image_tile_size)
        self.full_image_gaussian_chunk = int(full_image_gaussian_chunk)
        self.debug = os.environ.get("MIKRO3DGS_RENDER_DEBUG", "0") == "1"

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(f"[renderer] {msg}", flush=True)

    def _sync(self, label: str) -> None:
        if self.debug and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            print(f"[renderer] cuda sync after {label}", flush=True)

    def quaternion_to_rotmat(self, q: torch.Tensor) -> torch.Tensor:
        q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.empty((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
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
        min_var: float = 0.25,
        max_var: float = 4096.0,
    ) -> torch.Tensor:
        means_3d = means_3d.to(self.device).float()
        scales_3d = scales_3d.to(self.device).float()
        rotations = rotations.to(self.device).float()

        points_cam = camera.world_to_camera(means_3d)
        X, Y = points_cam[:, 0], points_cam[:, 1]
        Z = points_cam[:, 2].clamp_min(eps)
        fx, fy = camera.K[0, 0], camera.K[1, 1]

        N = means_3d.shape[0]
        J = torch.zeros((N, 2, 3), device=self.device, dtype=torch.float32)
        J[:, 0, 0] = fx / Z
        J[:, 0, 2] = -fx * X / (Z * Z)
        J[:, 1, 1] = fy / Z
        J[:, 1, 2] = -fy * Y / (Z * Z)

        R_gauss = self.quaternion_to_rotmat(rotations)
        S2 = torch.diag_embed(scales_3d * scales_3d)
        Sigma_world = R_gauss @ S2 @ R_gauss.transpose(1, 2)
        R_cam = camera.R.to(self.device).float()
        Sigma_cam = R_cam.unsqueeze(0) @ Sigma_world @ R_cam.T.unsqueeze(0)
        Sigma_2d = J @ Sigma_cam @ J.transpose(1, 2)

        eye = torch.eye(2, device=self.device, dtype=torch.float32).unsqueeze(0)
        Sigma_2d = Sigma_2d + min_var * eye

        eigvals, eigvecs = torch.linalg.eigh(Sigma_2d)
        eigvals = eigvals.clamp(min=min_var, max=max_var)
        return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(1, 2)

    def _prepare_inputs(self, means_3d, colors, opacities, scales_3d, rotations):
        means_3d = means_3d.to(self.device).float()
        colors = colors.to(self.device).float().clamp(0.0, 1.0)
        opacities = opacities.to(self.device).float().reshape(-1)
        scales_3d = scales_3d.to(self.device).float()
        if rotations is None:
            rotations = torch.zeros(
                (means_3d.shape[0], 4), device=self.device, dtype=torch.float32
            )
            rotations[:, 0] = 1.0
        else:
            rotations = rotations.to(self.device).float()
        return means_3d, colors, opacities, scales_3d, rotations

    def _render_region_from_precomputed(
        self,
        uv: torch.Tensor,
        depth: torch.Tensor,
        valid_mask: torch.Tensor,
        Sigma_2d: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        background: Optional[torch.Tensor],
        sigma_extent: float,
        gaussian_chunk: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_region = time.perf_counter()
        h, w = y1 - y0, x1 - x0
        self._dbg(
            f"region start x={x0}:{x1} y={y0}:{y1} size={w}x{h} gaussian_chunk={gaussian_chunk}"
        )
        image_acc = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
        alpha_acc = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

        self._dbg("region: before eigvalsh")
        eigvals = torch.linalg.eigvalsh(Sigma_2d.detach())
        self._sync("region eigvalsh")
        sigma_max = torch.sqrt(eigvals[:, -1].clamp_min(1e-6))
        margin = sigma_extent * sigma_max

        inside = (
            valid_mask
            & (depth > 1e-5)
            & (uv[:, 0] + margin >= x0)
            & (uv[:, 0] - margin < x1)
            & (uv[:, 1] + margin >= y0)
            & (uv[:, 1] - margin < y1)
        )
        idx = torch.where(inside)[0]
        self._sync("region candidate mask")
        self._dbg(f"region: candidate gaussians={idx.numel()}")

        if idx.numel() == 0:
            if background is not None:
                image_acc[:] = background.to(self.device).float().view(1, 1, 3)
            return image_acc, alpha_acc, inside

        self._dbg("region: before depth sort")
        idx = idx[torch.argsort(depth[idx])]  # front-to-back
        self._sync("region depth sort")
        uv_a = uv[idx]
        colors_a = colors[idx]
        alpha_a = opacities[idx]
        Sigma_a = Sigma_2d[idx]

        ys = torch.arange(y0, y1, device=self.device, dtype=torch.float32)
        xs = torch.arange(x0, x1, device=self.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")

        T = torch.ones((h, w, 1), device=self.device, dtype=torch.float32)

        chunk = max(1, int(gaussian_chunk))
        for start in range(0, idx.numel(), chunk):
            end = min(start + chunk, idx.numel())
            if self.debug and (start == 0 or start % (chunk * 10) == 0):
                self._dbg(f"region chunk {start}:{end}/{idx.numel()}")
            self._dbg("chunk: before inv_cov") if self.debug and start == 0 else None
            inv_cov = torch.linalg.inv(Sigma_a[start:end])
            self._sync("chunk inv_cov first") if self.debug and start == 0 else None
            uvc = uv_a[start:end]

            du = gx.unsqueeze(0) - uvc[:, 0].view(-1, 1, 1)
            dv = gy.unsqueeze(0) - uvc[:, 1].view(-1, 1, 1)
            mahal = (
                inv_cov[:, 0, 0].view(-1, 1, 1) * du * du
                + 2.0 * inv_cov[:, 0, 1].view(-1, 1, 1) * du * dv
                + inv_cov[:, 1, 1].view(-1, 1, 1) * dv * dv
            )

            self._sync("chunk mahal first") if self.debug and start == 0 else None
            gauss = torch.exp(-0.5 * mahal)
            alpha = (alpha_a[start:end].view(-1, 1, 1) * gauss).clamp(0.0, 0.995)
            one_minus = (1.0 - alpha).clamp_min(1e-6)

            local_T = torch.cumprod(
                torch.cat([torch.ones_like(one_minus[:1]), one_minus[:-1]], dim=0),
                dim=0,
            )
            weights = T.permute(2, 0, 1) * local_T * alpha

            image_acc = image_acc + torch.sum(
                weights.unsqueeze(-1) * colors_a[start:end].view(-1, 1, 1, 3),
                dim=0,
            )
            alpha_acc = alpha_acc + torch.sum(weights, dim=0).unsqueeze(-1)
            T = T * torch.prod(one_minus, dim=0).unsqueeze(-1)

        self._sync("region chunks done")
        self._dbg(f"region chunks done in {time.perf_counter() - t_region:.3f}s")
        alpha_acc = alpha_acc.clamp(0.0, 1.0)
        if background is not None:
            bg = background.to(self.device).float().view(1, 1, 3)
            image_acc = image_acc + bg * (1.0 - alpha_acc)

        return image_acc.clamp(0.0, 1.0), alpha_acc, inside

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
        with torch.no_grad():
            t_full = time.perf_counter()
            self._dbg("full render start")
            means_3d, colors, opacities, scales_3d, rotations = self._prepare_inputs(
                means_3d, colors, opacities, scales_3d, rotations
            )
            H, W = camera.image_size
            self._dbg("full render: before project")
            uv, depth, valid_mask = camera.project(means_3d)
            self._sync("full project")
            self._dbg("full render: before covariance")
            Sigma_2d = self.compute_2d_covariances(
                camera, means_3d, scales_3d, rotations
            )
            self._sync("full covariance")

            if background is None:
                image = torch.zeros((H, W, 3), device=self.device, dtype=torch.float32)
            else:
                image = (
                    background.to(self.device)
                    .float()
                    .view(1, 1, 3)
                    .expand(H, W, 3)
                    .clone()
                )
            alpha = torch.zeros((H, W, 1), device=self.device, dtype=torch.float32)
            inside_any = torch.zeros(
                (means_3d.shape[0],), device=self.device, dtype=torch.bool
            )

            tile = max(16, int(self.full_image_tile_size))
            chunk = max(8, int(self.full_image_gaussian_chunk))

            total_tiles = ((H + tile - 1) // tile) * ((W + tile - 1) // tile)
            tile_i = 0
            self._dbg(
                f"full render: H={H} W={W} tile={tile} total_tiles={total_tiles} chunk={chunk}"
            )
            for y0 in range(0, H, tile):
                y1 = min(H, y0 + tile)
                for x0 in range(0, W, tile):
                    x1 = min(W, x0 + tile)
                    tile_i += 1
                    self._dbg(
                        f"full tile {tile_i}/{total_tiles} x={x0}:{x1} y={y0}:{y1}"
                    )
                    tile_img, tile_alpha, tile_inside = (
                        self._render_region_from_precomputed(
                            uv=uv,
                            depth=depth,
                            valid_mask=valid_mask,
                            Sigma_2d=Sigma_2d,
                            colors=colors,
                            opacities=opacities,
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                            background=background,
                            sigma_extent=sigma_extent,
                            gaussian_chunk=chunk,
                        )
                    )
                    image[y0:y1, x0:x1] = tile_img
                    alpha[y0:y1, x0:x1] = tile_alpha
                    inside_any |= tile_inside

            self._sync("full render complete")
            self._dbg(f"full render done in {time.perf_counter() - t_full:.3f}s")
            return RenderOutput(
                image=image,
                alpha=alpha,
                uv=uv,
                depth=depth,
                valid_mask=valid_mask,
                inside_mask=inside_any,
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
        t_patch = time.perf_counter()
        self._dbg(
            f"patch render start x={patch_x} y={patch_y} size={patch_size} point_indices={None if point_indices is None else int(point_indices.numel())}"
        )
        means_3d, colors, opacities, scales_3d, rotations = self._prepare_inputs(
            means_3d, colors, opacities, scales_3d, rotations
        )

        if point_indices is not None:
            point_indices = point_indices.to(self.device).long()
            means_3d = means_3d[point_indices]
            colors = colors[point_indices]
            opacities = opacities[point_indices]
            scales_3d = scales_3d[point_indices]
            rotations = rotations[point_indices]

        H, W = camera.image_size
        x0 = int(patch_x)
        y0 = int(patch_y)
        x1 = min(W, x0 + int(patch_size))
        y1 = min(H, y0 + int(patch_size))

        self._dbg("patch: before project")
        uv, depth, valid_mask = camera.project(means_3d)
        self._sync("patch project")
        self._dbg("patch: before covariance")
        Sigma_2d = self.compute_2d_covariances(camera, means_3d, scales_3d, rotations)
        self._sync("patch covariance")
        self._dbg("patch: before region render")
        image, alpha, inside = self._render_region_from_precomputed(
            uv=uv,
            depth=depth,
            valid_mask=valid_mask,
            Sigma_2d=Sigma_2d,
            colors=colors,
            opacities=opacities,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            background=background,
            sigma_extent=sigma_extent,
            gaussian_chunk=self.tile_gaussians,
        )

        self._sync("patch render complete")
        self._dbg(f"patch render done in {time.perf_counter() - t_patch:.3f}s")
        return RenderOutput(
            image=image,
            alpha=alpha,
            uv=uv,
            depth=depth,
            valid_mask=valid_mask,
            inside_mask=inside,
        )
