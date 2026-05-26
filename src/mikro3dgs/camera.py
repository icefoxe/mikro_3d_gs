from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch


@dataclass
class Camera:
    K: torch.Tensor
    R: torch.Tensor
    t: torch.Tensor
    image_size: Tuple[int, int]
    device: torch.device = torch.device("cpu")
    distortion_model: Optional[str] = None
    distortion_params: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        self.K = self.K.to(self.device).float()
        self.R = self.R.to(self.device).float()
        self.t = self.t.to(self.device).float()

        if self.K.shape != (3, 3):
            raise ValueError(f"K must have shape (3, 3), got {self.K.shape}")
        if self.R.shape != (3, 3):
            raise ValueError(f"R must have shape (3, 3), got {self.R.shape}")
        if self.t.shape not in [(3,), (3, 1)]:
            raise ValueError(f"t must have shape (3,) or (3, 1), got {self.t.shape}")

        self.t = self.t.reshape(3, 1)

        if self.distortion_model is not None:
            self.distortion_model = self.distortion_model.upper()

        if self.distortion_params is None:
            self.distortion_params = torch.empty(
                (0,), device=self.device, dtype=torch.float32
            )
        else:
            self.distortion_params = (
                self.distortion_params.to(self.device).float().reshape(-1)
            )

    @property
    def height(self) -> int:
        return self.image_size[0]

    @property
    def width(self) -> int:
        return self.image_size[1]

    def world_to_camera(self, points_world: torch.Tensor) -> torch.Tensor:
        if points_world.ndim != 2 or points_world.shape[1] != 3:
            raise ValueError(
                f"points_world must have shape (N, 3), got {points_world.shape}"
            )
        points_world = points_world.to(self.device).float()
        points_cam = (self.R @ points_world.T) + self.t
        return points_cam.T

    def _radial_coeffs(self) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        model = self.distortion_model
        p = self.distortion_params

        if model in (None, "SIMPLE_PINHOLE", "PINHOLE") or p.numel() == 0:
            return zero, zero
        if model == "SIMPLE_RADIAL":
            return p[0], zero
        if model == "RADIAL":
            return p[0], p[1] if p.numel() > 1 else zero
        raise NotImplementedError(
            f"Distortion model {model} is not supported by Camera.project"
        )

    def apply_distortion(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k1, k2 = self._radial_coeffs()
        if float(k1.detach().cpu()) == 0.0 and float(k2.detach().cpu()) == 0.0:
            return x, y

        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        return x * radial, y * radial

    def projection_jacobian(
        self, points_cam: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        points_cam = points_cam.to(self.device).float()
        X = points_cam[:, 0]
        Y = points_cam[:, 1]
        Z = points_cam[:, 2].clamp_min(eps)

        x = X / Z
        y = Y / Z

        fx = self.K[0, 0]
        fy = self.K[1, 1]

        k1, k2 = self._radial_coeffs()
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2

        dr_dx = 2.0 * k1 * x + 4.0 * k2 * x * r2
        dr_dy = 2.0 * k1 * y + 4.0 * k2 * y * r2

        du_dx = fx * (radial + x * dr_dx)
        du_dy = fx * (x * dr_dy)
        dv_dx = fy * (y * dr_dx)
        dv_dy = fy * (radial + y * dr_dy)

        dx_dX = 1.0 / Z
        dx_dY = torch.zeros_like(Z)
        dx_dZ = -X / (Z * Z)

        dy_dX = torch.zeros_like(Z)
        dy_dY = 1.0 / Z
        dy_dZ = -Y / (Z * Z)

        J = torch.zeros(
            (points_cam.shape[0], 2, 3), device=self.device, dtype=torch.float32
        )
        J[:, 0, 0] = du_dx * dx_dX + du_dy * dy_dX
        J[:, 0, 1] = du_dx * dx_dY + du_dy * dy_dY
        J[:, 0, 2] = du_dx * dx_dZ + du_dy * dy_dZ
        J[:, 1, 0] = dv_dx * dx_dX + dv_dy * dy_dX
        J[:, 1, 1] = dv_dx * dx_dY + dv_dy * dy_dY
        J[:, 1, 2] = dv_dx * dx_dZ + dv_dy * dy_dZ
        return J

    def project(self, points_world: torch.Tensor, eps: float = 1e-6):
        points_cam = self.world_to_camera(points_world)
        depth = points_cam[:, 2]
        valid_mask = depth > eps

        z = depth + eps
        x = points_cam[:, 0] / z
        y = points_cam[:, 1] / z

        x_d, y_d = self.apply_distortion(x, y)

        u = self.K[0, 0] * x_d + self.K[0, 2]
        v = self.K[1, 1] * y_d + self.K[1, 2]
        uv = torch.stack([u, v], dim=1)
        return uv, depth, valid_mask

    def in_image_mask(self, uv: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError(f"uv must have shape (N, 2), got {uv.shape}")

        u = uv[:, 0]
        v = uv[:, 1]
        inside = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return valid_mask & inside


def look_at(eye, target, up=None):
    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], device=eye.device)

    eye = eye.float()
    target = target.float()
    up = up.float().to(eye.device)

    z = eye - target
    z = z / (torch.norm(z) + 1e-8)
    x = torch.cross(up, z, dim=0)
    x = x / (torch.norm(x) + 1e-8)
    y = torch.cross(z, x, dim=0)

    R = torch.stack([x, y, z], dim=0)
    t = -R @ eye
    return R, t


def generate_orbit_cameras(base_camera, num_views=60, radius=2.0, target=None):
    cameras = []
    device = base_camera.device

    if target is None:
        target = torch.tensor([0.0, 0.0, 0.0], device=device)
    else:
        target = target.to(device)

    for i in range(num_views):
        angle = 2 * math.pi * i / num_views
        eye = torch.tensor(
            [
                radius * math.cos(angle),
                0.5,
                radius * math.sin(angle),
            ],
            device=device,
        )

        R, t = look_at(eye, target)
        cameras.append(
            Camera(
                K=base_camera.K.clone(),
                R=R,
                t=t,
                image_size=base_camera.image_size,
                device=device,
                distortion_model=base_camera.distortion_model,
                distortion_params=base_camera.distortion_params.clone(),
            )
        )

    return cameras
