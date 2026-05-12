from dataclasses import dataclass
from typing import Tuple

import torch
import math


@dataclass
class Camera:
    """
    ledwo co to rozumiem
    model kamery pinhole z pełną macierzą intrinsics i dowolną rotacją

    args:
        K macierz intrinsics 3x3
        R macierz rotacji 3x3
        t wektor translacji 3
        image_size height, width
        device cpu cuda
    """

    K: torch.Tensor
    R: torch.Tensor
    t: torch.Tensor
    image_size: Tuple[int, int]
    device: torch.device = torch.device("cpu")

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

    @property
    def height(self) -> int:
        return self.image_size[0]

    @property
    def width(self) -> int:
        return self.image_size[1]

    def world_to_camera(self, points_world: torch.Tensor) -> torch.Tensor:
        """
        args:
            points_world: tensor shape (N, 3)

        returns:
            points_cam: tensor shape (N, 3)
        """
        if points_world.ndim != 2 or points_world.shape[1] != 3:
            raise ValueError(
                f"points_world must have shape (N, 3), got {points_world.shape}"
            )

        points_world = points_world.to(self.device).float()
        points_cam = (self.R @ points_world.T) + self.t
        return points_cam.T

    def project(self, points_world: torch.Tensor, eps: float = 1e-6):
        """
        rzutuje punkty 3D świata na obraz

        args:
            points_world: tensor (N, 3)
            eps: zabezpieczenie przed dzieleniem przez zero

        returns:
            uv: współrzędne pikseli (N, 2)
            depth: głębokość w układzie kamery (N,)
            valid_mask: maska punktów przed kamerą (N,)
        """
        points_cam = self.world_to_camera(points_world)  # (N, 3)
        depth = points_cam[:, 2]

        valid_mask = depth > eps

        x = points_cam[:, 0] / (depth + eps)
        y = points_cam[:, 1] / (depth + eps)

        ones = torch.ones_like(x)
        points_norm = torch.stack([x, y, ones], dim=1)  # (N, 3)

        uv_h = (self.K @ points_norm.T).T  # (N, 3)
        uv = uv_h[:, :2]

        return uv, depth, valid_mask

    def in_image_mask(self, uv: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        sprawdza które punkty leżą w granicach obrazu
        """
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError(f"uv must have shape (N, 2), got {uv.shape}")

        u = uv[:, 0]
        v = uv[:, 1]

        inside = (
            (u >= 0)
            & (u < self.width)
            & (v >= 0)
            & (v < self.height)
        )

        return valid_mask & inside
    
def look_at(eye, target, up=None):
    """
    buduje macierz R i t dla kamery patrzącej na target
    """
    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], device=eye.device)

    eye = eye.float()
    target = target.float()
    up = up.float().to(eye.device)

    z = (eye - target)
    z = z / (torch.norm(z) + 1e-8)
    
    x = torch.cross(up, z, dim=0)
    x = x / (torch.norm(x) + 1e-8)
    y = torch.cross(z, x, dim=0)

    R = torch.stack([x, y, z], dim=0)  # world -> camera

    t = -R @ eye
    print("poszlo")
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

        eye = torch.tensor([
            radius * math.cos(angle),
            0.5,  # lekko z góry
            radius * math.sin(angle)
        ], device=device)

        R, t = look_at(eye, target)
    #pozycja kamer pojedyńczo z każdej kamery
        cam = Camera(
            K=base_camera.K.clone(),
            R=R,
            t=t,
            image_size=base_camera.image_size,
            device=device
        )

        cameras.append(cam)
    print("poszlo")
    return cameras
