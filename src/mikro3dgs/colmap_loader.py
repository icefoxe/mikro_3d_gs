from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
import torch

sys.path.append("src")

from mikro3dgs.camera import Camera


@dataclass
class ColmapCameraModel:
    camera_id: int
    model: str
    width: int
    height: int
    params: List[float]


@dataclass
class ColmapImage:
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str


@dataclass
class ColmapPoint3D:
    point3d_id: int
    xyz: Tuple[float, float, float]
    rgb: Tuple[int, int, int]
    error: float


def qvec_to_rotmat(qvec: torch.Tensor) -> torch.Tensor:
    if qvec.shape != (4,):
        raise ValueError(f"qvec must have shape (4,), got {qvec.shape}")

    qvec = qvec / (torch.norm(qvec) + 1e-8)
    qw, qx, qy, qz = qvec

    R = torch.stack(
        [
            torch.stack(
                [
                    1 - 2 * qy * qy - 2 * qz * qz,
                    2 * qx * qy - 2 * qz * qw,
                    2 * qx * qz + 2 * qy * qw,
                ]
            ),
            torch.stack(
                [
                    2 * qx * qy + 2 * qz * qw,
                    1 - 2 * qx * qx - 2 * qz * qz,
                    2 * qy * qz - 2 * qx * qw,
                ]
            ),
            torch.stack(
                [
                    2 * qx * qz - 2 * qy * qw,
                    2 * qy * qz + 2 * qx * qw,
                    1 - 2 * qx * qx - 2 * qy * qy,
                ]
            ),
        ]
    ).float()

    return R


def build_intrinsics(camera_model: ColmapCameraModel) -> torch.Tensor:
    model = camera_model.model.upper()
    p = camera_model.params

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = p
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, _k = p
        fx = fy = f
    elif model == "RADIAL":
        f, cx, cy, _k1, _k2 = p
        fx = fy = f
    else:
        raise NotImplementedError(f"Camera model {camera_model.model} not supported")

    return torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def build_distortion(
    camera_model: ColmapCameraModel, device: torch.device
) -> tuple[str, torch.Tensor]:
    model = camera_model.model.upper()
    p = camera_model.params

    if model in ("SIMPLE_PINHOLE", "PINHOLE"):
        return model, torch.empty((0,), device=device, dtype=torch.float32)
    if model == "SIMPLE_RADIAL":
        return model, torch.tensor([p[3]], device=device, dtype=torch.float32)
    if model == "RADIAL":
        return model, torch.tensor([p[3], p[4]], device=device, dtype=torch.float32)

    raise NotImplementedError(f"Camera model {camera_model.model} not supported")


class ColmapLoader:
    def __init__(
        self, model_dir: str | Path, device: torch.device = torch.device("cpu")
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device

        self.cameras_path = self.model_dir / "cameras.txt"
        self.images_path = self.model_dir / "images.txt"
        self.points3D_path = self.model_dir / "points3D.txt"

        for p in [self.cameras_path, self.images_path, self.points3D_path]:
            if not p.exists():
                raise FileNotFoundError(f"File {p} does not exist")

        self.cameras_models: Dict[int, ColmapCameraModel] = {}
        self.images: Dict[int, ColmapImage] = {}
        self.points3D: Dict[int, ColmapPoint3D] = {}

    def load_all(self) -> None:
        self.cameras_models = self._load_cameras()
        self.images = self._load_images()
        self.points3D = self._load_points3D()

    def _read_non_comment_lines(self, path: Path) -> List[str]:
        lines: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                lines.append(line)
        return lines

    def _load_cameras(self) -> Dict[int, ColmapCameraModel]:
        lines = self._read_non_comment_lines(self.cameras_path)
        cameras: Dict[int, ColmapCameraModel] = {}
        for line in lines:
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[camera_id] = ColmapCameraModel(
                camera_id, model, width, height, params
            )
        return cameras

    def _load_images(self) -> Dict[int, ColmapImage]:
        lines = self._read_non_comment_lines(self.images_path)
        images: Dict[int, ColmapImage] = {}

        if len(lines) % 2 != 0:
            raise ValueError("Expected even number of non-comment lines in images.txt")

        for i in range(0, len(lines), 2):
            meta = lines[i].split()
            image_id = int(meta[0])
            qw, qx, qy, qz = map(float, meta[1:5])
            tx, ty, tz = map(float, meta[5:8])
            camera_id = int(meta[8])
            name = meta[9]
            images[image_id] = ColmapImage(
                image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name
            )
        return images

    def _load_points3D(self) -> Dict[int, ColmapPoint3D]:
        lines = self._read_non_comment_lines(self.points3D_path)
        points3D: Dict[int, ColmapPoint3D] = {}

        for line in lines:
            parts = line.split()
            point3d_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            error = float(parts[7])
            points3D[point3d_id] = ColmapPoint3D(
                point3d_id, (x, y, z), (r, g, b), error
            )
        return points3D

    def get_points_xyz_rgb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.points3D:
            raise ValueError("No 3D points loaded")

        xyz = torch.tensor(
            [p.xyz for p in self.points3D.values()],
            dtype=torch.float32,
            device=self.device,
        )
        rgb = (
            torch.tensor(
                [p.rgb for p in self.points3D.values()],
                dtype=torch.float32,
                device=self.device,
            )
            / 255.0
        )
        return xyz, rgb

    def get_image_by_name(self, image_name: str) -> Optional[ColmapImage]:
        if not self.images:
            raise ValueError("No images loaded")
        for img in self.images.values():
            if img.name == image_name:
                return img
        raise ValueError(f"Image {image_name} not found in COLMAP data")

    def get_first_image(self) -> ColmapImage:
        if not self.images:
            raise ValueError("No images loaded")
        return self.images[sorted(self.images.keys())[0]]

    def build_camera(self, colmap_image: ColmapImage) -> Camera:
        cam_model = self.cameras_models[colmap_image.camera_id]

        K = build_intrinsics(cam_model).to(self.device)
        distortion_model, distortion_params = build_distortion(cam_model, self.device)

        qvec = torch.tensor(
            [colmap_image.qw, colmap_image.qx, colmap_image.qy, colmap_image.qz],
            dtype=torch.float32,
            device=self.device,
        )
        R = qvec_to_rotmat(qvec).to(self.device)

        t = torch.tensor(
            [colmap_image.tx, colmap_image.ty, colmap_image.tz],
            dtype=torch.float32,
            device=self.device,
        )

        return Camera(
            K=K,
            R=R,
            t=t,
            image_size=(cam_model.height, cam_model.width),
            device=self.device,
            distortion_model=distortion_model,
            distortion_params=distortion_params,
        )
