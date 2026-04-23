from dataclasses import dataclass
from pathlib import Path
import torch
from typing import Tuple, List, Optional, Dict

import sys
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
    """
    Konwertuje kwaternion (qw, qx, qy, qz) na macierz rotacji 3x3.
    """
    if qvec.shape != (4,):
        raise ValueError(f"qvec must have shape (4,), got {qvec.shape}")
    
    qw, qx, qy, qz = qvec
    
    R = torch.tensor([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2],
    ], dtype=torch.float32) #upenić się, że jest poprawnie napisana macierz, bo to AI zrobiło, a ja nie znam się na kwaternionach
    
    return R

def build_intrinsics(camera_model: ColmapCameraModel) -> torch.Tensor:
    """
    Buduje macierz intrinsics K z parametrów COLMAP.

    Obsługuje modele:
    -SIMPLE_PINHOLE: fx, cx, cy
    -PINHOLE: fx, fy, cx, cy
    -SIMPLE_RADIAL: f, cx, cy, k
    -RADIAL: f, cx, cy, k1, k2
    """

    model = camera_model.model
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
        raise NotImplementedError(f"Camera model {model} not supported")

    K = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    return K

class ColmapLoader:
    def __init__(self, model_dir: str | Path, device: torch.device = torch.device("cpu")) -> None:
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
                camera_id=camera_id,
                model=model,
                width=width,
                height=height,
                params=params,
            )
        return cameras
    
    def _load_images(self) -> Dict[int, ColmapImage]:
        lines = self._read_non_comment_lines(self.images_path)
        images: Dict[int, ColmapImage] = {}
        '''
        W COLMAP images.txt
        metadata obrazu line 1,
        2d points line 2, i tak na przemian
        '''
        if len(lines) % 2 != 0:
            raise ValueError("Expected even number of lines in images.txt")
        
        for i in range(0, len(lines), 2):
            meta = lines[i].split()

            image_id = int(meta[0])
            qw, qx, qy, qz = map(float, meta[1:5])
            tx, ty, tz = map(float, meta[5:8])
            camera_id = int(meta[8])
            name = meta[9]

            images[image_id] = ColmapImage(
                image_id=image_id,
                qw=qw,
                qx=qx,
                qy=qy,
                qz=qz,
                tx=tx,
                ty=ty,
                tz=tz,
                camera_id=camera_id,
                name=name,
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
                point3d_id=point3d_id,
                xyz=(x, y, z),
                rgb=(r, g, b),
                error=error,
            )
        return points3D
    
    def get_points_xyz_rgb(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Zwraca współrzędne XYZ i kolory RGB punktów 3D jako tensory.
        """
        if not self.points3D:
            raise ValueError("No 3D points loaded")
        
        xyz_list = []
        rgb_list = []

        for point in self.points3D.values():
            xyz_list.append(point.xyz)
            rgb_list.append(point.rgb)

        xyz = torch.tensor(xyz_list, dtype=torch.float32, device=self.device)
        rgb = torch.tensor(rgb_list, dtype=torch.float32, device=self.device) / 255.0

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
        first_key = sorted(self.images.keys())[0]
        return self.images[first_key]
    
    def build_camera(self, colmap_image: ColmapImage) -> Camera:
        '''
        Buduje obiekt Camera do pipelineu.

        COLMAP używa transformacji x_cam = R * x_world + t 
        t to translacja w układzie kamery, a R to pozycja kamery w świecie. W naszym Camera world_to_camera robi x_cam = R * x_world + t, więc można bezpośrednio użyć R i t z COLMAPa.
        '''

        cam_model = self.cameras_models[colmap_image.camera_id]

        K = build_intrinsics(cam_model).to(self.device)

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

        camera = Camera(
            K=K,
            R=R,
            t=t,
            image_size=(cam_model.height, cam_model.width),
            device=self.device,
        )
        return camera