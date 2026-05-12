from pathlib import Path

import numpy as np
import torch


def save_gaussian_model_pt(
    output_path: str | Path,
    means_3d: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    scales_3d: torch.Tensor,
    rotations: torch.Tensor,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "means_3d": means_3d.detach().cpu(),
            "colors": colors.detach().cpu(),
            "opacities": opacities.detach().cpu(),
            "scales_3d": scales_3d.detach().cpu(),
            "rotations": rotations.detach().cpu(),
        },
        output_path,
    )


def load_gaussian_model_pt(model_path: str | Path, device: torch.device):
    data = torch.load(model_path, map_location=device)

    return (
        data["means_3d"].to(device),
        data["colors"].to(device),
        data["opacities"].to(device),
        data["scales_3d"].to(device),
        data["rotations"].to(device),
    )


def rgb_to_sh_dc(colors: np.ndarray) -> np.ndarray:
    """
    Konwersja RGB [0,1] do f_dc dla formatu 3DGS.
    """
    C0 = 0.28209479177387814
    return (colors - 0.5) / C0


def inverse_sigmoid_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def normalize_quaternion_np(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return q / (np.linalg.norm(q, axis=1, keepdims=True) + eps)

def save_gaussian_splat_ply(
    output_path: str | Path,
    means_3d: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    scales_3d: torch.Tensor,
    rotations: torch.Tensor,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xyz = means_3d.detach().cpu().numpy().astype(np.float32)
    rgb = colors.detach().cpu().numpy().astype(np.float32)
    opacity = opacities.detach().cpu().numpy().reshape(-1).astype(np.float32)
    scales = scales_3d.detach().cpu().numpy().astype(np.float32)
    rots = rotations.detach().cpu().numpy().astype(np.float32)

    n = xyz.shape[0]

    opacity_raw = inverse_sigmoid_np(opacity).astype(np.float32)
    scales_log = np.log(np.clip(scales, 1e-8, None)).astype(np.float32)
    rots = normalize_quaternion_np(rots).astype(np.float32)
    f_dc = rgb_to_sh_dc(rgb).astype(np.float32)

    data = np.zeros(
        n,
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ],
    )

    data["x"] = xyz[:, 0]
    data["y"] = xyz[:, 1]
    data["z"] = xyz[:, 2]

    data["f_dc_0"] = f_dc[:, 0]
    data["f_dc_1"] = f_dc[:, 1]
    data["f_dc_2"] = f_dc[:, 2]

    data["opacity"] = opacity_raw

    data["scale_0"] = scales_log[:, 0]
    data["scale_1"] = scales_log[:, 1]
    data["scale_2"] = scales_log[:, 2]

    data["rot_0"] = rots[:, 0]
    data["rot_1"] = rots[:, 1]
    data["rot_2"] = rots[:, 2]
    data["rot_3"] = rots[:, 3]

    with open(output_path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float f_dc_0\n"
            "property float f_dc_1\n"
            "property float f_dc_2\n"
            "property float opacity\n"
            "property float scale_0\n"
            "property float scale_1\n"
            "property float scale_2\n"
            "property float rot_0\n"
            "property float rot_1\n"
            "property float rot_2\n"
            "property float rot_3\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        f.write(data.tobytes())