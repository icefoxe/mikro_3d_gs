from pathlib import Path
import torch
import numpy as np

def save_gaussian_model_npz(
        output_path: str | Path,
        means_3d: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        base_scales: torch.Tensor,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "means_3d": means_3d.detach().cpu(),
        "colors": colors.detach().cpu(),
        "opacities": opacities.detach().cpu(),
        "base_scales": base_scales.detach().cpu(),
    }, output_path)

def load_gaussian_model_npz(model_path: str | Path, device: torch.device):
    data = torch.load(model_path, map_location=device)
    return (
        data["means_3d"].to(device),
        data["colors"].to(device),
        data["opacities"].to(device),
        data["base_scales"].to(device),
    )

def rgb_to_sh_dc(colors: np.ndarray) -> np.ndarray:
    """
    Przybliżona konwersja RGB [0,1] do f_dc dla formatu 3DGS.
    W oryginalnym 3DGS kolor DC jest skalowany przez stałą SH C0.
    """
    C0 = 0.28209479177387814
    return (colors - 0.5) / C0


def inverse_sigmoid_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def save_gaussian_splat_ply(
    output_path: str | Path,
    means_3d: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    base_scales: torch.Tensor,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xyz = means_3d.detach().cpu().numpy().astype(np.float32)
    rgb = colors.detach().cpu().numpy().astype(np.float32)
    opacity = opacities.detach().cpu().numpy().reshape(-1).astype(np.float32)
    scale = base_scales.detach().cpu().numpy().reshape(-1).astype(np.float32)

    n = xyz.shape[0]

    # 3DGS zapisuje opacity jako logit
    opacity_raw = inverse_sigmoid_np(opacity).astype(np.float32)

    # 3DGS zapisuje skale najczęściej w log-space
    scale = np.clip(scale, 1e-8, None)
    scale_log = np.log(scale).astype(np.float32)

    # kolor jako DC coefficient
    f_dc = rgb_to_sh_dc(rgb).astype(np.float32)

    # brak anizotropii → ta sama skala w XYZ
    scale_0 = scale_log
    scale_1 = scale_log
    scale_2 = scale_log

    # brak rotacji → quaternion identity
    rot_0 = np.ones(n, dtype=np.float32)
    rot_1 = np.zeros(n, dtype=np.float32)
    rot_2 = np.zeros(n, dtype=np.float32)
    rot_3 = np.zeros(n, dtype=np.float32)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property float f_dc_0\n")
        f.write("property float f_dc_1\n")
        f.write("property float f_dc_2\n")
        f.write("property float opacity\n")
        f.write("property float scale_0\n")
        f.write("property float scale_1\n")
        f.write("property float scale_2\n")
        f.write("property float rot_0\n")
        f.write("property float rot_1\n")
        f.write("property float rot_2\n")
        f.write("property float rot_3\n")
        f.write("end_header\n")

        for i in range(n):
            f.write(
                f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]} "
                f"0 0 0 "
                f"{f_dc[i,0]} {f_dc[i,1]} {f_dc[i,2]} "
                f"{opacity_raw[i]} "
                f"{scale_0[i]} {scale_1[i]} {scale_2[i]} "
                f"{rot_0[i]} {rot_1[i]} {rot_2[i]} {rot_3[i]}\n"
            )