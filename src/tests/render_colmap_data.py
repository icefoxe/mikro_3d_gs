import matplotlib.pyplot as plt
import torch

from pathlib import Path

import sys
sys.path.append("src")

from mikro3dgs.colmap_loader import ColmapLoader
from mikro3dgs.utils import save_image_tensor
from mikro3dgs.renderer import GaussianRenderer



def main() -> None:
    device = torch.device("cpu")

    model_dir = Path("data/helga_test_1")

    loader = ColmapLoader(model_dir=model_dir, device=device)
    loader.load_all()

    xyz, rgb = loader.get_points_xyz_rgb()
    first_image = loader.get_first_image()
    camera = loader.build_camera(first_image)

    print("Using camera from image:", first_image.name)
    print("Points:", xyz.shape[0])

    # Na początek wszystkie puntky dostają:
    # kolor z Colmap, opacity 1.0, base_scale 0.05

    opacities = torch.ones(xyz.shape[0], device=device)
    base_scales = torch.full((xyz.shape[0],), 0.05, device=device)

    renderer = GaussianRenderer(device=device)
    out = renderer.render(
        camera=camera,
        means_3d=xyz,
        colors=rgb,
        opacities=opacities,
        base_scales=base_scales,
    )

    print("Valid points:", out.valid_mask.sum().item())
    print("Points in image:", out.inside_mask.sum().item())

    output_path = Path("output/test_colmap_render.png")
    save_image_tensor(out.image.cpu(), output_path)
    print(f"Rendered image saved to {output_path}")

    plt.figure(figsize=(10, 6))
    plt.imshow(out.image.cpu().numpy())
    plt.title("Rendered COLMAP Points")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()