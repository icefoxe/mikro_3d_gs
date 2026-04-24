from pathlib import Path
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

import sys
sys.path.append("src")

from mikro3dgs.colmap_loader import ColmapLoader
from mikro3dgs.gaussians import GaussianModel
from mikro3dgs.losses import mse_loss
from mikro3dgs.utils import load_image_as_tensor, save_image_tensor
from mikro3dgs.renderer import GaussianRenderer


def main() -> None:
    # na razie cuda sie wypierdala bo out of memory
    device = torch.device("cpu")
    print("Using device for training:", device)

    model_dir = Path("data/helga_test_1")
    images_dir = model_dir / "images"
    output_dir = Path("output/train_single_view")
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = ColmapLoader(model_dir = model_dir, device=device)
    loader.load_all()

    xyz, rgb = loader.get_points_xyz_rgb()

    max_points = 3000
    perm = torch.randperm(xyz.shape[0])[:max_points]
    xyz = xyz[perm]
    rgb = rgb[perm]
    
    colmap_img = loader.get_first_image()
    camera = loader.build_camera(colmap_img)

    print("Training on image:", colmap_img.name)
    print("Points:", xyz.shape[0])

    target_path = images_dir / colmap_img.name
    target_image = load_image_as_tensor(target_path, device = device, size = (480, 270))
    camera.image_size = (270, 480)
    camera.K[0, :] *= 480 /  1920
    camera.K[1, :] *= 270 / 1080


    print("Target image shape:", target_image.shape)
    print("Camera image size:", camera.image_size)

    if target_image.shape[0] != camera.image_size[0] or target_image.shape[1] != camera.image_size[1]:
        raise ValueError(f"Image side mismatch: target image has shape {target_image.shape}, but camera expects {camera.image_size}")
    
    init_opacities = torch.ones((xyz.shape[0],), device=device) * 0.2
    init_scales = torch.ones((xyz.shape[0],), device=device) * 0.003

    model = GaussianModel(
        means_3d=xyz,
        colors=rgb,
        opacities=init_opacities,
        base_scales=init_scales,
        learn_means=False,
        learn_colors=True,
        learn_opacities=True,
        learn_scales=False,
    ).to(device)

    renderer = GaussianRenderer(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)



    num_iterations = 200
    losses = []

    for step in tqdm(range(num_iterations), desc="Training single view"):
        optimizer.zero_grad()

        params = model.get_parameters()
        render_output = renderer.render(
            means_3d=params.means_3d,
            colors=params.colors,
            opacities=params.opacities,
            base_scales=params.base_scales,
            camera=camera,
        )

        pred_image = render_output.image
        loss = mse_loss(render_output.image, target_image)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 20 == 0 or step == num_iterations - 1:
            print(f"Step {step}, Loss: {loss.item():.6f}")
            save_image_tensor(render_output.image, output_dir / f"render_{step:04d}.png")

    save_image_tensor(target_image, output_dir / "target.png")
    save_image_tensor(pred_image, output_dir / "final_render.png")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(target_image.detach().cpu().numpy())
    plt.title("Target Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_image.detach().cpu().numpy())
    plt.title(f"Predicted Image\nStep {step}, Loss: {loss.item():.6f}")
    plt.axis("off")

    plt.subplot(1, 2, 3)
    plt.plot(losses)
    plt.title("Loss")
    plt.xlabel("Step")
    plt.ylabel("MSE")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":    
    main()
