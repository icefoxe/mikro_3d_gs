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

    
    colmap_img = loader.get_first_image()
    camera = loader.build_camera(colmap_img)

    camera.image_size = (270, 480)
    camera.K[0, :] *= 480 /  1920
    camera.K[1, :] *= 270 / 1080

    with torch.no_grad():
        uv, depth, valid_mask = camera.project(xyz)
        inside_mask = camera.in_image_mask(uv, valid_mask)
        visible_idx = torch.where(inside_mask)[0]
        visible_uv = uv[visible_idx]

    
    max_points = 10000
    perm = visible_idx[torch.randperm(len(visible_idx))[:max_points]]
    xyz = xyz[perm]
    rgb = rgb[perm]


    print("Training on image:", colmap_img.name)
    print("Points:", xyz.shape[0])

    target_path = images_dir / colmap_img.name
    target_image = load_image_as_tensor(target_path, device = device, size = (480, 270))



    print("Target image shape:", target_image.shape)
    print("Camera image size:", camera.image_size)

    if target_image.shape[0] != camera.image_size[0] or target_image.shape[1] != camera.image_size[1]:
        raise ValueError(f"Image side mismatch: target image has shape {target_image.shape}, but camera expects {camera.image_size}")
    
    init_opacities = torch.ones((xyz.shape[0],), device=device) * 0.9
    with torch.no_grad():
        uv, depth, valid_mask = camera.project(xyz)
        fx = camera.K[0, 0]

        target_sigma_px = 4.0
        init_scales = target_sigma_px * depth / fx
        init_scales = torch.clamp(init_scales, min=0.001, max=0.03)
        

    model = GaussianModel(
        means_3d=xyz,
        colors=rgb,
        opacities=init_opacities,
        base_scales=init_scales,
        learn_means=True,
        learn_colors=True,
        learn_opacities=True,
        learn_scales=True,
    ).to(device)

    renderer = GaussianRenderer(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)



    num_iterations = 300
    losses = []

    for step in tqdm(range(num_iterations), desc="Training single view"):
        optimizer.zero_grad()

        params = model.get_parameters()

        with torch.no_grad():
            _, depth, _ = camera.project(params.means_3d)
            sigma_2d = renderer.compute_2d_radius(
                params.base_scales,
                depth,
                focal_length=camera.K[0, 0].item(),
            )

        # print("base_scales:", params.base_scales.min().item(), params.base_scales.mean().item(), params.base_scales.max().item())
        # print("sigma_2d px:", sigma_2d.min().item(), sigma_2d.mean().item(), sigma_2d.max().item())
                
        patch_size = 64

        j = torch.randint(0, visible_uv.shape[0], (1,)).item()
        center = visible_uv[j]

        patch_x = int(center[0].item() - patch_size // 2)
        patch_y = int(center[1].item() - patch_size // 2)

        patch_x = max(0, min(camera.width - patch_size, patch_x))
        patch_y = max(0, min(camera.height - patch_size, patch_y))

        # patch_x = torch.randint(low=0, high=camera.width - patch_size, size=(1,)).item()
        # patch_y = torch.randint(low=0, high=camera.height - patch_size, size=(1,)).item()

        render_output = renderer.render_patch(
            means_3d=params.means_3d,
            colors=params.colors,
            opacities=params.opacities,
            base_scales=params.base_scales,
            camera=camera,
            patch_x=patch_x,
            patch_y=patch_y,
            patch_size=patch_size,
        )

        pred_patch = render_output.image
        target_patch = target_image[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size]

        mask = (render_output.alpha > 1e-4).float()

        num_patches = 2
        loss_total = 0.0
        for _ in range(num_patches):
            patch_x = torch.randint(low=0, high=camera.width - patch_size, size=(1,)).item()
            patch_y = torch.randint(low=0, high=camera.height - patch_size, size=(1,)).item()

            render_output = renderer.render_patch(
                means_3d=params.means_3d,
                colors=params.colors,
                opacities=params.opacities,
                base_scales=params.base_scales,
                camera=camera,
                patch_x=patch_x,
                patch_y=patch_y,
                patch_size=patch_size,
            )

            pred_patch = render_output.image
            target_patch = target_image[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size]

            mask = (render_output.alpha > 1e-4).float()

            patch_loss = ((pred_patch - target_patch) ** 2 * mask).sum() / (mask.sum() * 3 + 1e-8)
            loss_total += patch_loss

        # loss = ((pred_patch - target_patch) ** 2 * mask).sum() / (mask.sum() *3 + 1e-8)
        loss = loss_total / num_patches
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 20 == 0 or step == num_iterations - 1:
            final_out = renderer.render(
                camera=camera,
                means_3d=params.means_3d,
                colors=params.colors,
                opacities=params.opacities,
                base_scales=params.base_scales,
            )
            print(f"Step {step}, Loss: {loss.item():.6f}")
            save_image_tensor(final_out.image, output_dir / f"render_{step:04d}.png")

    save_image_tensor(target_image, output_dir / "target.png")
    save_image_tensor(final_out.image, output_dir / "final_render.png")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(target_image.detach().cpu().numpy())
    plt.title("Target Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(final_out.image.detach().cpu().numpy())
    plt.title(f"Predicted Image\nStep {step}, Loss: {loss.item():.6f}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.title("Loss")
    plt.xlabel("Step")
    plt.ylabel("MSE")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":    
    main()
