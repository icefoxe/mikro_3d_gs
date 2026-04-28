from pathlib import Path
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

import sys
sys.path.append("src")

from mikro3dgs.colmap_loader import ColmapLoader
from mikro3dgs.gaussians import GaussianModel
from mikro3dgs.utils.utils import load_image_as_tensor, save_image_tensor
from mikro3dgs.renderer import GaussianRenderer
from mikro3dgs.utils.model_io import save_gaussian_model_pt, save_gaussian_splat_ply



def main() -> None:
    # na razie cuda sie wypierdala bo out of memory
    device = torch.device("cpu")
    print("Using device for training:", device)

    model_dir = Path("data/helga_test_1")
    images_dir = model_dir / "images"
    output_dir = Path("output/train_multiview")
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = ColmapLoader(model_dir = model_dir, device=device)
    loader.load_all()

    xyz, rgb = loader.get_points_xyz_rgb()

    colmap_images = list(loader.images.values())
    # colmap_images = colmap_images[:100] # ograniczenie zdjęć

    cameras = []
    image_tensors = []

    for img in colmap_images:
        cam = loader.build_camera(img)

        cam.image_size = (270, 480)
        cam.K[0, :] *= 480 /  1920
        cam.K[1, :] *= 270 / 1080

        target_path = images_dir / img.name
        target_image = load_image_as_tensor(target_path, device = device, size = (480, 270))

        cameras.append(cam)
        image_tensors.append(target_image)
    
    print("Loaded views:", len(cameras))



    max_points = 12000
    if xyz.shape[0] > max_points:
        perm = torch.randperm(xyz.shape[0], device=device)[:max_points]
        xyz = xyz[perm]
        rgb = rgb[perm]

    print("Training points:", xyz.shape[0])
    print("Camera image size:", cameras[0].image_size)

    if target_image.shape[0] != cameras[0].image_size[0] or target_image.shape[1] != cameras[0].image_size[1]:
        raise ValueError(f"Image side mismatch: target image has shape {target_image.shape}, but camera expects {cameras[0].image_size}")
    
    init_opacities = torch.ones((xyz.shape[0],), device=device) * 0.9

    with torch.no_grad():
        uv, depth, valid_mask = cameras[0].project(xyz)
        fx = cameras[0].K[0, 0]

        target_sigma_px = 4.0
        init_scales = target_sigma_px * depth / fx
        init_scales = torch.clamp(init_scales, min=0.001, max=0.03)
        init_scales_3d = torch.stack(
            [
                init_scales * 1.0,
                init_scales * 0.3,
                init_scales * 0.1,
            ],
            dim=1,
        )

        

    model = GaussianModel(
        means_3d=xyz,
        colors=rgb,
        opacities=init_opacities,
        scales_3d=init_scales_3d,
        learn_means=True,
        learn_colors=True,
        learn_opacities=True,
        learn_scales=True,
        learn_rotations=True,
    ).to(device)

    renderer = GaussianRenderer(device=device)

    optimizer = torch.optim.Adam([
        {"params": model.colors_raw, "lr": 1e-2},
        {"params": model.opacities_raw, "lr": 5e-3},
        {"params": model.scales_raw, "lr": 2e-3},
        {"params": model.rotations_raw, "lr": 1e-3},
    ])



    num_iterations = 5000
    patch_size = 96
    num_patches = 2
    losses = []

    final_out = None
    eval_view_idx = 0

    for step in tqdm(range(num_iterations), desc="Training multiview"):
        optimizer.zero_grad()
        params = model.get_parameters()

        loss_total = 0.0
        valid_patch_count = 0

        for _ in range(num_patches):
            view_idx = torch.randint(0, len(cameras), (1,)).item()
            camera = cameras[view_idx]
            target_image = image_tensors[view_idx]

            with torch.no_grad():
                uv, depth, valid_mask = camera.project(params.means_3d)
                inside_mask = camera.in_image_mask(uv, valid_mask)
                visible_idx = torch.where(inside_mask)[0]

            if visible_idx.numel() == 0:
                continue

            j = torch.randint(0, visible_idx.shape[0], (1,)).item()
            center = uv[visible_idx[j]]

            patch_x = int(center[0].item() - patch_size // 2)
            patch_y = int(center[1].item() - patch_size // 2)

            patch_x = max(0, min(camera.width - patch_size, patch_x))
            patch_y = max(0, min(camera.height - patch_size, patch_y))

            render_output = renderer.render_patch(
                camera=camera,
                means_3d=params.means_3d,
                colors=params.colors,
                opacities=params.opacities,
                scales_3d=params.scales_3d,
                rotations=params.rotations,
                patch_x=patch_x,
                patch_y=patch_y,
                patch_size=patch_size,
            )

            pred_patch = render_output.image
            target_patch = target_image[
                patch_y:patch_y + patch_size,
                patch_x:patch_x + patch_size,
            ]

            mask = 0.5 + 0.5 * (render_output.alpha > 1e-4).float()

            l1 = (torch.abs(pred_patch - target_patch) * mask).sum() / (
                mask.sum() * 3 + 1e-8
            )
            mse = (((pred_patch - target_patch) ** 2) * mask).sum() / (
                mask.sum() * 3 + 1e-8
            )

            patch_loss = 0.8 * l1 + 0.2 * mse
            loss_total = loss_total + patch_loss
            valid_patch_count += 1

        if valid_patch_count == 0:
            continue

        loss = loss_total / valid_patch_count

        # lekka regularyzacja, żeby skale nie uciekały do zera
        loss = loss + 0.001 * (1.0 / (params.scales_3d.mean() + 1e-6))

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # if step % 10 == 0:
        #     print("scale mean:", params.scales_3d.mean().item())
        #     print("scale min/max:", params.scales_3d.min().item(), params.scales_3d.max().item())

        if step % 100 == 0 or step == num_iterations - 1:
            eval_camera = cameras[eval_view_idx]
            eval_target = image_tensors[eval_view_idx]
            eval_params = model.get_parameters()

            final_out = renderer.render(
                camera=eval_camera,
                means_3d=eval_params.means_3d,
                colors=eval_params.colors,
                opacities=eval_params.opacities,
                scales_3d=eval_params.scales_3d,
                rotations=eval_params.rotations,
            )
            print(f"Step {step}, Loss: {loss.item():.6f}")
            save_image_tensor(final_out.image, output_dir / f"render_{step:04d}.png")
    
    params = model.get_parameters()
    
    if final_out is None:
        final_out = renderer.render(
                camera=cameras[eval_view_idx],
                means_3d=params.means_3d,
                colors=params.colors,
                opacities=params.opacities,
                scales_3d=params.scales_3d,
                rotations=params.rotations,
            )

    save_image_tensor(target_image, output_dir / "target.png")
    save_image_tensor(final_out.image, output_dir / "final_render.png")
    

    

    save_gaussian_splat_ply(
        output_dir / "gaussian_model_final.ply",
        means_3d=params.means_3d,
        colors=params.colors,
        opacities=params.opacities,
        scales_3d=params.scales_3d,
        rotations=params.rotations,
    )

    save_gaussian_model_pt(
        output_dir / "gaussian_model_final.pt",
        means_3d=params.means_3d,
        colors=params.colors,
        opacities=params.opacities,
        scales_3d=params.scales_3d,
        rotations=params.rotations,
    )

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_tensors[eval_view_idx].detach().cpu().numpy())
    plt.title("Target Image 0")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(final_out.image.detach().cpu().numpy())
    plt.title(f"Predicted Image 0 \nStep {step}, Loss: {loss.item():.6f}")
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
