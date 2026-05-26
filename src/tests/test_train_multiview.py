from pathlib import Path
import sys

sys.path.append("src")

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from mikro3dgs.colmap_loader import ColmapLoader
from mikro3dgs.gaussians import GaussianModel
from mikro3dgs.utils.utils import load_image_as_tensor, save_image_tensor
from mikro3dgs.renderer import GaussianRenderer
from mikro3dgs.utils.model_io import save_gaussian_model_pt, save_gaussian_splat_ply
from mikro3dgs.losses import combined_loss



def rebuild_optimizer(model, position_lr=1.6e-4):
    return torch.optim.Adam(
        [
            {"params": [model.means_3d], "lr": position_lr, "name": "means"},
            {"params": [model.colors_raw], "lr": 2.5e-3, "name": "colors"},
            {"params": [model.opacities_raw], "lr": 2.5e-2, "name": "opacities"},
            {"params": [model.scales_raw], "lr": 4.0e-3, "name": "scales"},
            {"params": [model.rotations_raw], "lr": 1.0e-3, "name": "rotations"},
        ],
        eps=1e-15,
    )


def update_learning_rate(optimizer, step, num_iterations):
    t = min(1.0, step / max(1, num_iterations))
    xyz_lr = 1.6e-4 * (0.01**t)
    for group in optimizer.param_groups:
        if group.get("name") == "means":
            group["lr"] = xyz_lr


@torch.no_grad()
def estimate_initial_scales(
    xyz: torch.Tensor, k: int = 4, min_scale=0.0015, max_scale=0.06
) -> torch.Tensor:
    n = xyz.shape[0]
    chunk = 4096
    kth = []
    for s in range(0, n, chunk):
        d = torch.cdist(xyz[s : s + chunk], xyz)
        vals, _ = torch.topk(d, k=min(k + 1, n), largest=False)
        kth.append(vals[:, -1])
    base = torch.cat(kth).clamp(min_scale, max_scale)
    scales = torch.stack([base * 1.4, base * 0.7, base * 0.7], dim=1)
    return scales


@torch.no_grad()
def make_split_children(
    means, colors, opacities, scales, rotations, split_mask, children=2
):
    m = means[split_mask]
    c = colors[split_mask]
    o = opacities[split_mask]
    s = scales[split_mask]
    r = rotations[split_mask]
    if m.numel() == 0:
        return None

    noise = torch.randn(
        (m.shape[0] * children, 3), device=m.device
    ) * s.repeat_interleave(children, dim=0)
    child_means = m.repeat_interleave(children, dim=0) + 0.5 * noise
    child_scales = (s.repeat_interleave(children, dim=0) / (0.8 * children)).clamp_min(
        1e-4
    )
    child_opacities = (o.repeat_interleave(children, dim=0) / children).clamp(
        1e-4, 0.95
    )
    child_colors = c.repeat_interleave(children, dim=0)
    child_rotations = r.repeat_interleave(children, dim=0)
    return child_means, child_colors, child_opacities, child_scales, child_rotations


@torch.no_grad()
def prune_and_densify(
    model,
    optimizer,
    opacity_min=0.008,
    scale_split_thresh=0.018,
    grad_thresh=2e-5,
    max_gaussians=250000,
):
    params = model.get_parameters()
    means, colors = params.means_3d.detach(), params.colors.detach()
    opacities, scales, rotations = (
        params.opacities.detach(),
        params.scales_3d.detach(),
        params.rotations.detach(),
    )

    keep = (
        (opacities > opacity_min)
        & torch.isfinite(means).all(dim=1)
        & torch.isfinite(scales).all(dim=1)
    )
    means, colors, opacities, scales, rotations = (
        means[keep],
        colors[keep],
        opacities[keep],
        scales[keep],
        rotations[keep],
    )

    grad = model.means_3d.grad
    if grad is not None:
        grad_norm = grad.detach().norm(dim=1)[keep]
    else:
        grad_norm = torch.zeros((means.shape[0],), device=means.device)

    split_mask = (
        (scales.max(dim=1).values > scale_split_thresh) | (grad_norm > grad_thresh)
    ) & (opacities > opacity_min * 2.0)
    capacity = max_gaussians - means.shape[0]
    if capacity > 0 and split_mask.any():
        candidate_idx = torch.where(split_mask)[0]
        max_parents = min(candidate_idx.numel(), capacity // 2)
        if max_parents > 0:
            score = (
                grad_norm[candidate_idx] + 0.1 * scales[candidate_idx].max(dim=1).values
            )
            chosen = candidate_idx[
                torch.topk(score, k=max_parents, largest=True).indices
            ]
            mask = torch.zeros_like(split_mask)
            mask[chosen] = True
            children = make_split_children(
                means, colors, opacities, scales, rotations, mask, children=2
            )
            if children is not None:
                cm, cc, co, cs, cr = children
                means = torch.cat([means, cm], dim=0)
                colors = torch.cat([colors, cc], dim=0)
                opacities = torch.cat([opacities, co], dim=0)
                scales = torch.cat([scales, cs], dim=0)
                rotations = torch.cat([rotations, cr], dim=0)

    model.replace_parameters(means, colors, opacities, scales, rotations)
    return rebuild_optimizer(model)


@torch.no_grad()
def choose_patch(camera, uv, inside_mask, patch_size):
    visible_idx = torch.where(inside_mask)[0]
    if visible_idx.numel() == 0:
        return None
    center = uv[
        visible_idx[
            torch.randint(0, visible_idx.shape[0], (1,), device=uv.device).item()
        ]
    ]
    px = int(center[0].item() - patch_size // 2)
    py = int(center[1].item() - patch_size // 2)
    px = max(0, min(camera.width - patch_size, px))
    py = max(0, min(camera.height - patch_size, py))
    return px, py


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device for training:", device)

    model_dir = Path("data/helga_test_1")
    images_dir = model_dir / "images"
    output_dir = Path("output/train_multiview_9")
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = ColmapLoader(model_dir=model_dir, device=device)
    loader.load_all()
    xyz, rgb = loader.get_points_xyz_rgb()

    colmap_images = list(loader.images.values())
    cameras, image_tensors = [], []
    train_size = (960, 540)  # width, height

    for img in colmap_images:
        cam = loader.build_camera(img)
        old_h, old_w = cam.image_size
        new_w, new_h = train_size
        sx, sy = new_w / old_w, new_h / old_h
        cam.K[0, :] *= sx
        cam.K[1, :] *= sy
        cam.image_size = (new_h, new_w)

        target_path = images_dir / img.name
        target_image = load_image_as_tensor(
            target_path, device=device, size=train_size
        ).clamp(0.0, 1.0)
        cameras.append(cam)
        image_tensors.append(target_image)

    print("Loaded views:", len(cameras))
    print("Training points:", xyz.shape[0])
    print("Camera image size:", cameras[0].image_size)

    init_opacities = torch.full((xyz.shape[0],), 0.08, device=device)
    init_scales_3d = estimate_initial_scales(xyz)

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
        scale_min=0.0008,
        scale_max=0.25,
    ).to(device)

    renderer = GaussianRenderer(device=device, tile_gaussians=192)
    optimizer = rebuild_optimizer(model)

    num_iterations = 1000
    patch_size = 96
    num_patches = 3
    max_patch_points = 1600
    patch_margin = 72
    preview_every = 200
    full_eval_every = 0
    background = torch.tensor(
        [1.0, 1.0, 1.0], device=device
    )
    losses = []
    final_out = None
    eval_view_idx = 0

    for step in tqdm(range(num_iterations), desc="Training multiview"):
        update_learning_rate(optimizer, step, num_iterations)
        optimizer.zero_grad(set_to_none=True)
        params = model.get_parameters()
        loss_total = 0.0
        valid_patch_count = 0

        for _ in range(num_patches):
            view_idx = torch.randint(0, len(cameras), (1,), device=device).item()
            camera = cameras[view_idx]
            target_image = image_tensors[view_idx]

            with torch.no_grad():
                uv, depth, valid_mask = camera.project(params.means_3d)
                inside_mask = camera.in_image_mask(uv, valid_mask)
                picked = choose_patch(camera, uv, inside_mask, patch_size)
                if picked is None:
                    continue
                patch_x, patch_y = picked

                candidate_mask = (
                    valid_mask
                    & (uv[:, 0] >= patch_x - patch_margin)
                    & (uv[:, 0] < patch_x + patch_size + patch_margin)
                    & (uv[:, 1] >= patch_y - patch_margin)
                    & (uv[:, 1] < patch_y + patch_size + patch_margin)
                )
                candidate_idx = torch.where(candidate_mask)[0]
                if candidate_idx.numel() == 0:
                    continue
                if candidate_idx.numel() > max_patch_points:
                    nearest = torch.argsort(depth[candidate_idx])[:max_patch_points]
                    candidate_idx = candidate_idx[nearest]

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
                point_indices=candidate_idx,
                background=background,
                sigma_extent=3.0,
            )

            pred_patch = render_output.image
            target_patch = target_image[
                patch_y : patch_y + patch_size, patch_x : patch_x + patch_size
            ]

            mask = 0.25 + 0.75 * render_output.alpha.detach().clamp(0.0, 1.0)
            patch_loss = combined_loss(
                pred=pred_patch,
                target=target_patch,
                mask=mask,
                opacities=params.opacities[candidate_idx],
                scales=params.scales_3d[candidate_idx],
            )
            loss_total = loss_total + patch_loss
            valid_patch_count += 1

        if valid_patch_count == 0:
            continue

        loss = loss_total / valid_patch_count
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step > 700 and step % 700 == 0:
            before = model.means_3d.shape[0]
            optimizer = prune_and_densify(
                model,
                optimizer,
                opacity_min=0.006,
                scale_split_thresh=0.016,
                max_gaussians=220000,
            )
            after = model.means_3d.shape[0]
            print(f"Prune/densify: {before} -> {after} gaussians")

        if step > 0 and step % 3000 == 0:
            model.reset_opacity(0.03)

        losses.append(float(loss.item()))
        if step % 25 == 0:
            tqdm.write(
                f"Step {step}, Loss: {loss.item():.6f}, Gaussians: {model.means_3d.shape[0]}"
            )


        # if (step > 0 and step % preview_every == 0) or step == num_iterations - 1:
        #     with torch.no_grad():
        #         eval_params = model.get_parameters()
        #         preview_camera = cameras[eval_view_idx]
        #         preview_target = image_tensors[eval_view_idx]
        #         uv, depth, valid_mask = preview_camera.project(eval_params.means_3d)
        #         inside_mask = preview_camera.in_image_mask(uv, valid_mask)
        #         picked = choose_patch(preview_camera, uv, inside_mask, patch_size)
        #         if picked is not None:
        #             px, py = picked
        #             candidate_mask = (
        #                 valid_mask
        #                 & (uv[:, 0] >= px - patch_margin)
        #                 & (uv[:, 0] < px + patch_size + patch_margin)
        #                 & (uv[:, 1] >= py - patch_margin)
        #                 & (uv[:, 1] < py + patch_size + patch_margin)
        #             )
        #             candidate_idx = torch.where(candidate_mask)[0]
        #             if candidate_idx.numel() > max_patch_points:
        #                 nearest = torch.argsort(depth[candidate_idx])[:max_patch_points]
        #                 candidate_idx = candidate_idx[nearest]
        #             preview_out = renderer.render_patch(
        #                 camera=preview_camera,
        #                 means_3d=eval_params.means_3d,
        #                 colors=eval_params.colors,
        #                 opacities=eval_params.opacities,
        #                 scales_3d=eval_params.scales_3d,
        #                 rotations=eval_params.rotations,
        #                 patch_x=px,
        #                 patch_y=py,
        #                 patch_size=patch_size,
        #                 point_indices=candidate_idx,
        #                 background=background,
        #                 sigma_extent=3.0,
        #             )
        #             save_image_tensor(
        #                 preview_out.image, output_dir / f"preview_patch_{step:04d}.png"
        #             )

        if full_eval_every and step > 0 and step % full_eval_every == 0:
            with torch.no_grad():
                eval_params = model.get_parameters()
                final_out = renderer.render(
                    camera=cameras[eval_view_idx],
                    means_3d=eval_params.means_3d,
                    colors=eval_params.colors,
                    opacities=eval_params.opacities,
                    scales_3d=eval_params.scales_3d,
                    rotations=eval_params.rotations,
                    background=background,
                    sigma_extent=3.0,
                )
                save_image_tensor(
                    final_out.image, output_dir / f"render_{step:04d}.png"
                )

    params = model.get_parameters()
    if final_out is None:
        print("Training finished. Rendering final full image; this can take a while.")
        with torch.no_grad():
            final_out = renderer.render(
                cameras[eval_view_idx],
                params.means_3d,
                params.colors,
                params.opacities,
                params.scales_3d,
                background=background,
                rotations=params.rotations,
            )

    save_image_tensor(image_tensors[eval_view_idx], output_dir / "target.png")
    save_image_tensor(final_out.image, output_dir / "final_render.png")
    save_gaussian_splat_ply(
        output_dir / "gaussian_model_final.ply",
        params.means_3d,
        params.colors,
        params.opacities,
        params.scales_3d,
        params.rotations,
    )
    save_gaussian_model_pt(
        output_dir / "gaussian_model_final.pt",
        params.means_3d,
        params.colors,
        params.opacities,
        params.scales_3d,
        params.rotations,
    )

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_tensors[eval_view_idx].detach().cpu().numpy())
    plt.title("Target Image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(final_out.image.detach().cpu().numpy())
    plt.title(f"Predicted Image\nLoss: {losses[-1]:.6f}")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.title("Loss")
    plt.xlabel("Step")
    plt.ylabel("L1 + DSSIM")
    plt.tight_layout()
    plt.savefig(output_dir / "training_summary.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
