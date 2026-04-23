import matplotlib.pyplot as plt
import torch
import sys
sys.path.append("src")

from mikro3dgs.camera import Camera
from mikro3dgs.renderer import GaussianRenderer

device = torch.device("cpu")

K = torch.tensor([
    [300.0,   0.0, 128.0],
    [  0.0, 300.0, 128.0],
    [  0.0,   0.0,   1.0],
], device=device)

R = torch.eye(3, device=device)
t = torch.zeros(3, device=device)

camera = Camera(
    K=K,
    R=R,
    t=t,
    image_size=(256, 256),
    device=device,
)

means_3d = torch.tensor([
    [0.0, 0.0, 2.0],
    [0.3, 0.0, 2.0],
    [-0.3, 0.0, 2.0],
    [0.0, 0.3, 2.0],
], device=device)

colors = torch.tensor([
    [1.0, 0.0, 0.0],   # czerwony
    [0.0, 1.0, 0.0],   # zielony
    [0.0, 0.0, 1.0],   # niebieski
    [1.0, 1.0, 0.0],   # żółty
], device=device)

opacities = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
base_scales = torch.tensor([0.05, 0.05, 0.05, 0.05], device=device)

renderer = GaussianRenderer(device=device)
out = renderer.render(
    camera=camera,
    means_3d=means_3d,
    colors=colors,
    opacities=opacities,
    base_scales=base_scales,
)

plt.figure(figsize=(6, 6))
plt.imshow(out.image.cpu().numpy())
plt.title("Rendered Gaussian Splats")
plt.axis("off")
plt.show()