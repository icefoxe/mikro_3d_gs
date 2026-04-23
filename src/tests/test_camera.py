import torch
import sys
sys.path.append("src")

from mikro3dgs.camera import Camera

device = torch.device("cpu")

K = torch.tensor([
    [500.0,   0.0, 128.0],
    [  0.0, 500.0, 128.0],
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


points_world = torch.tensor([
    [0.0, 0.0, 2.0],
    [0.5, 0.0, 2.0],
    [0.0, 0.5, 2.0],
    [0.0, 0.0, -1.0],  # za kamerą
], device=device)

uv, depth, valid_mask = camera.project(points_world)
inside_mask = camera.in_image_mask(uv, valid_mask)

print("uv:\n", uv)
print("depth:\n", depth)
print("valid_mask:\n", valid_mask)
print("inside_mask:\n", inside_mask)