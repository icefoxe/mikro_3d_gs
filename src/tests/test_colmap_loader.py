import torch
import sys
sys.path.append("src")

from mikro3dgs.colmap_loader import ColmapLoader



device = torch.device("cpu")

loader = ColmapLoader(model_dir="data/helga_test_1", device=device)


loader.load_all()

print("Loaded cameras:", len(loader.cameras_models))
print("Loaded images:", len(loader.images))
print("Loaded 3D points:", len(loader.points3D))

xyz, rgb = loader.get_points_xyz_rgb()
print("xyz shape:", xyz.shape)
print("rgb shape:", rgb.shape)

img = loader.get_first_image()
print("First image name:", img.name)

camera = loader.build_camera(img)
print("Camera K:\n", camera.K)
print("Camera image size:", camera.image_size)


