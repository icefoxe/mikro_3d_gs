from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

def save_image_tensor(image_tensor: torch.Tensor, output_path:str | Path) -> None:
   
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_np = image_tensor.detach().cpu().numpy()
    plt.imsave(output_path, image_np)



def load_image_as_tensor(image_path: str | Path, device: torch.device) -> torch.Tensor:
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).to(device)
    return image_tensor