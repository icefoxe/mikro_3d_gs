from pathlib import Path

import matplotlib.pyplot as plt
import torch

def save_image_tensor(image_tensor: torch.Tensor, output_path:str | Path) -> None:
   
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_np = image_tensor.detach().cpu().numpy()
    plt.imsave(output_path, image_np)