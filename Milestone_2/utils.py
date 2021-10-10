import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision


def show_pair(img: torch.Tensor) -> None:
    """Displays pair of before / after images

    Args:
        img: image pair of shape [6, H, W]
    """
    split_img = list(torch.split(img, 3))
    np_img = torchvision.utils.make_grid(split_img, nrow=2).numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)), interpolation="nearest")


def generate_submission(preds: torch.Tensor, path: str = "submission.csv") -> None:
    """Generate submission csv file

    Args:
        preds: Predicted labels
        path: path to CSV file to write to
    """
    df = pd.DataFrame(preds.tolist(), columns=["target"])
    df.to_csv(path, index=False)
    print(f"Submission saved to {path}")
