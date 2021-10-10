import collections
import os
from typing import Callable, Optional, Tuple

import pandas as pd
import tifffile
import torch
from torch.utils import data as data
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import ToTensor


class PatchPairsDataset(data.Dataset):
    """Loads pairs of aerial image patches from paths provided in a CSV file"""

    def __init__(
        self,
        csv_path: str,
        pairs_folder_path: str,
        transform: Optional[Callable] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.pairs_folder_path = pairs_folder_path
        if "target" in self.df.columns:
            self.has_target = True
            self.targets = torch.tensor(self.df["target"], dtype=torch.float32).reshape(
                -1, 1
            )
        else:
            self.has_target = False
            # Fill targets with -1 if no targets are given
            self.targets = torch.empty(len(self.df), 1, dtype=torch.float32).fill_(-1.0)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image pair
        pair_path = os.path.join(self.pairs_folder_path, self.df["pair"][idx])
        # Image is of shape [6, H, W] (first 3 channels for before, last 3 for after)
        img = ToTensor()(tifffile.imread(pair_path))

        # Load target (if there is one)
        target = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def split_dataset(
    dataset: Dataset, split: float, seed: int = 1234
) -> Tuple[Subset, Subset]:
    """Splits dataset into a train / val set based on a split value and seed
    Args:
        dataset: dataset to split
        split: The proportion of the dataset to include in the validation split,
            must be between 0 and 1.
        seed: Seed used to generate the split
    Returns:
        Subsets of the input dataset
    """
    # Verify that the dataset is Sized
    if not isinstance(dataset, collections.abc.Sized):
        raise ValueError("Dataset is not Sized!")

    if not (0 <= split <= 1):
        raise ValueError(f"Split value must be between 0 and 1. Value: {split}")

    val_length = int(len(dataset) * split)
    train_length = len(dataset) - val_length
    splits = random_split(
        dataset,
        [train_length, val_length],
        generator=torch.Generator().manual_seed(seed),
    )
    return splits
