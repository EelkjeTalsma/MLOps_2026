from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    """
    PatchCamelyon (PCAM) Dataset reader for H5 format.
    """

    def __init__(self, x_path: str, y_path: str, transform: Optional[Callable] = None, filter_data: bool = False):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform
        self.filter_data = filter_data

        # TODO: Initialize dataset
        # 1. Check if files exist
        # 2. Open h5 files in read mode
        if not self.x_path.exists() or not self.y_path.exists():
            raise FileNotFoundError
        self.x_h5 = h5py.File(self.x_path, "r")
        self.y_h5 = h5py.File(self.y_path, "r")
        self.images = self.x_h5["x"]
        self.labels = self.y_h5["y"]
        if filter_data:
            self.indices = [
                i for i in range(len(self.labels))
                if 0 < np.mean(self.images[i]) < 255
            ]
        else:
            self.indices = list(range(len(self.labels)))

    def __len__(self) -> int:
        # TODO: Return length of dataset
        # The dataloader will know hence how many batches to create
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement data retrieval
        # 1. Read data at idx
        # 2. Convert to uint8 (for PIL compatibility if using transforms)
        # 3. Apply transforms if they exist
        # 4. Return tensor image and label (as long)
        idx = self.indices[idx]
        img = self.images[idx]
        label = self.labels[idx]
        img = np.clip(img, 0, 255)
        # img = np.clip(img / 255.0, 0.0, 1.0).astype(np.float32)
        # img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        # label_tensor = torch.tensor(label, dtype=torch.long)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.long).squeeze()
        if self.transform is not None:
            # img_tensor = self.transform(img_tensor)
            img = self.transform(img)
        return img, label
