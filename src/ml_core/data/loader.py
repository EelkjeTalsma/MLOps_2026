from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
from .pcam import PCAMDataset


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])
	
    # TODO: Define Transforms
    # train_transform = ...
    # val_transform = ...

    # TODO: Define Paths for X and Y (train and val)
    
    # TODO: Instantiate PCAMDataset for train and val

    # TODO: Create DataLoaders
    # train_loader = ...
    # val_loader = ...
    train_transform = transforms.Compose([])
    val_transform = transforms.Compose([])

    train_x = base_path / "camelyonpatch_level_2_split_train_x.h5"
    train_y = base_path / "camelyonpatch_level_2_split_train_y.h5"
    val_x = base_path / "camelyonpatch_level_2_split_valid_x.h5"
    val_y = base_path / "camelyonpatch_level_2_split_valid_y.h5"

    train_dataset = PCAMDataset(x_path=str(train_x), y_path=str(train_y), transform=train_transform)
    val_dataset = PCAMDataset(x_path=str(val_x), y_path=str(val_y), transform=val_transform)

    train_labels = np.array([train_dataset[i][1].item() for i in range(len(train_dataset))])
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    train_sampler = WeightedRandomSampler(weights=sample_weights,num_samples=len(sample_weights),replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get("batch_size", 32),
        sampler=train_sampler,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=False
    )

    return train_loader, val_loader
