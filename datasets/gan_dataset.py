import pathlib
import numpy as np
import torch
from PIL import Image


class GANDataset(torch.utils.data.Dataset):
    """
    Custom dataset for training.

    Args:
        img_dir (str): Directory containing images.
        transform (callable, optional): Optional transform to be applied.
    """

    def __init__(self, img_dir, transform=None):
        """
        Initialize GANDataset.

        Args:
            img_dir (str): Directory containing images.
            transform (callable, optional): Optional transform to be applied.
        """
        self.transform = transform

        # Collect image paths
        self.img_paths = list(pathlib.Path(img_dir).glob('*'))

        # Check if there are images in the dataset
        assert len(self.img_paths) > 0, "No images found in the dataset."

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Open image using PIL
        image = Image.open(self.img_paths[idx], 'r').convert("L")
        image = np.asarray(image)

        # Apply transformations if specified
        if self.transform is not None:
            transformed_data = self.transform(image=image)
            image = transformed_data['image']

        # Ensure the image has at least one channel (for grayscale)
        if len(image.shape) == 2:
            image = image.reshape(1, *image.shape)

        return image
