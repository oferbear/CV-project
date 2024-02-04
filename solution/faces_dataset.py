"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform
        self.type = {'real': 0, 'fake': 1}

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        if index >= self.__len__():
            raise IndexError("Exceeded dataset size")
        if index < len(self.real_image_names):
            current_used_data = self.real_image_names
            label = "real"
        else:
            current_used_data = self.fake_image_names
            index %= len(self.real_image_names)
            label = "fake"
        with Image.open(os.path.join(self.root_path, label, current_used_data[index])) as sample:
            if self.transform:
                sample = self.transform(sample)
            return sample, self.type[label]

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.real_image_names) + len(self.fake_image_names)
