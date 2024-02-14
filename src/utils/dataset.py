import os

import scipy.io
import torch


class SDSSDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_dir,
        augmentation=None,
        preprocessing=None,
    ):
        self.data_dicts = []
        for file_path in [
            os.path.join(images_dir, image_id)
            for image_id in sorted(os.listdir(images_dir))
        ]:
            self.data_dicts.append(scipy.io.loadmat(file_path))
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        data = self.data_dicts[i]

        image = data["image"]
        mask = data["seg_map"]

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.data_dicts)
