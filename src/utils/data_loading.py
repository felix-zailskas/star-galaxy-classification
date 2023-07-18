import os

import albumentations as album
import scipy.io
import torch


def get_training_augmentation():
    train_transform = [
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)


class BuildingsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_dir,
        class_rgb_values=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.file_paths = [
            os.path.join(images_dir, image_id)
            for image_id in sorted(os.listdir(images_dir))
        ]
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        file_path = self.file_paths[i]
        data = scipy.io.loadmat(file_path)

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
        return len(self.file_paths)
