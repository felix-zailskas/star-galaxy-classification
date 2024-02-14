import datetime
import os

import torch
from torch.utils.data import DataLoader

from models.UNet import UNet
from utils.augmentations import get_training_augmentation
from utils.dataset import SDSSDataset
from utils.logger import Logger
from utils.model_training import train_model

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = Logger("Training Logger")  # , f"../logs/training/{current_time}/")

BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_EPOCHS = 2
LEARNING_RATE = 0.01
UNET_DEPTH = 1
IN_CHANNELS = 5
NUM_CLASSES = 3

if __name__ == "__main__":
    DATA_DIR = "../data/"
    # check if dedicated processing hardware is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    x_train_dir = os.path.join(DATA_DIR, "train/")
    x_valid_dir = os.path.join(DATA_DIR, "validation/")

    # Get train and val dataset instances
    train_dataset = SDSSDataset(x_train_dir, augmentation=get_training_augmentation())
    valid_dataset = SDSSDataset(x_valid_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, depth=UNET_DEPTH)

    train_model(
        model=model,
        device=device,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
    )

    save_obj = {
        "state_dict": model.state_dict(),
        "parameters": {
            "depth": UNET_DEPTH,
            "in_channels": IN_CHANNELS,
            "num_classes": NUM_CLASSES,
        },
    }
    torch.save(save_obj, f"../models/{current_time}.pth")
