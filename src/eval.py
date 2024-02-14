import datetime
import os

import torch
from torch.utils.data import DataLoader

from models.UNet import UNet
from utils.dataset import SDSSDataset
from utils.logger import Logger
from utils.model_training import evaluate_model

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = Logger("Testing Logger")  # , f"../logs/training/{current_time}/")

BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_EPOCHS = 2

if __name__ == "__main__":
    # check if dedicated processing hardware is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load the trained model
    model_name = "2024-02-14_19-55-46.pth"
    model_data = torch.load(f"../models/{model_name}")
    model = UNet(
        in_channels=model_data["parameters"]["in_channels"],
        num_classes=model_data["parameters"]["num_classes"],
        depth=model_data["parameters"]["depth"],
    )
    model.load_state_dict(model_data["state_dict"])

    # Load and preprocess the test dataset
    DATA_DIR = "../data/"
    x_test_dir = os.path.join(DATA_DIR, "test/")

    test_dataset = SDSSDataset(x_test_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    logger.info("Testing Model")
    # Run evaluation
    criterion = torch.nn.CrossEntropyLoss()
    avg_loss = evaluate_model(model, criterion, device, test_loader)
    logger.info(f"Average testing loss {avg_loss}")
