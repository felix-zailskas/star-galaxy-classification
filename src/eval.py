import logging
import os

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from models.UNet import UNet, UNetWithGradCAM
from utils.dataset import BuildingsDataset


def evaluate(model, criterion, device, test_loader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.permute(0, 3, 1, 2).float().to(device)
            labels = labels.permute(0, 3, 1, 2).float().to(device)

            output_tuple = model(inputs)
            outputs = output_tuple[0]
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            # Log metrics to W&B
            wandb.log({"Test Loss": loss.item()})

    average_loss = total_loss / len(test_loader.dataset)
    return average_loss


def eval_main(model, test_loader):
    # Set device
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Move model to device
    model = model.to(device)

    # Initialize W&B
    wandb.init(project="star-galaxy-classification")

    # Evaluate the model
    test_loss = evaluate(model, criterion, device, test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    # Log final metrics to W&B
    wandb.log({"Final Test Loss": test_loss})

    # Finish the W&B run
    wandb.finish()


if __name__ == "__main__":
    # Load the trained model
    model = UNetWithGradCAM(UNet(5, 3))
    model.load_state_dict(torch.load("model.pth"))

    # Load and preprocess the test dataset
    DATA_DIR = "../data/"
    x_test_dir = os.path.join(DATA_DIR, "test/")

    test_dataset = BuildingsDataset(x_test_dir)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Run evaluation
    eval_main(model, test_loader)
