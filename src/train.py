import logging
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
from eval import evaluate
from model import UNet, UNetWithGradCAM
from utils.data_loading import BuildingsDataset


def train():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device {device}")

    # Define hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Create an instance of the UNet model and move it to the GPU
    model = UNetWithGradCAM(UNet(5, 3)).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load and preprocess your dataset
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Initialize W&B
    wandb.init(
        project="star-galaxy-classification",
        config={"batch_size": batch_size, "learning_rate": learning_rate},
    )

    # Log the model architecture
    wandb.watch(model)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Set the model to training mode
        model.train()

        # Iterate over batches of data
        for step, (inputs, labels) in enumerate(train_loader):
            # Move inputs and labels to the GPU
            inputs = inputs.permute(0, 3, 1, 2).float().to(device)
            labels = labels.permute(0, 3, 1, 2).float().to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output_tuple = model(inputs)
            outputs = output_tuple[0]
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track the average loss
            running_loss += loss.item() * inputs.size(0)

            # Log metrics to W&B
            wandb.log({"step": step, "loss": loss.item()})

        # Set the model to evaluation mode
        model.eval()

        # Evaluate on the validation set
        val_loss = evaluate(model, criterion, device, valid_loader)

        # Print epoch statistics
        epoch_loss = running_loss / len(train_dataset)
        print(
            f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}"
        )

        # Log epoch metrics to W&B
        wandb.log({"epoch": epoch, "epoch_loss": epoch_loss, "val_loss": val_loss})

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")

    # Save the model in W&B
    wandb.save("model.pth")

    # Finish the W&B run
    wandb.finish()


if __name__ == "__main__":
    DATA_DIR = "../data/"

    x_train_dir = os.path.join(DATA_DIR, "train/")
    x_valid_dir = os.path.join(DATA_DIR, "validation/")

    # Get train and val dataset instances
    train_dataset = BuildingsDataset(x_train_dir)
    valid_dataset = BuildingsDataset(x_valid_dir)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    train()
