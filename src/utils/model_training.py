import torch
from tqdm import tqdm

from utils.logger import Logger


def evaluate_model(
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
):
    model.to(device=device)
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader, desc="Evaluating model")):
            in_image, gt_masks = batch
            in_image = in_image.permute(0, 3, 1, 2).float().to(device)
            gt_masks = gt_masks.permute(0, 3, 1, 2).float().to(device)

            out_masks = model(in_image)

            loss = criterion(out_masks, gt_masks)
            total_loss += loss.item() * in_image.size(0)

    average_loss = total_loss / len(data_loader.dataset)
    return average_loss


def train_model(
    model: torch.nn.Module,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    logger: Logger,
):
    logger.info(f"Training on device: {device}")
    model.to(device=device)

    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")

        # training
        model.train()
        running_loss = 0
        for idx, batch in enumerate(tqdm(train_loader, desc="Training model")):
            in_image, gt_masks = batch
            in_image = in_image.permute(0, 3, 1, 2).float().to(device)
            gt_masks = gt_masks.permute(0, 3, 1, 2).float().to(device)

            optimizer.zero_grad()

            # forward pass
            out_masks = model(in_image)
            loss = criterion(out_masks, gt_masks)

            # backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * in_image.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # evaluation
        model.eval()
        val_loss = evaluate_model(model, criterion, device, val_loader)

        logger.info(
            f"Epoch {epoch}/{num_epochs} - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}"
        )
