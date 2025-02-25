import os
import sys
import time
import copy
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to PYTHONPATH if needed
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)


##########################################
# Loss and Metrics Functions
##########################################
def dice_loss(pred, target, smooth=1.0):
    """Computes Dice loss for segmentation."""
    pred = pred.contiguous()
    target = target.contiguous()
    # Sum over H and W dimensions
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2.0 * intersection + smooth) / (
        pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth
    )
    loss = 1 - dice
    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    """
    Calculates the combined Binary Cross Entropy (BCE) and Dice loss.
    Updates the metrics dictionary.
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred_sigmoid = torch.sigmoid(pred)
    dice = dice_loss(pred_sigmoid, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)

    batch_size = target.size(0)
    metrics['bce'] += bce.detach().cpu().numpy() * batch_size
    metrics['dice'] += dice.detach().cpu().numpy() * batch_size
    metrics['loss'] += loss.detach().cpu().numpy() * batch_size

    return loss


def print_metrics(metrics, epoch_samples, phase):
    """Prints average metrics for the current phase."""
    outputs = [f"{k}: {metrics[k] / epoch_samples:4f}" for k in metrics.keys()]
    print(f"{phase}: {', '.join(outputs)}")


##########################################
# Training Function
##########################################
def train_model(model, dataloaders, optimizer, scheduler, num_epochs=25, save_name=None):
    """
    Trains the model and logs training details to TensorBoard.
    Flattens the (B, P, C, H, W) input into (B*P, C, H, W) so that the model can process it.
    Saves the best model weights.
    """
    writer = SummaryWriter(log_dir=f"runs/{save_name}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)
        since = time.time()

        # Run both training and validation phases
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                for param_group in optimizer.param_groups:
                    print("LR:", param_group["lr"])
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels, _ in dataloaders[phase]:
                # Move inputs and labels to device
                torch.cuda.empty_cache()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Flatten the patch dimension: (B, P, C, H, W) -> (B*P, C, H, W)
                B, P, C, H, W = inputs.shape
                inputs_flat = inputs.view(-1, C, H, W)
                labels_flat = labels.view(-1, C, H, W)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs_flat)
                    loss = calc_loss(outputs, labels_flat, metrics)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                epoch_samples += (B * P)

            epoch_loss = metrics["loss"] / epoch_samples
            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            print_metrics(metrics, epoch_samples, phase)

            if phase == "val" and epoch_loss < best_loss:
                print("Saving best model weights")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == "train":
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

            # Update scheduler after training phase
            if phase == "train":
                scheduler.step()

        time_elapsed = time.time() - since
        print("{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Learning Rate", current_lr, epoch)

        # Log model graph on first epoch
        if epoch == 0:
            writer.add_graph(model, inputs_flat)

    print(f"Best val loss: {best_loss:4f}")
    model.load_state_dict(best_model_wts)

    # Plot and save the training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(f"{save_name}.png")
    plt.show()

    writer.add_figure("Loss Curve", plt.gcf())
    writer.close()

    return model


##########################################
# Scheduler Selector
##########################################
def select_scheduler(optimizer, scheduler_type, **kwargs):
    if scheduler_type == "StepLR":
        return lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type == "MultiStepLR":
        return lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif scheduler_type == "ExponentialLR":
        return lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_type == "ReduceLROnPlateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


##########################################
# Run Training Pipeline
##########################################
def run(UNet, dataloader, dataset_name, save_name=None):
    """
    Initializes the model, optimizer, and scheduler then trains the model.
    Saves the trained model weights.
    """
    num_class = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(num_class).to(device)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    scheduler_type = "StepLR"
    scheduler_params = {"step_size": 7, "gamma": 0.1}
    exp_lr_scheduler = select_scheduler(optimizer_ft, scheduler_type, **scheduler_params)

    start = time.time()
    model = train_model(model, dataloader, optimizer_ft, exp_lr_scheduler, num_epochs=30, save_name=save_name)
    end = time.time()

    elapsed_time_min = (end - start) / 60
    print(f"Elapsed time: {elapsed_time_min:.2f} minutes\n")

    if save_name is None:
        save_name = "test_1s"
    results_dir = os.path.join("train/results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    save_dir = os.path.join(results_dir, f"{save_name}_{scheduler_type}.pth")

    torch.save(model.state_dict(), save_dir)
    print(f"Model saved to {save_dir}")
