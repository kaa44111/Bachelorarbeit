import os
import sys
import time
import copy
from collections import defaultdict
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to PYTHONPATH if needed
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)
    
# from .config import get_args  # Import CLI arguments
# # Get command-line arguments
# args = get_args()

import argparse

def get_args():
    """
    Defines and parses command-line arguments for experimentation.
    """
    parser = argparse.ArgumentParser(description="U-Net Training Experiments")

    # Experiment configuration
    parser.add_argument("--experiment", type=str, default="baseline", help="Experiment name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--patch_size", type=int, default=250, help="Patch size for cropping")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--scheduler", type=str, default="StepLR", choices=["StepLR", "ReduceLROnPlateau"], help="Scheduler type")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    parser.add_argument("--batch_norm", action="store_true", help="Use Batch Normalization")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained encoder")

    return parser.parse_args()
args = get_args()


def compute_accuracy(outputs, labels, threshold=0.5):
    # Convert model outputs to probabilities and threshold them
    preds = (torch.sigmoid(outputs) > threshold).float()
    # Compare with ground truth labels (assumed binary)
    correct = (preds == labels).float().sum()
    total = torch.numel(labels)
    return correct / total

# Loss Functions
def dice_loss(pred, target, smooth=1.0):
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    return (1 - dice).mean()

def calculate_iou(pred, target, threshold=0.5):
    # Binarize predictions and targets
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def calculate_f1(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    return (2*tp + 1e-6)/(2*tp + fp + fn + 1e-6)

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(torch.sigmoid(pred), target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    batch_size = target.size(0)
    metrics['bce'] += bce.item() * batch_size
    metrics['dice'] += dice.item() * batch_size
    metrics['loss'] += loss.item() * batch_size
    return loss

# Training Function
def train_model(model, dataloaders, optimizer, scheduler, num_epochs=25, save_name=None):
    writer = SummaryWriter(log_dir=f"runs/{save_name}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    
    # Lists for epoch-level logging
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_ious, val_ious = [], []
    train_f1s, val_f1s = [], []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}\n{'-' * 10}")
        # Initialize metrics accumulators
        train_loss = 0.0
        train_acc = 0.0
        train_iou = 0.0
        train_f1 = 0.0
        train_samples = 0
        
        # ---------------------
        # Training Phase
        # ---------------------
        model.train()
        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Use loss functions for optimization
            metrics_dict = defaultdict(float)
            loss = calc_loss(outputs, labels, metrics_dict)
            loss.backward()
            optimizer.step()
            
            batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size
            
            # Compute evaluation metrics for logging (no gradient needed)
            acc = compute_accuracy(outputs, labels, threshold=0.5)
            iou = calculate_iou(torch.sigmoid(outputs), labels, threshold=0.5)
            f1 = calculate_f1(torch.sigmoid(outputs), labels, threshold=0.5)
            
            train_acc += acc.item() * batch_size
            train_iou += iou.item() * batch_size
            train_f1 += f1.item() * batch_size
            train_samples += batch_size
        
        epoch_train_loss = train_loss / train_samples
        epoch_train_acc = train_acc / train_samples
        epoch_train_iou = train_iou / train_samples
        epoch_train_f1 = train_f1 / train_samples
        
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        train_ious.append(epoch_train_iou)
        train_f1s.append(epoch_train_f1)
        
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar('IoU/train', epoch_train_iou, epoch)
        writer.add_scalar('F1/train', epoch_train_f1, epoch)
        
        # ---------------------
        # Validation Phase
        # ---------------------
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_iou = 0.0
        val_f1 = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                metrics_dict = defaultdict(float)
                loss = calc_loss(outputs, labels, metrics_dict)
                batch_size = inputs.size(0)
                val_loss += loss.item() * batch_size
                
                # Compute metrics for evaluation
                acc = compute_accuracy(outputs, labels, threshold=0.5)
                iou = calculate_iou(torch.sigmoid(outputs), labels, threshold=0.5)
                f1 = calculate_f1(torch.sigmoid(outputs), labels, threshold=0.5)
                
                val_acc += acc.item() * batch_size
                val_iou += iou.item() * batch_size
                val_f1 += f1.item() * batch_size
                val_samples += batch_size
        
        epoch_val_loss = val_loss / val_samples
        epoch_val_acc = val_acc / val_samples
        epoch_val_iou = val_iou / val_samples
        epoch_val_f1 = val_f1 / val_samples
        
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        val_ious.append(epoch_val_iou)
        val_f1s.append(epoch_val_f1)
        
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
        writer.add_scalar('IoU/val', epoch_val_iou, epoch)
        writer.add_scalar('F1/val', epoch_val_f1, epoch)
        
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Train IoU: {epoch_train_iou:.4f}, Train F1: {epoch_train_f1:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val IoU: {epoch_val_iou:.4f}, Val F1: {epoch_val_f1:.4f}")
        
        # Save best model weights based on validation loss
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        
        scheduler.step()
    
    # Load best model weights and close TensorBoard writer
    model.load_state_dict(best_model_wts)
    writer.close()
    
    return model

def select_scheduler(optimizer, scheduler_type, **kwargs):
    """
    Select and return a learning rate scheduler.

    Parameters:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be scheduled.
        scheduler_type (str): Type of scheduler. Options are 'StepLR' or 'ReduceLROnPlateau'.
        **kwargs: Additional keyword arguments for the scheduler.

    Returns:
        scheduler: An instantiated learning rate scheduler.
    """
    if scheduler_type == "StepLR":
        # Example: step_size=7, gamma=0.1
        return lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type == "ReduceLROnPlateau":
        # Example: mode='min', factor=0.1, patience=5
        return lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

# Run Training Pipeline
def run(UNet, dataloaders, dataset_name, save_name=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Initializing model...")
    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = select_scheduler(optimizer, args.scheduler, step_size=7, gamma=0.1)
    print("Starting training...")
    model = train_model(model, dataloaders, optimizer, scheduler, num_epochs=args.epochs, save_name=args.experiment)
    save_path = os.path.join("train/results", dataset_name, f"{args.experiment}_Patch{args.patch_size}_LR{args.lr}_{args.scheduler}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
