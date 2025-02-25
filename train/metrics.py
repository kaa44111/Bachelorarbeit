import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F

# Add parent directory to PYTHONPATH if needed
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

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
    pred = (pred > threshold).float()
    intersection = (pred & target).sum()
    union = (pred | target).sum()
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