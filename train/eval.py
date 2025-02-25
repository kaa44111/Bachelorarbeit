import os, sys
# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the project root to be the parent directory of the current file
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# Add project root to the Python path if it's not already there
if project_root not in sys.path:
    sys.path.append(project_root)


import torch
import matplotlib.pyplot as plt
from train import calc_loss, compute_accuracy, calculate_iou, calculate_f1
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from collections import defaultdict
#from train.train import run
from torchvision.transforms import v2
import seaborn as sns
from datasets.new_dataset import CustomDataset
import matplotlib.pyplot as plt
from models.UNet_new import UNet

import torch

def load_model(checkpoint_path, model_class, device, **model_kwargs):
    """
    Loads a model from a checkpoint.

    Parameters:
        checkpoint_path (str): Path to the saved checkpoint file.
        model_class (class): The class of the model to instantiate.
        device (torch.device): The device on which to load the model.
        **model_kwargs: Additional keyword arguments to instantiate the model.

    Returns:
        model: The loaded model, moved to the specified device and set to eval mode.
    """
    try:
        # Try loading the state dictionary first
        state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model = model_class(**model_kwargs)
        model.load_state_dict(state_dict)
        print("Loaded model weights using state_dict.")
    except Exception as e:
        print("Failed to load state_dict, attempting to load full model. Error:", e)
        model = torch.load(checkpoint_path, map_location=device)
        print("Loaded full model object.")

    model.to(device)
    model.eval()
    return model

print("Model weights loaded successfully.")

def visualize_predictions(model, dataloader, num_samples=3, threshold=0.5):
    import matplotlib.pyplot as plt
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    samples_shown = 0
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > threshold).float()
            
            for i in range(inputs.size(0)):
                if samples_shown >= num_samples:
                    return
                img = inputs[i].cpu().permute(1, 2, 0).numpy()
                gt_mask = labels[i].cpu().squeeze().numpy()
                pred_mask = preds[i].cpu().squeeze().numpy()
                
                plt.figure(figsize=(12,4))
                plt.subplot(1,3,1)
                plt.imshow(img)
                plt.title("Input Image")
                plt.subplot(1,3,2)
                plt.imshow(gt_mask, cmap='gray')
                plt.title("Ground Truth")
                plt.subplot(1,3,3)
                plt.imshow(pred_mask, cmap='gray')
                plt.title("Prediction")
                plt.show()
                
                samples_shown += 1

def test_model(model, dataloader, threshold=0.5):
    """
    Evaluates the model on a given dataloader and prints the average loss, accuracy, IoU, and F1 score.
    
    Parameters:
        model (torch.nn.Module): Trained model.
        dataloader (torch.utils.data.DataLoader): Dataloader for evaluation (validation or test set).
        threshold (float): Threshold for converting logits to binary predictions.
    
    Returns:
        Tuple containing average loss, accuracy, IoU, and F1 score.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    total_samples = 0
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Compute loss for evaluation
            metrics_dict = defaultdict(float)
            loss = calc_loss(outputs, labels, metrics_dict)
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            
            # Compute metrics using your helper functions
            acc = compute_accuracy(outputs, labels, threshold=threshold)
            # For IoU and F1, apply sigmoid to get probabilities
            iou = calculate_iou(torch.sigmoid(outputs), labels, threshold=threshold)
            f1 = calculate_f1(torch.sigmoid(outputs), labels, threshold=threshold)
            
            total_acc += acc.item() * batch_size
            total_iou += iou.item() * batch_size
            total_f1 += f1.item() * batch_size
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    avg_iou = total_iou / total_samples
    avg_f1 = total_f1 / total_samples
    
    print(f"Evaluation Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {avg_acc:.4f}")
    print(f"  IoU: {avg_iou:.4f}")
    print(f"  F1 Score: {avg_f1:.4f}")
    
    return avg_loss, avg_acc, avg_iou, avg_f1


if __name__ == '__main__':

    try:
        test_dir = 'data/data_modified/Dichtflächen/processed_NIO'

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        transformations = v2.Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        transformations1 = v2.Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=False),
        ])

        test_dataset = CustomDataset(test_dir,dataset_name='Dichtflächen', transform=transformations , count=3,is_labeled=False)

        test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=0)
        
        model = load_model('train/results/Dichtflächen_Cropped/baseline_Patch250_LR0.0001_StepLR.pth', UNet, device, n_channels=3, n_classes=1, bilinear=False)
        results = test_model(UNet, test_loader, threshold=0.5)
        visualize_predictions(model, test_loader, threshold=0.5, num_samples=3)
    
    except Exception as e:
        print(f"Error loading model: {e}")
