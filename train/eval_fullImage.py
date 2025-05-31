import sys
import os

#Den Projektpfad zu sys.path hinzufügen
project_path = os.path.abspath(os.path.dirname(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from models.UNet_new import UNet
from torchvision import transforms


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

def sliding_window_inference(image_tensor, model, patch_size, stride):
    """
    Evaluates a full image by extracting patches with a sliding window, running them
    through the model, and stitching the output patches back together.
    
    Parameters:
        image_tensor (Tensor): Full image as a tensor of shape (C, H, W)
        model (nn.Module): The trained segmentation model.
        patch_size (tuple): The size (height, width) of the patches (e.g., (256, 256)).
        stride (tuple): The stride (vertical, horizontal) for the sliding window.
        
    Returns:
        Tensor: The reconstructed prediction of shape (n_classes, H, W)
    """
    model.eval()
    device = next(model.parameters()).device
    C, H, W = image_tensor.shape
    ph, pw = patch_size
    sh, sw = stride

    # We'll accumulate predictions and a count map to average overlapping predictions.
    n_classes = model.n_classes if hasattr(model, "n_classes") else 1
    output = torch.zeros((n_classes, H, W), device=device)
    count_map = torch.zeros((n_classes, H, W), device=device)

    with torch.no_grad():
        # Slide over the image
        for i in range(0, H - ph + 1, sh):
            for j in range(0, W - pw + 1, sw):
                patch = image_tensor[:, i:i+ph, j:j+pw].unsqueeze(0).to(device)
                pred = model(patch)  # Expected shape: (1, n_classes, ph, pw)
                pred = F.softmax(pred, dim=1) if n_classes > 1 else torch.sigmoid(pred)
                pred = pred.squeeze(0)  # (n_classes, ph, pw)
                output[:, i:i+ph, j:j+pw] += pred
                count_map[:, i:i+ph, j:j+pw] += 1

        # If there are border areas not covered due to stride, you may need to handle them here.
        # For now, we assume the patch extraction fully covers the image.
        # Average overlapping predictions:
        output /= count_map
    return output

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load full image and convert to tensor
    img = Image.open("data/Dichtflächen/grabs/1.bmp").convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [C, H, W] in [0, 1]
    ])
    image_tensor = transform(img)  # shape: (3, H, W)

    model = load_model('train/results/Dichtflächen_Cropped/baseline_Patch250_LR0.0001_StepLR.pth', UNet, device, n_channels=3, n_classes=1, bilinear=False)

    # Set patch size and stride (e.g., 256x256 patches, stride 128 for 50% overlap)
    patch_size = (256, 256)
    stride = (128, 128)

    # Evaluate full image
    prediction = sliding_window_inference(image_tensor, model, patch_size, stride)

    # Convert prediction to numpy for visualization (example: if binary segmentation, threshold it)
    prediction_np = prediction.cpu().numpy()
    # For binary segmentation, you might do:
    if prediction_np.shape[0] == 1:
        binary_pred = (prediction_np[0] > 0.5).astype(np.uint8)
        import matplotlib.pyplot as plt
        plt.imshow(binary_pred, cmap="gray")
        plt.title("Segmentation Prediction")
        plt.show()
