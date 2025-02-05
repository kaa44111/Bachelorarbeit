import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision.transforms import v2 
# import seaborn as sns
# import matplotlib.pyplot as plt

# from datasets.OneFeature import CustomDataset
# from utils.heatmap_utils import show_masks_pred1, show_masks_pred, save_valuation


# def show_predictions(images, masks, preds, idx):
#     """
#     Zeigt die tatsächlichen Masken und die vorhergesagten Masken für eine gegebene Index an.
#     """
#     fig, axes = plt.subplots(3, 3, figsize=(15, 15))

#     for i in range(3):
#         # Original image
#         img = images[idx + i].cpu().numpy().transpose((1, 2, 0))
#         axes[i, 0].imshow(img)
#         axes[i, 0].set_title('Original Image')
#         axes[i, 0].axis('off')

#         # Ground truth mask
#         mask = masks[idx + i].cpu().squeeze().numpy()
#         sns.heatmap(mask, ax=axes[i, 1], cmap='viridis')
#         axes[i, 1].set_title('Original Mask')
#         axes[i, 1].axis('off')

#         # Predicted mask
#         pred = preds[idx + i].cpu().squeeze().numpy()
#         sns.heatmap(pred, ax=axes[i, 2], cmap='viridis')
#         axes[i, 2].set_title('Predicted Mask')
#         axes[i, 2].axis('off')

#     plt.tight_layout()
#     plt.show()

# def test(UNet,test_dir,test_trained_model,transformations,dataset_name=None):
#     num_class = 1
#     #num_class = 6
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     model = UNet(num_class).to(device)
#     model.load_state_dict(torch.load(test_trained_model, map_location=device))
#     model.eval()

#     test_dataset = CustomDataset(test_dir, dataset_name=dataset_name, transform=transformations, count=3)
#     test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=0)

#     images, masks_tensor = next(iter(test_loader))
#     images = images.to(device)
#     masks_tensor = masks_tensor.to(device)

#     pred = model(images)
#     pred = F.sigmoid(pred)
#     max = pred.max()
#     pred = pred.data.cpu()#.numpy()
#     print(pred.shape)
#     print(images.shape)
#     print(masks_tensor.shape)


#     show_predictions(images, masks_tensor, pred, 0)

#     #show_masks_pred(mask=masks_tensor,pred=pred)
#     #save_valuation(images, masks_tensor, pred)



# if __name__ == '__main__':
#     try:
#         from models.UNet import UNet
#         #from models.UNetBatchNorm import UNetBatchNorm
#         #from models.UNetMaxPool import UNetMaxPool

#         test_dir = 'data/Ölflecken'
#         dataset_name = 'Ölflecken'
#         test_trained_model = 'train/results/Ölflecken/test_train.pth'

#         test(UNet,test_dir,dataset_name,test_trained_model)

#     except Exception as e:
#         print(f"An error occurred: {e}")

#___________________________________________________________
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
from datasets.test import CustomDataset
import matplotlib.pyplot as plt

def load_model(model_class, checkpoint_path, num_class=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_class(num_class)
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    model.eval()
    return model

def save_predictions_with_originals_matplotlib(model, dataloader, output_folder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_folder, exist_ok=True)

    for i, (inputs) in enumerate(dataloader):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

        preds = preds.cpu().numpy()
        inputs = inputs.cpu().numpy()

        for j in range(preds.shape[0]):
            original_img = inputs[j].transpose(1, 2, 0)  # Assuming channel first format
            original_img = (original_img * 255).astype(np.uint8)
            pred_img = preds[j][0] * 255  # Assuming single channel output
            pred_img = pred_img.astype(np.uint8)

            # Plotting the images using matplotlib
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(original_img)
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            
            ax[1].imshow(pred_img, cmap='gray')
            ax[1].set_title('Predicted Mask')
            ax[1].axis('off')

            # Save the figure
            save_path = os.path.join(output_folder, f"combined_{i * preds.shape[0] + j}.png")
            plt.savefig(save_path)
            plt.close(fig)

    print(f"Combined images saved to {output_folder}")

def show_predictions_with_heatmaps(model, dataloader, num_images_per_window=3):    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    images, preds = [], []

    for inputs in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            pred = torch.sigmoid(outputs)

        images.extend(inputs.cpu().numpy())
        preds.extend(pred.cpu().numpy())

    total_images = len(images)
    num_windows = (total_images + num_images_per_window - 1) // num_images_per_window

    for window_index in range(num_windows):
        start_index = window_index * num_images_per_window
        end_index = min(start_index + num_images_per_window, total_images)
        num_images_in_window = end_index - start_index

        fig, axes = plt.subplots(num_images_in_window, 2, figsize=(10, 5 * num_images_in_window))

        for i in range(num_images_in_window):
            img_index = start_index + i
            img = images[img_index].transpose(1, 2, 0)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            pred = preds[img_index][0]
            sns.heatmap(pred, ax=axes[i, 1], cmap='viridis', vmin=0.0, vmax=1.0)
            axes[i, 1].set_title('Predicted Mask')
            axes[i, 1].axis('off')

        plt.tight_layout()
        plt.show()

def test_model(UNet, test_dataloader, checkpoint_path, output_folder):
    # Load the model
    model = load_model(UNet, checkpoint_path)

    # Save predictions with originals using matplotlib
    print("Saving predictions with originals using matplotlib...")
    save_predictions_with_originals_matplotlib(model, test_dataloader, output_folder)

if __name__ == '__main__':

    try:
        from models.UNetBatchNorm import UNetBatchNorm
        test_dir = 'data/data_modified/Dichtflächen_Cropped/patched_NIO'
    
        transformations = v2.Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        test_dataset = CustomDataset(test_dir, transform=transformations,is_labeled=None)
        test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True, num_workers=0)

        model = load_model(UNetBatchNorm,'train/results/Dichtflächen_Cropped/Patched_BN_NIO_NoAug_20_Adam_StepLR.pth')

        #test_model(UNetBatchNorm, test_loader, 'train/results/Dichtflächen/test_UNetBatchNorm.pth', 'test_models/evaluate/Dichfläche')

        # Zeige die Vorhersagen für die ersten 3 Bilder mit Heatmaps an
        show_predictions_with_heatmaps(model, test_loader, num_images_per_window=2)    
    
    except Exception as e:
        print(f"An error occurred: {e}")
