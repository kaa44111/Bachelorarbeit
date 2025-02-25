import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

# Add project root to PYTHONPATH (if needed)
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

# Import your training utilities and U-Net model
from train.train import run, train_model, print_metrics, select_scheduler, calc_loss
from models.UNetBatchNorm import UNetBatchNorm

##########################################
# Helper function: Extract All Patches
##########################################
def extract_all_patches(image, patch_size=(256, 256)):
    """
    Extracts all non-overlapping patches of size patch_size from an image.
    Skips partial patches if they don't fit exactly.
    
    Parameters:
        image (np.array): Input image as a numpy array. Shape (H, W) or (H, W, C).
        patch_size (tuple): Desired patch size (height, width).
        
    Returns:
        List of patches (each patch is a numpy array with shape patch_size or patch_size+C).
    """
    h, w = image.shape[:2]
    ph, pw = patch_size

    patches = []
    for y in range(0, h, ph):
        for x in range(0, w, pw):
            # Extract patch; works for grayscale or RGB
            patch = image[y:y+ph, x:x+pw, ...] if image.ndim == 3 else image[y:y+ph, x:x+pw]
            # Only use patches that match exactly the desired size
            if patch.shape[0] == ph and patch.shape[1] == pw:
                patches.append(patch)
    return patches

##########################################
# Custom Dataset for Industrial Images
##########################################
class CustomDataset(Dataset):
    def __init__(self, root_dir, dataset_name=None, transform=None, mask_transform=None,
                 count=None, patch_size=250):
        """
        Parameters:
            root_dir (str): Root directory for the dataset.
            dataset_name (str): Optionally specify dataset type to adjust mask filename.
            transform: Transformation pipeline for images.
            mask_transform: Transformation pipeline for masks.
            count (int): Optionally use only a subset of images.
            patch_size (int): The height/width of each extracted patch.
        """
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.transform = transform
        self.mask_transform = mask_transform
        self.count = count
        self.patch_size = patch_size

        # Folder names (adjust if needed)
        self.image_folder = os.path.join(root_dir, 'grabs')
        self.mask_folder = os.path.join(root_dir, 'masks')

        # Get all image files (sorted numerically if filenames contain digits)
        all_image_files = sorted(os.listdir(self.image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))


        self.image_files = []
        self.mask_files = []

        # Build corresponding lists for image and mask files.
        # If your naming convention differs among your three datasets,
        # you may adjust or extend this logic accordingly.
        for image_file in all_image_files:
            base_name = os.path.splitext(image_file)[0]
            self.image_files.append(image_file)
            if dataset_name == "RetinaVessel":
                mask_name = f"{base_name}.tiff"
            elif dataset_name == "Ölflecken":
                mask_name = f"{base_name}_1.bmp"
            elif dataset_name == "circle_data":
                mask_name = f"{base_name}1.png"
            else:
                # Default: same base name with .tif extension.
                mask_name = f"{base_name}.tif"
            
            mask_path = os.path.join(self.mask_folder, mask_name)
            if os.path.exists(mask_path):
                self.mask_files.append(mask_name)
            else:
                # For images without anomalies, you could optionally use a blank mask
                # instead of raising an error. For now, we assume a blank mask is stored.
                raise FileNotFoundError(f"Mask file not found: {mask_path}")

        if self.count is not None:
            self.image_files = self.image_files[:self.count]
            self.mask_files = self.mask_files[:self.count]

        print(f"Found {len(self.image_files)} images")
        print(f"Found {len(self.mask_files)} masks")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            # Load image and mask
            img_path = os.path.join(self.image_folder, self.image_files[idx])
            image = np.array(Image.open(img_path).convert('RGB'))
            mask_path = os.path.join(self.mask_folder, self.mask_files[idx])
            mask = np.array(Image.open(mask_path).convert('L'))
            
            # Extract all non-overlapping patches
            image_patches = extract_all_patches(image, patch_size=(self.patch_size, self.patch_size))
            mask_patches = extract_all_patches(mask, patch_size=(self.patch_size, self.patch_size))

            processed_patches = []
            processed_mask_patches = []
            defect_labels = []  # 1 if the patch contains defect pixels, 0 otherwise

            for img_patch, mask_patch in zip(image_patches, mask_patches):
                if self.transform:
                    img_patch = self.transform(Image.fromarray(img_patch))
                if self.mask_transform:
                    mask_patch = self.mask_transform(Image.fromarray(mask_patch))
                processed_patches.append(img_patch)
                processed_mask_patches.append(mask_patch)
                # A patch is considered defective if any pixel in its mask > 0.5
                label = torch.any(mask_patch > 0.5).float()
                defect_labels.append(label)
            
            # Stack patches so that each image returns:
            #   image_tensor: (num_patches, C, H, W)
            #   mask_tensor: (num_patches, C, H, W)
            #   defect_labels: (num_patches,)
            image_tensor = torch.stack(processed_patches)
            mask_tensor = torch.stack(processed_mask_patches)
            defect_labels = torch.stack(defect_labels)

            return image_tensor, mask_tensor, defect_labels

        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None, None, None

##########################################
# Visualization Function
##########################################
def show_image_and_mask(image, mask, save_path=None):
    """
    Displays a single patch (or saves the figure).
    
    Parameters:
       image: A tensor of shape (C, H, W)
       mask: A tensor of shape (C, H, W) or (H, W)
       save_path (str, optional): If provided, save the figure to disk.
    """
    # If image has extra dimensions, select the first patch
    if image.ndim > 3:
        image = image[0]
    if mask.ndim > 2:
        mask = mask[0]
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    ax[0].set_title("Image Patch")
    ax[1].imshow(mask.squeeze().cpu().numpy(), cmap='gray')
    ax[1].set_title("Mask Patch")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

def binarize(tensor, threshold=0.001):
    return (tensor > threshold).float()

##########################################
# Dataloader Setup Function
##########################################
def get_dataloaders(root_dir, dataset_name=None, batch_size=2, split_size=0.8, patch_size=250):
    """
    Creates train and validation DataLoaders.
    The transformation pipelines use torchvision.transforms.v2.
    
    Parameters:
       root_dir: Dataset root directory.
       dataset_name: Name to help adjust file lookup (if needed).
       batch_size: Number of images per batch.
       split_size: Fraction for training split.
       patch_size: Size for non-overlapping patch extraction.
    """
    # Define the transformation pipelines for images and masks.
    transformations = v2.Compose([
        v2.ToTensor(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    transformations_mask = v2.Compose([
        v2.ToTensor(),
        v2.ToDtype(torch.float32),
        binarize,  # Binarize mask (0 or 1)
        
    ])

    dataset = CustomDataset(root_dir=root_dir, dataset_name=dataset_name,
                            transform=transformations, mask_transform=transformations_mask,
                            patch_size=patch_size)
    dataset_size = len(dataset)
    train_size = int(split_size * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    dataloaders = {'train': train_loader, 'val': val_loader}
    return dataloaders, dataset

##########################################
# Main (Example Usage)
##########################################
if __name__ == '__main__':
    data_dir = 'data/Prototype'
    dataset_name = 'Prototype'  # Adjust if you have specific naming conventions

    # For industrial inspection, you might set a patch size that captures fine defects;
    # here we use 250x250, but feel free to adjust.
    dataloaders, dataset = get_dataloaders(root_dir=data_dir, dataset_name=dataset_name,
                                           batch_size=2, split_size=0.8, patch_size=250)
    batch = next(iter(dataloaders['train']))
    images, masks, defects = batch
    print("Images shape (B, P, C, H, W):", images.shape)
    print("Masks shape (B, P, C, H, W):", masks.shape)
    print("Defect labels shape (B, P):", defects.shape)
    #print("Defect labels (first batch):", defects[0])

    # # Find a patch where mask_patch.max() > 0 in the batch.
    # found = False
    # for batch_idx in range(images.shape[0]):  # iterate over images in the batch
    #     for patch_idx in range(images.shape[1]):  # iterate over each patch in the image
    #         mask_patch = masks[batch_idx, patch_idx]  # shape (C, H, W)
    #         if mask_patch.max() > 0:
    #             print(f"Defective patch found at batch index {batch_idx}, patch index {patch_idx}")
    #             img_patch = images[batch_idx, patch_idx]
    #             print("Mask max value:", mask_patch.max())
    #             show_image_and_mask(img_patch, mask_patch, save_path=f'patch_example_{batch_idx}_{patch_idx}.png')
    #             found = True
    #             break
    #     if found:
    #         break
    # if not found:
    #     print("No defective patch found where mask_patch.max() > 0.")

    run(UNetBatchNorm, dataloaders, dataset_name, save_name="Patched_BN_NIO_NoAug_20_Adam")    
    # model = UNetBatchNorm(in_channels=3, out_channels=1, init_features=32)

    