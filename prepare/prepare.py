import sys
import os

#Den Projektpfad zu sys.path hinzufügen
project_path = os.path.abspath(os.path.dirname(__file__))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from PIL import Image
import numpy as np

from utils.data_utils import find_image_and_mask_files_folder


def extract_patches(image, patch_size, use_padding):
    '''
    If Padding = True -> Zero Padding :Rand wird mit 0 werte gefüllt.
    If Padding = Flase -> Rand wird einfach rausgeschnitten.
    '''

    width, height = image.size
    patches = []
    for i in range(0, width, patch_size):
        for j in range(0, height, patch_size):
            box = (i, j, i + patch_size, j + patch_size)
            if use_padding:
                patch = Image.new(image.mode, (patch_size, patch_size))
                patch.paste(image.crop(box), (0, 0))
                patches.append(patch)
            else:
                if box[2] <= width and box[3] <= height:
                    patch = image.crop(box)
                    patches.append(patch)
    return patches

def downsample_image(img, scale_factor):
    new_size = (int(img.width / scale_factor), int(img.height / scale_factor))
    return img.resize(new_size, Image.Resampling.LANCZOS)

def is_empty_mask(mask_patch):
    return np.array(mask_patch).sum() == 0


#####################
# Preprocessing for pre-separated dataset
#####################


def process_pre_separated_dataset(dataset_name, patch_size, use_padding=False):
    base_dir = f"data/{dataset_name}"
    for category in ["IO", "NIO"]:
        image_folder = os.path.join(base_dir, category, "grabs")
        mask_folder = os.path.join(base_dir, category, "masks")
        # Define output directory for patches for this category:
        output_dir = os.path.join("data/data_modified", dataset_name, f"patched_{category}")
        output_img_dir = os.path.join(output_dir, "grabs")
        output_mask_dir = os.path.join(output_dir, "masks")
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_mask_dir, exist_ok=True)
        
        image_files = sorted(os.listdir(image_folder))
        mask_files = sorted(os.listdir(mask_folder))  # Assuming they match
        
        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(image_folder, img_file)
            mask_path = os.path.join(mask_folder, mask_file)
            try:
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('L')
                
                # Extract patches from both:
                img_patches = extract_patches(img, patch_size, use_padding)
                mask_patches = extract_patches(mask, patch_size, use_padding)
                
                for i, (img_patch, mask_patch) in enumerate(zip(img_patches, mask_patches)):
                    img_name = f"{os.path.splitext(img_file)[0]}_patch{i+1}.tif"
                    mask_name = f"{os.path.splitext(mask_file)[0]}_patch{i+1}.tif"
                    output_img_path = os.path.join(output_img_dir, img_name)
                    output_mask_path = os.path.join(output_mask_dir, mask_name)
                    img_patch.save(output_img_path)
                    mask_patch.save(output_mask_path)
            except Exception as e:
                print(f"Error processing {img_file} or {mask_file}: {e}")
    print("Preprocessing for pre-separated dataset complete.")
