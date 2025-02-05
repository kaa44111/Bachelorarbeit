import torch
from torchvision.transforms import v2
import os
import shutil
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

def find_image_and_mask_files_folder(root_dir, dataset_name=None):
    '''retruns image_folder, mask_folder, image_files, mask_files'''

    image_folder = os.path.join(root_dir, 'train', 'grabs')
    if not os.path.exists(image_folder):
        image_folder = os.path.join(root_dir, 'grabs')

    mask_folder = os.path.join(root_dir, 'train', 'masks')
    if not os.path.exists(mask_folder):
        mask_folder = os.path.join(root_dir, 'masks')
    
    if dataset_name is None:
        dataset_name = os.path.basename(root_dir.rstrip('/\\'))

    all_image_files = sorted(os.listdir(image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))

    image_files = []
    mask_files = []

    for image_file in all_image_files:
        base_name = os.path.splitext(image_file)[0]

        if dataset_name == "WireCheck":
            mask_name = f"{base_name}.tif"
        elif dataset_name == "RetinaVessel":
            mask_name = f"{base_name}_1.bmp" 
        elif dataset_name == "Ölflecken":
            mask_name = f"{base_name}_1.bmp"
        elif dataset_name == "circle_data":
            mask_name = f"{base_name}1.png"
        else:
            mask_name = f"{base_name}.tif"

        if os.path.exists(os.path.join(mask_folder, mask_name)):
            image_files.append(image_file)
            mask_files.append(mask_name)
        # else:
        #     raise FileNotFoundError(f"Maske für Bild {image_file} nicht gefunden.")
        
    print(f"Found {len(image_files)} images")
    print(f"Found {len(mask_files)} masks")

    return image_folder, mask_folder, image_files, mask_files

#image_folder, mask_folder, image_files, mask_files = find_image_and_mask_files_folder("data/RetinaVessel","RetinaVessel")

def show_image_and_mask(image, mask):
    '''
    Zeigt die Bilder und die dazu gehörigen masken an.
    '''
    # Rücktransformieren des Bildes (um die Normalisierung rückgängig zu machen)
    image = image.numpy().transpose((1, 2, 0))

    # Maske umwandeln
    mask = mask.squeeze().numpy()

    # Anzeigen des Bildes und der Maske
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.show()

def plot_losses(train_losses, val_losses, save_path):
    '''Saves the train and test Losses in picture'''
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='train_loss', color='red')
    plt.plot(val_losses, label='val_loss', color='blue')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def show_normalized_images(unnormalized_image, normalized_image, mask):
    '''
    Zeigt das unnormierte Bild, das normierte Bild und die dazugehörige Maske an.
    '''
    # Rücktransformieren des unnormalisierten Bildes (um die Normalisierung rückgängig zu machen)
    unnormalized_image = unnormalized_image.permute(1, 2, 0).numpy()
    
    # Rücktransformieren des normalisierten Bildes
    normalized_image = normalized_image.permute(1, 2, 0).numpy()
    
    # Maske umwandeln
    mask = mask.squeeze().numpy()

    # Anzeigen des unnormalisierten Bildes, des normalisierten Bildes und der Maske
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.title("Unnormalized Image")
    plt.imshow(unnormalized_image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Normalized Image")
    plt.imshow(normalized_image)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.show()


def compute_mean_std(image_folder):
    '''
    Berechnet Mittelwerte und Standardabweichungen der Pixelwerte für die Bilder in einem Ordner.
    '''
    # Initialize sums and squared sums for each channel
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    total_pixels = 0

    # Transformation
    transform = transforms.ToTensor()

    # Iterate over all images in the folder
    for image_name in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        
        # Sum up the values and their squares
        channels_sum += image.sum(dim=[1, 2])
        channels_squared_sum += (image ** 2).sum(dim=[1, 2])
        total_pixels += image.size(1) * image.size(2)  # Anzahl der Pixel pro Bild

    # Calculate mean and std across the entire dataset
    mean = channels_sum / total_pixels
    std = torch.sqrt(channels_squared_sum / total_pixels - mean ** 2)

    return mean, std

def compute_mean_std_from_dataset(dataset):
    loader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=0)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in tqdm(loader):
        images = images[0]  # Remove batch dimension
        mean += images.mean(dim=[1, 2])
        std += images.std(dim=[1, 2])
        total_images += 1

    mean /= total_images
    std /= total_images

    return mean, std
    
# def split_data(root_dir, train_dir, val_dir, test_size=0.2, random_state=42):
#     """
#     Split Data into test and validate folders
#     Example:
#         root_dir = 'data/geometry_shapes'
#         train_dir = 'data/circle_data/train'
#         val_dir = 'data/circle_data/val'
#     """
#     # Erstellen der Zielordner
#     for dir in [train_dir, val_dir]:
#         grabs_dir = os.path.join(dir, 'grabs')
#         masks_dir = os.path.join(dir, 'masks')
#         os.makedirs(grabs_dir, exist_ok=True)
#         os.makedirs(masks_dir, exist_ok=True)
    
#     # Pfade zu den Bilder- und Maskenordnern
#     image_folder = os.path.join(root_dir, 'grabs')
#     mask_folder = os.path.join(root_dir, 'masks')
    
#     # Listen der Bild- und Maskendateien
#     image_files = sorted(os.listdir(image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
#     mask_files = sorted(os.listdir(mask_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
    
#     # Aufteilen der Daten in Trainings- und Validierungssätze
#     train_images, val_images = train_test_split(image_files, test_size=test_size, random_state=random_state, shuffle=False)

#     # Kopieren der Trainingsdaten
#     for image_file in train_images:
#         shutil.copy(os.path.join(image_folder, image_file), os.path.join(train_dir, 'grabs', image_file))
#         # Annahme, dass die Maske denselben Namen wie das Bild hat, jedoch mit einer Endung von 1
#         mask_file = image_file.split('.')[0] + '1.png'
#         if os.path.exists(os.path.join(mask_folder, mask_file)):
#             shutil.copy(os.path.join(mask_folder, mask_file), os.path.join(train_dir, 'masks', mask_file))

#     # Kopieren der Validierungsdaten
#     for image_file in val_images:
#         shutil.copy(os.path.join(image_folder, image_file), os.path.join(val_dir, 'grabs', image_file))
#         # Annahme, dass die Maske denselben Namen wie das Bild hat, jedoch mit einer Endung von 1
#         mask_file = image_file.split('.')[0] + '1.png'
#         if os.path.exists(os.path.join(mask_folder, mask_file)):
#             shutil.copy(os.path.join(mask_folder, mask_file), os.path.join(val_dir, 'masks', mask_file))


def rename_masks(mask_folder,image_folder):
    '''
    Gives Masks the same prefix as the Image name
    '''
    image_files = sorted(os.listdir(image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
    mask_files = sorted(os.listdir(mask_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))

    for i, image_file in enumerate(image_files):
        base_name = image_file.split('.')[0]
        for j in range(6):  # Annahme: Es gibt 6 Masken pro Bild
            old_mask_name = mask_files[i * 6 + j]
            new_mask_name = f"{base_name}{j}.png"
            old_mask_path = os.path.join(mask_folder, old_mask_name)
            new_mask_path = os.path.join(mask_folder, new_mask_name)
            
            if os.path.exists(old_mask_path):
                os.rename(old_mask_path, new_mask_path)
                print(f"Renamed {old_mask_name} to {new_mask_name}")

def rename_images_in_directory(directory):
    # Erstellen Sie eine Liste aller Bilddateien im Verzeichnis
    image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    
    # Sortieren Sie die Bilddateien nach ihrem numerischen Wert
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    # Schritt 1: Benennen Sie jede Bilddatei in einen temporären Namen um
    for file_name in image_files:
        # Extrahieren Sie den numerischen Teil des Dateinamens und erhöhen Sie ihn um 1
        new_number = int(os.path.splitext(file_name)[0]) + 1
        temp_name = f"temp_{new_number}.png"
        
        # Erstellen Sie die vollständigen Pfade für das Umbenennen
        old_path = os.path.join(directory, file_name)
        temp_path = os.path.join(directory, temp_name)
        
        # Benennen Sie die Datei um
        os.rename(old_path, temp_path)
        print(f"Temporarily renamed {old_path} to {temp_path}")
    
    # Schritt 2: Benennen Sie die temporären Dateien in ihre endgültigen Namen um
    temp_files = [f for f in os.listdir(directory) if f.startswith('temp_')]
    
    for temp_name in temp_files:
        # Entfernen Sie den 'temp_'-Präfix, um den endgültigen Namen zu erhalten
        final_name = temp_name.replace('temp_', '')
        
        # Erstellen Sie die vollständigen Pfade für das Umbenennen
        temp_path = os.path.join(directory, temp_name)
        final_path = os.path.join(directory, final_name)
        
        # Benennen Sie die Datei um
        os.rename(temp_path, final_path)
        print(f"Renamed {temp_path} to {final_path}")

#rename_images_in_directory('data/geometry_shapes/train/grabs')

