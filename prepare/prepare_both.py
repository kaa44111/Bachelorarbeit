import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

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

def process_images(dataset_name, downsample_factor=None, patch_size=None, use_padding=False):
    root_dir = f"data/{dataset_name}"
    image_folder, mask_folder, image_files, mask_files = find_image_and_mask_files_folder(root_dir, dataset_name)
    
    # # Debugging-Ausgaben hinzufügen
    # print(f"Image Folder: {image_folder}")
    # print(f"Mask Folder: {mask_folder}")
    # print(f"Image Files: {image_files}")
    # print(f"Mask Files: {mask_files}")
    
    output_base = f"data/data_modified/{dataset_name}"
    if downsample_factor and patch_size:
        output_dir = f"{output_base}/processed_NIO"
    elif downsample_factor:
        output_dir = f"{output_base}/downsampled_NIO"
    elif patch_size:
        output_dir = f"{output_base}/patched_NIO"
    else:
        raise ValueError("Entweder downsample_factor oder patch_size muss angegeben werden.")
    
    if os.path.exists(output_dir):
        print(f"Verzeichnis {output_dir} existiert bereits. Keine weiteren Operationen werden durchgeführt.")
        return output_dir
    
    image_modified = os.path.join(output_dir, "grabs")
    mask_modified = os.path.join(output_dir, "masks")

    os.makedirs(image_modified, exist_ok=True)
    os.makedirs(mask_modified, exist_ok=True)

    empty_mask_count = 0
    non_empty_mask_count = 0

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(mask_folder, mask_file)
        
        # Überprüfen, ob die Bild- und Maskenpfade existieren
        if not os.path.exists(img_path):
            print(f"Bilddatei nicht gefunden: {img_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"Maskendatei nicht gefunden: {mask_path}")
            continue

        try:
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            # Debugging-Ausgaben
            print(f"Verarbeite Bild: {img_path}")
            print(f"Bildgröße vor Downsampling: {img.size}")
            print(f"Maskengröße vor Downsampling: {mask.size}")

            if downsample_factor is not None:
                img = downsample_image(img, downsample_factor)
                mask = downsample_image(mask, downsample_factor)

                # Debugging-Ausgaben
                print(f"Bildgröße nach Downsampling: {img.size}")
                print(f"Maskengröße nach Downsampling: {mask.size}")

            if patch_size:
                img_patches = extract_patches(img, patch_size, use_padding)
                mask_patches = extract_patches(mask, patch_size, use_padding)

                for i, (img_patch, mask_patch) in enumerate(zip(img_patches, mask_patches)):
                    # if is_empty_mask(mask_patch):
                    #     if empty_mask_count < 100:
                    #         empty_mask_count += 1  # Inkrementiere den Zähler
                    #     else:
                    #         non_empty_mask_count += 1
                    #         continue  # Überspringe, wenn mehr als 4 leere Masken

                    if is_empty_mask(mask_patch):
                        empty_mask_count += 1  # Inkrementiere den Zähler
                        continue  # Überspringe leere Masken
                    else:
                        non_empty_mask_count += 1
                    
                    img_name = f"{os.path.splitext(img_file)[0]}_patch{i+1}.tif"
                    mask_name = f"{os.path.splitext(img_file)[0]}_patch{i+1}.tif"
                    
                    output_img_path = os.path.join(image_modified, img_name)
                    output_mask_path = os.path.join(mask_modified, mask_name)
                    
                    # Debugging-Ausgaben
                    print(f"Speichern des Bildes: {output_img_path}")
                    print(f"Speichern der Maske: {output_mask_path}")

                     # Debugging-Ausgaben
                    print(f"Bildgröße nach Patching: {img_patch.size}")
                    print(f"Maskengröße nach Patching: {mask_patch.size}")
                    
                    img_patch.save(output_img_path)
                    mask_patch.save(output_mask_path)
            else:

                if is_empty_mask(mask):
                    empty_mask_count += 1
                else:
                    non_empty_mask_count += 1

                img_name = f"{os.path.splitext(img_file)[0]}_binned.tif"
                mask_name = f"{os.path.splitext(img_file)[0]}_binned.tif"
                    
                output_img_path = os.path.join(image_modified, img_name)
                output_mask_path = os.path.join(mask_modified, mask_name)
                
                # Debugging-Ausgaben
                print(f"Speichern des Bildes: {output_img_path}")
                print(f"Speichern der Maske: {output_mask_path}")
                
                img.save(output_img_path)
                mask.save(output_mask_path)

        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei {img_file} oder {mask_file}: {e}")
    
    #process_test_images(dataset_name,downsample_factor,patch_size,use_padding,output_dir)

    print(f"Verarbeitung abgeschlossen. Ergebnisse gespeichert in: {output_dir}")
    print(f"Anzahl der Bilder mit leeren Masken: {empty_mask_count}")
    print(f"Anzahl der Bilder mit nicht-leeren Masken: {non_empty_mask_count}")

    return output_dir

def process_test_images(dataset_name, downsample_factor=None, patch_size=None, use_padding=False, output_processed=None):
    test_image_folder = os.path.join("data", dataset_name, 'test')
    if not os.path.exists(test_image_folder):
        print(f"Kein 'test'-Verzeichnis gefunden in data/{dataset_name}")
        return

    test_image_files = [f for f in os.listdir(test_image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tif','.bmp'))]
    output_base = f"data/data_modified/{dataset_name}"

    if output_processed is None:
        output_dir = os.path.join(output_base, "test_processed1")
    else :
        output_dir = os.path.join(output_processed,"test")
    
    if os.path.exists(output_dir):
        print(f"Verzeichnis {output_dir} existiert bereits. Keine weiteren Operationen werden durchgeführt.")
        return output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Zähler für die Anzahl der verarbeiteten Bilder
    processed_count = 0

    for img_file in test_image_files:

        if processed_count >= 4:  # Abbrechen, wenn mehr als 5 Bilder verarbeitet wurden
            break

        img_path = os.path.join(test_image_folder, img_file)
        if not os.path.exists(img_path):
            print(f"Bilddatei nicht gefunden: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert('RGB')

            if downsample_factor is not None:
                img = downsample_image(img, downsample_factor)

            if patch_size:
                img_patches = extract_patches(img, patch_size, use_padding)
                for i, img_patch in enumerate(img_patches):
                    img_name = f"{os.path.splitext(img_file)[0]}_patch{i+1}.tif"
                    output_img_path = os.path.join(output_dir, img_name)
                    img_patch.save(output_img_path)
            else:
                img_name = f"{os.path.splitext(img_file)[0]}_binned.tif"
                output_img_path = os.path.join(output_dir, img_name)
                img.save(output_img_path)

            processed_count += 1  # Inkrementiere den Zähler

        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei {img_file}: {e}")

    print(f"Verarbeitung der Testbilder abgeschlossen. Ergebnisse gespeichert in: {output_dir}")
    return output_dir


if __name__ == '__main__':
     try:
        #Dataset Informations (root_dir, dataset_name)
        '''
        Default dataset_name = data/{dataset_name}
        '''
        root_dir= 'data/Dichtflächen_Cropped'
        dataset_name = 'Dichtflächen_Cropped'
        
        train_dir = process_images(dataset_name,downsample_factor=2,patch_size=192)
        #process_test_images(dataset_name,patch_size=192,downsample_factor=2,output_processed=None)

     except Exception as e:
        print(f"An error occurred: {e}")
