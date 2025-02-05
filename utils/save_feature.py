import os
from PIL import Image
import shutil
import numpy as np

def copy_feature_masks(src_folder, dest_folder, file_suffix):
    """
    Kopiert alle Maskenbilder, die das spezifische Feature enthalten, in einen neuen Ordner.

    :param src_folder: Der Quellordner, der die Maskenbilder enthält.
    :param dest_folder: Der Zielordner, in den die ausgewählten Maskenbilder kopiert werden.
    :param feature_label: Das Label des Features, nach dem gesucht werden soll.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Liste aller Masken im Quellordner
    mask_files = os.listdir(src_folder)

    for mask_file in mask_files:
        if mask_file.endswith(file_suffix + '.png'):  # Anpassung je nach Dateiendung
            mask_path = os.path.join(src_folder, mask_file)
            shutil.copy(mask_path, dest_folder)
            print(f"Copied {mask_file} to {dest_folder}")

# Ordnerpfade und Feature-Label
src_folder = 'data/geometry_shapes/masks'
dest_folder = 'data/geometry_shapes/mask_circle'
file_suffix = '1'  # Ersetzen Sie dies durch das tatsächliche Label des Features

# copy_feature_masks(src_folder, dest_folder, file_suffix)