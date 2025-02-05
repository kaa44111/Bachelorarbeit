import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from PIL import Image

def masks_to_colorimg(masks):
    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56)])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)
            
    return colorimg.astype(np.uint8)

def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


def show_masks_pred(mask, pred):
    # Wählen Sie den ersten Batch aus und konvertieren Sie ihn in NumPy (auf CPU kopieren)
    true_masks = mask[0].cpu().numpy()
    pred_masks = pred[0].cpu().numpy()

    # Anzahl der Bilder
    num_images = true_masks.shape[0]

    fig, axes = plt.subplots(2, num_images, figsize=(15, 5))

    for i in range(num_images):
        image = true_masks[i]
        axes[0, i].imshow(image, cmap='gray')
        axes[0, i].axis('off')

    for i in range(num_images):
        image = pred_masks[i]
        sns.heatmap(image, ax=axes[1, i], cmap='viridis', cbar=True)
        axes[1, i].axis('off')

    plt.show()

def show_masks_pred1(mask, pred):
    # Wählen Sie den ersten Batch aus und konvertieren Sie ihn in NumPy (auf CPU kopieren)
    true_masks = mask[0].cpu().numpy()
    pred_masks = pred[0].cpu().numpy()

    
    image = true_masks[0]
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    image = pred_masks[0]
    sns.heatmap(image, cmap='viridis', cbar=True)
    plt.axis('off')

    plt.show()

def save_valuation(inputs, labels, pred):
        # Change channel-order and make 3 channels for matplot
    input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]
    
    i = 0
    for imgs in input_images_rgb:
        i += 1
        im = Image.fromarray(imgs)
        im.save("data\\geometry_shapes\\validate\\grabs\\" + str(i) + ".png")


    # Map each channel (i.e. class) to each color
    target_masks_rgb = [masks_to_colorimg(x) for x in labels.cpu().numpy()]
    
    i = 0
    for imgsM in target_masks_rgb:
        i += 1
        im = Image.fromarray(imgsM)
        im.save("data\\geometry_shapes\\validate\\masks\\" + str(i) + ".png")
    
    
    
    pred_rgb = [masks_to_colorimg(x) for x in pred]
    
    i = 0
    for prd in pred_rgb:
        i += 1
        im = Image.fromarray(prd)
        im.save("data\\geometry_shapes\\validate\\pred\\" + str(i) + ".png")