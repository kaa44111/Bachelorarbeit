import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.utils
from collections import defaultdict
import time
import copy
from PIL import Image
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    # Verwende Binary Cross Entropy (BCE) Loss
    bce = F.binary_cross_entropy_with_logits(pred, target)

    # Wende Sigmoid an, um die Ausgaben in den Bereich [0,1] zu skalieren
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    # Kombiniere BCE und Dice Loss
    loss = bce * bce_weight + dice * (1 - bce_weight)

    # Metriken für Monitoring aktualisieren
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))



def train_model(model,dataloaders, optimizer, scheduler, num_epochs=25,save_name=None):
    # TensorBoard Writer initialisieren
    writer = SummaryWriter(log_dir=f'runs/{save_name}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_loss = 1e10
    #scaler = GradScaler()  # GradScaler initialisieren

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    train_losses = []
    val_losses = []

    # Each epoch has a training and validation phase
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # # Debugging-Ausgaben
                # print(f"Inputs shape: {inputs.shape}, dtype: {inputs.dtype}")
                # print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
                # print(f"Max input value: {inputs.max()}, Min input value: {inputs.min()}")
                # print(f"Max label value: {labels.max()}, Min label value: {labels.min()}")

                # # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):

                    #save_images_und_masks(inputs, labels)

                    # with autocast():  # Autocast für automatische Mischpräzision
                    #     outputs = model(inputs)
                    #     loss = calc_loss(outputs, labels, metrics)

                    # print(f"Outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
                    # print(f"Max output value: {outputs.max()}, Min output value: {outputs.min()}")

                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # scaler.scale(loss).backward()  # Scale den Verlust und berechne die Gradienten
                        # scaler.step(optimizer)  # Optimierungsschritt mit Scaler
                        # scaler.update()  # Scaler aktualisieren

                # statistics
                epoch_samples += inputs.size(0)

            #print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'train':
                train_losses.append(epoch_loss)
                writer.add_scalar('Loss/train', epoch_loss, epoch)
            else:
                val_losses.append(epoch_loss)
                writer.add_scalar('Loss/val', epoch_loss, epoch)

            print_metrics(metrics, epoch_samples, phase)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

         # Lernrate protokollieren
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)

         # Optional: Modellgraph nach dem ersten Forward-Pass protokollieren
        if epoch == 0:
            writer.add_graph(model, inputs)

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Diagramm des Trainingsverlaufs speichern
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(f"{save_name}.png")
    plt.show()

    writer.add_figure('Loss Curve', plt.gcf())
    writer.close()

    return model

def select_scheduler(optimizer, scheduler_type, **kwargs):
    if scheduler_type == 'StepLR':
        return lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type == 'MultiStepLR':
        return lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif scheduler_type == 'ExponentialLR':
        return lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_type == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def run(UNet,dataloader,dataset_name,save_name=None):
    num_class = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(num_class).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    scheduler_type = 'StepLR'
    scheduler_params = {'step_size': 7, 'gamma': 0.1}

    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    # Select scheduler
    exp_lr_scheduler = select_scheduler(optimizer_ft, scheduler_type, **scheduler_params)

    start = time.time()
    model = train_model(model,dataloader, optimizer_ft, exp_lr_scheduler, num_epochs=30,save_name=save_name)
    end = time.time()

    # Verstrichene Zeit in Sekunden
    elapsed_time_s = end - start
    elapsed_time_min = elapsed_time_s / 60

    print(f"Elapsed time: {elapsed_time_min:.2f} minutes")
    print("\n")

    if save_name is None:
        save_name = 'test_1s'

    results_dir = os.path.join('train/results',dataset_name)
    save_dir = f"{results_dir}/{save_name}_{scheduler_type}.pth"

    # Speichern des trainierten Modells
    torch.save(model.state_dict(), save_dir)
    print(f"Model saved to {save_dir}")
    

# if __name__ == '__main__':
#      try:
        
#      except Exception as e:
#         print(f"An error occurred: {e}")

############
# Geometry_dataset
# num_class = 6
# epochs = 30
# Images as Grey value : 10.24 minutes
#___________________________
# Images as RGB : 12.18 minutes

###########
# Geometry_dataset
# num_class = 6
# epochs = 75
# Images as Grey value : 25.61 minutes
# LR 1.0000000000000002e-06
# Best val loss: 0.118130
#_________________________
# Images as RGB : 35.16 minutes
# LR 1.0000000000000002e-06
# Best val loss: 0.147360

############
# num_class = 6
# device = gpu
# lr=1e-4
# epochs=35
# batchsize=30
# trainset = 100, valset = 20