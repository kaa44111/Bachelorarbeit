import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import torch
import torch.utils
from torchvision.transforms import v2

#from datasets.MultipleFeature import get_data_loaders
from datasets.OneFeature import get_dataloaders

#Train
from train.train import run
from train.train_compare import run_compare

#Test
from test_models.ImageOnly_test import test
from test_models.test_different_models import test_compare

#Falls man die Bilder Normalisiern will
from utils.data_utils import compute_mean_std

#Modelle:
from models.UNet import UNet
from models.UNetBatchNorm import UNetBatchNorm
from models.UNetNoMaxPool import UNetMaxPool


'''
Den Teil auskommentiern den man grad ausführen möchte.

Parameteren nach belieben anpassen:
- Modell : UNet, UNetBatchNorm, UNetMaxPool

- Transformationen:
    - wenn nichts angegeben dann werden die Original Bilder antrainiert.

- save_name: Name der gespeicherten Modell Ergebnisse
    - Default: 'train/results/RetinaVessel/test_1.pth'

- batch_size
'''

if __name__ == '__main__':
     try:
        batch_size = 15
        save_name = 'test_1'
        
        #mean, std = compute_mean_std(os.path.join(root_dir, 'grabs'))
        transformations = v2.Compose([
            v2.RandomEqualize(p=1.0),
            v2.ToPureTensor(),
            v2.ToDtype(torch.float32, scale=True),
            #v2.Normalize(mean=mean, std=std),
        ])

        train_dir= 'data_modified/RetinaVessel/train'
        test_dir='data_modified/RetinaVessel/test'
        dataset_name = 'RetinaVessel'

        dataloader,_ = get_dataloaders(root_dir=train_dir, dataset_name=dataset_name, batch_size=batch_size, transformations=transformations)
        #_____________________________________________________________

        # ####Training für ein Modell Starten
        # print("Train Model with RetinaVessel Dataset:")
        # run(UNet,dataloader,dataset_name,save_name)

        # results_dir = os.path.join('train/results',dataset_name)
        # trained_model = f"{results_dir}/{save_name}.pth"

        # ####Testen für das antrainerte Modell Starten
        # print("Test Results:")
        # test(UNet=UNet,test_dir=test_dir,trained_path=trained_model)

        #______________________________________________________________

        #####Training für alle Modelle Starten
        run_compare(dataloader,dataset_name)

        #####Test für alle trainerte Modelle
        test_compare(test_dir, dataset_name, UNet, UNetMaxPool, UNetBatchNorm)
        
     except Exception as e:
        print(f"An error occurred: {e}")

#_________________________________

# test_different_models.py Ergebnisse:

# UNet Best val loss: 0.628031
# UNetMaxPool Best val loss: 0.633133
# UNetBatchNorm Best val loss: 0.467543


# UNet training time: 4m 18s
# UNetMaxPool training time: 1m 12s
# UNetBatchNorm training time: 20m 8s


# UNet inference time: 0.0410 seconds
# UNetMaxPool inference time: 0.0428 seconds
# UNetBatchNorm inference time: 0.0566 seconds


# UNet parameters: 31031745
# UNetMaxPool parameters: 31031745
# UNetBatchNorm parameters: 31043521

#_________________________________

# test_different_models.py Ergebnisse:
    
# UNet training time: 3.34 min
# UNetMaxPool training time: 0.99 min
# UNetBatchNorm training time: 19.07 min


# UNet inference time: 0.0181 seconds
# UNetMaxPool inference time: 0.0266 seconds
# UNetBatchNorm inference time: 0.0298 seconds


# UNet parameters: 31031745
# UNetMaxPool parameters: 31031745
# UNetBatchNorm parameters: 31043521