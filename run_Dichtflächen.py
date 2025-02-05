import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import torch
import torch.utils
from torchvision.transforms import v2
from torchvision import transforms

#from datasets.MultipleFeature import get_data_loaders
from datasets.OneFeature import get_dataloaders

#Prepare
from prepare.prepare_both import process_images
from prepare.bounding_box import process_and_save_cropped_images

# #Train
# from train.train import run
# from train.train_compare import run_compare

# #Test
# from test_models.test_model import test
# from test_models.test_different_models import test_compare

# #Falls man die Bilder Normalisiern will
# from utils.data_utils import compute_mean_std, show_image_and_mask

#Modelle:
from models.UNet import UNet
# from models.UNetBatchNorm import UNetBatchNorm
# from models.UNetNoMaxPool import UNetNoMaxPool


if __name__ == '__main__':
     try:
        #Dataset Informations (root_dir, dataset_name)
        '''
        Default dataset_name = data/{dataset_name}
        '''
        root_dir= 'data/Dichtflächen'
        dataset_name = 'Dichtflächen'
        #save_name = 'test'

        #Prepare Dataset (downsampe, batch, both)
        '''
        Default downsample : scale_factor = 2
        Default patch:  patch_size= 200
        '''
        train_dir = process_images(dataset_name,patch_size=192)

        #Get Dataloader
        '''
        Default Transform: ToPureTensor(), ToDtype(torch.float32, scale=True)
        Default batch_size : 15
        Default split_size : 0.8
        '''
        trans = v2.Compose([
            # v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            # v2.ToDtype(torch.uint8, scale=True),
            v2.ToPureTensor(),
            v2.ToDtype(torch.float32, scale=True),

        ])
        dataloader,custom_dataset = get_dataloaders(root_dir=train_dir,dataset_name=dataset_name,batch_size=20,transformations=trans) 

        image, mask =custom_dataset[0]
        print('Image:')
        print(image.shape)
        print(image.min(), image.max())
        print('Mask:')
        print(mask.shape)
        print(mask.min(), mask.max())

        #_____________________________________________________________

        # ####Training für ein Modell Starten
        # print("Train Model with Dichtflächen Dataset:")
        # run(UNet, dataloader, dataset_name)
        
        # save_name = 'test_1s'
        # results_dir = os.path.join('train/results',dataset_name)
        # trained_model = f"{results_dir}/{save_name}.pth"
        
        # test(UNet=UNet,test_dir=train_dir,transformations = trans,test_trained_model=trained_model)

        # #______________________________________________________________

        # #####Training für alle Modelle Starten
        # run_compare(dataloader,dataset_name)

        # #####Test für alle trainerte Modelle
        # test_compare(test_dir, dataset_name, UNet, UNetMaxPool, UNetBatchNorm)
        
     except Exception as e:
        print(f"An error occurred: {e}")
