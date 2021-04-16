import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import random_split, DataLoader

import pickle 

import os

class CIFAR10DataPrep():
    
    def __init__(self, transformer_config = None):
        self.transformer_config = transformer_config
        self.stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.img_size = 32
        self.img_padding = 4
        self.val_data = int(0.01 * 50000)
        
        # Path parameters
        self.data_dir = transformer_config['data_dir']
        self.dataloader_dir = transformer_config['dataloader_dir']#'./data/dataloader'
        self.batch_size = transformer_config['batch_size']
        
        # Define transformers to nomalize data
        self.train_normalizer = transforms.Compose([
            transforms.RandomCrop(self.img_size, padding=self.img_padding, padding_mode='reflect'), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize(*self.stats,inplace=True)])
        
        self.test_normalizer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*self.stats,inplace=True)])
        
    def prepare_pipeline_dataloader(self, pipeline_size = 200):
        train_val_set = datasets.CIFAR10(root=self.data_dir, train=True, download=False, transform=self.train_normalizer)
        to_split, _ = random_split(train_val_set, [pipeline_size, len(train_val_set) - pipeline_size])
        train_set, val_set = random_split(to_split, [int(0.9*pipeline_size), int(0.1*pipeline_size)])
        
        test_set = datasets.CIFAR10(root=self.data_dir, train=False, download=False, transform=self.test_normalizer)
        test_to_split, _ = random_split(test_set, [100, len(test_set) - 100])
        
        train_dataloader = DataLoader(train_set, batch_size = self.batch_size, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_set, batch_size = self.batch_size, shuffle=True, num_workers=0)
        test_dataloader = DataLoader(test_set, batch_size = self.batch_size, shuffle=False, num_workers=0)
        
        self._save_dataloader(train_dataloader, "pip_train_dataloader.pickle")
        self._save_dataloader(val_dataloader, "pip_val_dataloader.pickle")
        self._save_dataloader(test_dataloader, "pip_test_dataloader.pickle")
        
    def prepare_dataloader(self):
        
        ##### data set and loader creating block #####
        train_val_set = datasets.CIFAR10(root=self.data_dir, train=True, download=False, transform=self.train_normalizer)
        
        train_set, val_set = random_split(train_val_set, [len(train_val_set) - self.val_data, self.val_data])
        
        test_set = datasets.CIFAR10(root=self.data_dir, train=False, download=False, transform=self.test_normalizer)

        train_dataloader = DataLoader(train_set, batch_size = self.batch_size, shuffle=True, num_workers=0)
        
        val_dataloader = DataLoader(val_set, batch_size = self.batch_size, shuffle=True, num_workers=0)
        
        test_dataloader = DataLoader(test_set, batch_size = self.batch_size, shuffle=False, num_workers=0)
        
        self._save_dataloader(train_dataloader, "train_dataloader.pickle")
        self._save_dataloader(val_dataloader, "val_dataloader.pickle")
        self._save_dataloader(test_dataloader, "test_dataloader.pickle")
        
    def _save_dataloader(self, dataloader, save_name):
        ##### saving block #####
        if not os.path.isdir(self.dataloader_dir):
            os.mkdir(self.dataloader_dir)
        
        pickle.dump(
            dataloader,
            open(os.path.join(self.dataloader_dir, save_name), "wb"),
            protocol=4,
        )




