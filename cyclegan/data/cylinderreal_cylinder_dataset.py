import random
import os
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
import torch
import numpy as np

from PIL import Image


CLASS_NUMBERS = {
    "top_left_INITIAL": 0,
    "top_left_FINAL": 1,
    "top_right_INITIAL": 2,
    "top_right_FINAL": 3,
    "bottom_left_INITIAL": 4,
    "bottom_left_FINAL": 5,
    "bottom_right_INITIAL": 6,
    "bottom_right_FINAL": 7,
}


def find_num(str):
    return [int(s) for s in str.split('_') if s.isdigit()][0]

def find_corner(str):
    split = str.split('_')
    return split[1] + '_' + split[2]

def get_label(corner, file_name, num_classes=8):
    if "INITIAL" in file_name:
        class_num = CLASS_NUMBERS.get(f"{corner}_INITIAL")
    else:
        class_num = CLASS_NUMBERS.get(f"{corner}_FINAL")
    label = np.zeros(num_classes)
    label[class_num] = 1
    return label


class CylinderRealCylinder(BaseDataset):
    def name(self):
        return 'CylinderRealCylinderDataset'  # long cylinder -> short cylinder

    def initialize(self, opt):
        # for mean and std, for now just load in initial and final states and calculate on that
        # MAKE SURE TO RECALCULATE mean and std after final data collection is done
        self.opt = opt
        # self.root = opt.dataroot
        self.root = '/ssd003/home/adilasif/adil_code/cycada/closest_corner_data'
        print(opt)
        self.A_dataset = []
        self.B_dataset = []
        self._create_dataset("cylinder_real", "source")
        self._create_dataset("cylinder", "target")

        # working with equal sized datasets for now (though we can just take min of either domain)
        # assert len(self.A_dataset) == len(self.B_dataset), "source and target domains are different sizes"

        random.shuffle(self.A_dataset)
        random.shuffle(self.B_dataset)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3792, 0.4544, 0.5703),
                                 (0.5020, 0.4922, 0.5133)),
            transforms.Resize((128, 128))
            ])
    
    def _create_dataset(self, end_effector, domain):
        successful_sims = {(find_num(str), find_corner(str)) for str in os.listdir(self.root + f'/{end_effector}/pickled_data') if "True" in str}

        corners = ["top_right", "top_left", "bottom_right", "bottom_left"]
        for corner in corners:
            for file_name in os.listdir(self.root + f'/{end_effector}/side/{corner}'):
                ep_num = int(file_name[file_name.find('_') + 1: file_name.find('_', file_name.find('_') + 1)])
                if (ep_num, corner) in successful_sims and ("INITIAL" in file_name or "FINAL" in file_name):
                    prefix = self.root + f'/{end_effector}/side/{corner}'
                    if domain == "target":
                        self.B_dataset.append(f'{prefix}/{file_name}')
                    elif domain == "source":
                        label = get_label(corner, file_name)
                        self.A_dataset.append((f'{prefix}/{file_name}', label))

    def __getitem__(self, index):

        A_img_path, A_label = self.A_dataset[index]
        A_img = Image.open(A_img_path).convert("RGB")

        A_img = self.transform(A_img)
        A_path = '%01d_%05d.png' % (np.argwhere(A_label)[0][0], index)

        B_img_path = self.B_dataset[index]
        B_img = Image.open(B_img_path).convert("RGB")
        B_img = self.transform(B_img)

        item = {}
        item.update({'A': A_img,
                     'A_paths': A_path,
                     'A_label': A_label
                 })
        
        item.update({'B': B_img})
        return item
        
    def __len__(self):
        return min(len(self.A_dataset), len(self.B_dataset))

# dataset = CylinderRealCylinder()
# dataset.initialize(None)
# from torch.utils.data import DataLoader

# dataloader = DataLoader(dataset, batch_size=64)
# channels_sum, channels_squared_sum, num_batches = 0, 0, 0
# for i, temp in enumerate(dataloader):
#     stacked = torch.cat([temp["A"], temp["B"]])
#     print(stacked.shape)
#     channels_sum += torch.mean(stacked, dim=[0,2,3])
#     channels_squared_sum += torch.mean(stacked**2, dim=[0,2,3])
#     num_batches += 1
# mean = channels_sum / num_batches

# # std = sqrt(E[X^2] - (E[X])^2)
# std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

# print(mean)
# print(std)