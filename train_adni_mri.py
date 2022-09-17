import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Sequence, Tuple
from cv2 import transform
from monai.transforms import RandAffine,ScaleIntensity,ResizeWithPadOrCrop

import h5py
import numpy as np
import torch
import random
import torchio as tio
import torchvision.transforms as transforms 
import monai.transforms
from preprocess import mean, std, preprocess_input_function
from torch import nn, sub
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

LOG = logging.getLogger(__name__)

DIAGNOSIS_CODES_MULTICLASS = {
    "CN": np.array(0, dtype=np.int64),
    "MCI": np.array(1, dtype=np.int64),
    "Dementia": np.array(2, dtype=np.int64),
}

DIAGNOSIS_CODES_BINARY = {
    "CN": np.array(0, dtype=np.int64),
    "Dementia": np.array(1, dtype=np.int64),
}


def get_mri_image_transform(is_training):
    # /1/MRI/T1/data           Dataset {113, 137, 113}
    img_transforms = []

    #img_transforms.append(ScaleIntensity(minv=0,maxv=1))
    img_transforms.append(tio.RescaleIntensity(out_min_max=(0, 1)))

    if is_training:
        img_transforms.append(
            tio.RandomAffine(
                scales=0.05,  # 95 - 105%
                degrees=12.5,  # +-12.5 degree in each dimension
                translation=12,  # +-12 pixels offset in each dimension. Default pad mode is 'otsu'
                image_interpolation="linear",
                p=0.5,
            )
        )

    img_transform = tio.Compose(img_transforms)
    return img_transform


# def get_pet_image_transform(is_training):
#     # /1/PET/FDG/data          Dataset {160, 160, 96}
#     img_transforms = []

#     img_transforms.append(tio.RescaleIntensity(out_min_max=(0, 1)))

#     if is_training:
#         img_transforms.append(
#             tio.RandomAffine(
#                 scales=0.05,  # 95 - 105%
#                 degrees=12.5,  # +-12.5 degree in each dimension
#                 translation=12,  # +-12 pixels offset in each dimension. Default pad mode is 'otsu'
#                 image_interpolation="linear",
#                 p=0.5,
#             )
#         )

    # img_transform = tio.Compose(img_transforms)
    # return img_transform


def get_image_transform(is_training):
    return get_mri_image_transform(is_training) #for notebook
    # return {
    #    "t1": get_mri_image_transform(is_training),
    #    "fdg": get_pet_image_transform(is_training),
    # }


class AdniDateset(Dataset):
    def __init__(self, filepaths, target_labels: Sequence[str], transform=None, target_transform=None, is_training=False) -> None:
        self.filepaths = filepaths
        self.target_labels = target_labels
        self.transform = transform
        self.target_transform = target_transform
        self.is_training = is_training
        self.indices = list()
        self.subjects = list()

        self._load()

    def _load(self):
        target_labels = self.target_labels
        data = []
        targets = {k: [] for k in target_labels}
        visits = []
        data_index = 0
        temp_subjects = dict()
        temp_img_indx = dict()
        for filename in self.filepaths: # added by Icxel
            with h5py.File(filename, mode='r') as fin: # added by Icxel
                for name, g in fin.items():
                    if name == "stats":
                        continue
                    visits.append((g.attrs["RID"], g.attrs["VISCODE"]))

                    mri = g["MRI"]["T1"]["data"][:]
                    #pet = g["PET"]["FDG"]["data"][:]

                    subject = torch.from_numpy(mri[np.newaxis].astype(np.float32)) # notebook
                    img_rescale = tio.CropOrPad((138, 138, 138))
                    subject = img_rescale(subject)

                    num_slides = 26
                    indx = subject.numpy().shape
                    indx = indx[-1]//2 - num_slides//2

                    for i in range(num_slides):
                        subject_to_append = subject[0,:,:,indx + i]
                        for label in target_labels:
                            target = g.attrs[label]
                            targets[label].append(target)
                        data.append(subject_to_append.unsqueeze(0))
                        temp_img_indx[data_index] = indx + i
                        temp_subjects[data_index] = subject
                        data_index = data_index + 1
                    assert(len(targets[target_labels[0]]) == len(data))
                
        index_list = list(range(len(data)))
        random.shuffle(index_list)
        new_data = list()
        new_targets = dict()
        new_targets[target_labels[0]] = list()
        for j,val in enumerate(index_list):
            new_data.insert(j, data[val])
            new_targets[target_labels[0]].insert(j, targets[target_labels[0]][val])
            self.indices.insert(j, temp_img_indx[val])
            self.subjects.insert(j, temp_subjects[val])

        self.data = new_data
        self.targets = new_targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        subject, indx = self.get_3d_img_to_transform(index)
        data_point = []
        
        if self.transform is not None:
            img = self.transform(subject)
        
        img_2d = img[0,:,:,indx]
        img_3ch = torch.from_numpy(np.stack((img_2d,)*3, axis=0))
        data_point.append(img_3ch)
       
        for label in self.target_labels:
            target = self.targets[label][index]
            if self.target_transform is not None:
                target = self.target_transform[label](target)
            data_point.append(target)

        return tuple(data_point)

    def get_3d_img_to_transform(self, index):
        return self.subjects[index], self.indices[index]

