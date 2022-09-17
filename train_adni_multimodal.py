import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import random
import torchio as tio
import torchvision.transforms as transforms 
import monai.transforms
from preprocess_multimodal import mean, std, preprocess_input_function
from settings_multimodal import img_size_mri, img_size_pet
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


def get_pet_image_transform(is_training):
    # /1/PET/FDG/data          Dataset {160, 160, 96}
    img_transforms = []

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


def get_image_transform(is_training):
    #return get_mri_image_transform(is_training) #for notebook
    return {
        "t1": get_mri_image_transform(is_training),
        "fdg": get_pet_image_transform(is_training),
     }


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
                    if name == "stats" or g["PET"]["FDG"].attrs["imageuid"] == "1074115":
                        print("found!")
                        continue
                    visits.append((g.attrs["RID"], g.attrs["VISCODE"]))
                    mri = g["MRI"]["T1"]["data"][:]
                    pet = g["PET"]["FDG"]["data"][:]

                    subject = OrderedDict([
                        ("t1", torch.from_numpy(mri[np.newaxis].astype(np.float32))), # only gray matter
                        ("fdg", torch.from_numpy(pet[np.newaxis].astype(np.float32))), # blurry
                    ])

                    subject_to_append = dict()
                    img_rescale_mri = tio.CropOrPad((img_size_mri, img_size_mri, img_size_mri))
                    img_rescale_pet = tio.CropOrPad((img_size_pet, img_size_pet, img_size_pet))
                    for key, img in subject.items():
                        if key == "t1":
                            if img_rescale_mri is not None:
                                img = img_rescale_mri(img)
                        elif key == "fdg":
                            if img_rescale_pet is not None:
                                img = img_rescale_pet(img)
                        subject[key] = img
                    
                    key1= "t1"
                    key2 = "fdg"
                    num_slides = 22
                    indx1 = subject[key1].numpy().shape
                    indx1 = indx1[-1]//2 - num_slides//2

                    indx2 = subject[key2].numpy().shape
                    indx2 = indx2[-1]//2 - num_slides//2

                    for i in range(num_slides):
                        subject_to_append[key1] = subject[key1][0,:,:,indx1 + i]
                        subject_to_append[key2] = subject[key2][0,:,:,indx2 + i]
                        for label in target_labels:
                            target = g.attrs[label]
                            targets[label].append(target)
                        final_subject = OrderedDict([
                            (key1, subject_to_append[key1].unsqueeze(0)),
                            (key2, subject_to_append[key2].unsqueeze(0))
                        ])

                        data.append(final_subject)
                        temp_img_indx[data_index] = indx1 + i
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
       
        for key, img in subject.items():
            if self.transform is not None and key in self.transform:
                img = self.transform[key](img)
            img_2d = img[0,:,:,indx]
            img_3ch = torch.from_numpy(np.stack((img_2d,)*3, axis=0).astype(np.float32))
            data_point.append(img_3ch)

        for label in self.target_labels:
            target = self.targets[label][index]
            if self.target_transform is not None:
                target = self.target_transform[label](target)
            data_point.append(target)

        return tuple(data_point)

    def get_3d_img_to_transform(self, index):
        return self.subjects[index], self.indices[index]