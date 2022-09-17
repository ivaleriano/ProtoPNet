from collections import OrderedDict
from torch.utils.data import Dataset
from typing import Sequence
from wrappers_multimodal import NamedDataLoader, mesh_collate
from torchvision.utils import save_image
from log import create_logger
import numpy as np
import torchio as tio
import h5py
import torch
import random
import argparse
import os
import shutil

from local_analysis_onemodal_aopc import LocalAnalysis
#from settings import img_size

DIAGNOSIS_CODES_BINARY = {
    "CN": np.array(0, dtype=np.int64),
    "Dementia": np.array(1, dtype=np.int64),
}

def get_image_transform():
    # /1/MRI/T1/data           Dataset {113, 137, 113}
    img_transforms = []
    img_transforms.append(tio.RescaleIntensity(out_min_max=(0, 1)))
    img_transform = tio.Compose(img_transforms)
    return img_transform


transform = get_image_transform()
target_labels = ["DX"]
target_transform_map = DIAGNOSIS_CODES_BINARY
target_transform = {"DX": lambda x: target_transform_map[x]}
indices = list()
subjects = list()
    

data = []
targets = {k: [] for k in target_labels}
visits = []
data_index = 0
data_index_cn = 0
data_index_ad = 0
temp_subjects = dict()
temp_img_indx = dict()
subject_nums = []
with h5py.File('/mnt/nas/Users/Sebastian/adni-mri-pet/registered/classification-nomci/mri-pet/1-train.h5', mode='r') as fin:
    for name, g in fin.items():
        if name == "stats" or g["PET"]["FDG"].attrs["imageuid"] == "1074115":
            print("found!")
            continue


        for label in target_labels:
                target = g.attrs[label]
        if target == 'CN':
            data_index_cn = data_index_cn + 1
        else:
            data_index_ad = data_index_ad + 1
        
        data_index = data_index + 1

        visits.append((g.attrs["RID"], g.attrs["VISCODE"]))
        mri = g["MRI"]["T1"]["data"][:]
        pet = g["PET"]["FDG"]["data"][:]
        subject_num = g["PET"]["FDG"].attrs["imageuid"]

        subject_to_append = dict()
        subject_mri = torch.from_numpy(mri[np.newaxis].astype(np.float32)) # only gray matter
        subject_pet = torch.from_numpy(pet[np.newaxis].astype(np.float32)) # blurry
                
        img_rescale_mri = tio.CropOrPad((138, 138, 138))
        img_rescale_pet = tio.CropOrPad((130, 130, 130))
        subject_mri = img_rescale_mri(subject_mri)
        subject_mri = transform(subject_mri)
        subject_pet = img_rescale_pet(subject_pet)
        subject_pet = transform(subject_pet)
        
        num_slides = 22
        indx = subject_mri.numpy().shape
        indx = indx[-1]//2 - num_slides//2

        '''for i in range(num_slides):
            subject_to_save_mri = subject_mri[0,:,:,indx + i]
            subject_to_save_pet = subject_pet[0,:,:,indx + i]
            for label in target_labels:
                target = g.attrs[label]
            mri_img_numpy = subject_to_save_mri.numpy()
            pet_img_numpy = subject_to_save_pet.numpy()
            img_3ch_mri =  torch.from_numpy(np.stack((mri_img_numpy,), axis=0))
            img_3ch_pet =  torch.from_numpy(np.stack((pet_img_numpy,), axis=0))
            image_name_mri = "example_image" + str(i) + "_" + target + "_mri_" + subject_num
            image_name_pet = "example_image" + str(i) + "_" + target + "_pet_" + subject_num
            save_image(img_3ch_mri, '/home/icxel/shape_continuum/ProtoPNet/testing_images/' + image_name_mri + ".png")
            save_image(img_3ch_pet, '/home/icxel/shape_continuum/ProtoPNet/testing_images/' + image_name_pet + ".png")
        data_index = data_index + 1'''
    print("Total subjects ", data_index)
    print("Total CN", data_index_cn)
    print("Total AD", data_index_ad)