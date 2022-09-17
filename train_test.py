import os
import shutil

# with open('shape_continuum/ProtoPNet/CUB_200_2011/CUB_200_2011/train_test_split.txt') as f:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
#     lines = f.readlines()
#     is_training = dict()
#     for l in lines:
#         info = l.split()
#         is_training[info[0]] = int(info[1])
    
#     direc_str = 'shape_continuum/ProtoPNet/datasets/cub200_cropped/images/'
#     direc = os.fsencode(direc_str)
    
#     for folder in os.listdir(direc):
#         folder_name = os.fsdecode(folder)
#         dirStr = os.fsencode(direc_str + folder_name + '/')
#         for file in os.listdir(dirStr):
#             filename = os.fsdecode(file)
#             id = filename.split('_')[0]
#             if is_training[id]:
#                 target_path = os.path.join('shape_continuum/ProtoPNet/datasets/cub200_cropped/train_cropped', folder_name)
#                 if not os.path.exists(target_path):
#                     os.makedirs(target_path)
#                 shutil.copy(direc_str + folder_name + '/' + filename, target_path + '/' + filename)     
#             else:
#                 target_path = os.path.join('shape_continuum/ProtoPNet/datasets/cub200_cropped/test_cropped', folder_name)
#                 if not os.path.exists(target_path):
#                     os.makedirs(target_path)
#                 shutil.copy(direc_str + folder_name + '/' + filename, target_path + '/' + filename)

# direc_str = 'shape_continuum/ProtoPNet/images'
# cropped_str1 = 'shape_continuum/ProtoPNet/datasets/cub200_cropped/test_cropped'
# cropped_str2 = 'shape_continuum/ProtoPNet/datasets/cub200_cropped/train_cropped'
# direc = os.fsencode(direc_str)
# cropped_dir1 = os.fsencode(cropped_str1)
# cropped_dir2 = os.fsencode(cropped_str2)

# for file in os.listdir(direc):
#     found = False
#     filename = os.fsdecode(file)
#     for f in os.listdir(cropped_dir1):
#         fname = os.fsdecode(f)
#         if fname == filename:
#             found = True
#             os.remove(direc_str + '/' + filename)
#             break
#     if(found is False):
#         for f2 in os.listdir(cropped_dir2):
#             f2name = os.fsdecode(f2)
#             if f2name == filename:
#                 found = True
#                 os.remove(direc_str + '/' + filename)
#                 break

# for folder in os.listdir('shape_continuum/ProtoPNet/datasets/cub200_cropped/train_cropped/'):
#     num = int(folder.split('.')[0])
#     if num >  14:
#         shutil.move('shape_continuum/ProtoPNet/datasets/cub200_cropped/train_cropped/' + folder + '/' + folder[4:], 'shape_continuum/ProtoPNet/datasets/cub200_cropped/train_cropped_augmented/')


# for folder in os.listdir('shape_continuum/ProtoPNet/datasets/cub200_cropped/train_cropped/'):
#      num = int(folder.split('.')[0])
#      if num >  14 and num < 192:
#          for folder2 in os.listdir('shape_continuum/ProtoPNet/datasets/cub200_cropped/train_cropped_augmented/'):
#              if folder2 == folder.strip().split('.')[1]:
#                  os.rename('shape_continuum/ProtoPNet/datasets/cub200_cropped/train_cropped_augmented/' + folder2, 'shape_continuum/ProtoPNet/datasets/cub200_cropped/train_cropped_augmented/' + folder)
import h5py
import numpy as np
import cv2
import torch
import torchio as tio
from torchvision.utils import save_image
from train_adni_multimodal import get_image_transform


## SAVING TESTING IMAGES
DIAGNOSIS_CODES_BINARY = {
    "CN": np.array(0, dtype=np.int64),
    "Dementia": np.array(1, dtype=np.int64),
}

target_labels = ["DX"]
target_transform_map = DIAGNOSIS_CODES_BINARY
target_transform = {"DX": lambda x: target_transform_map[x]}
targets = {k: [] for k in target_labels}

img_transform = get_image_transform(False)
cropOrPadMri = tio.CropOrPad((138,138,138))

images = dict()


with h5py.File('/mnt/nas/Users/Sebastian/adni-mri-pet/classification-nomci/mri-pet/1-test.h5', mode="r") as fin:
    images_count = 0
    subjects_count = 0
    for name, g in fin.items():
        if name == "stats":
            continue
        if subjects_count > 5:
            break
        #visits.append((g.attrs["RID"], g.attrs["VISCODE"]))
        for label in target_labels:
            target = g.attrs[label]
            targets[label].append(target)

        mri = g["MRI"]["T1"]["data"][:]
        mri = torch.from_numpy(mri[np.newaxis].astype(np.float32))
        print(mri.size())
        mri = cropOrPad(mri)
        print(mri.size())
        mri = img_transform(mri)
        print(mri.size())

        num_slides = 26
        indx = mri.numpy().shape
        indx = indx[-1]//2 - num_slides//2

        for i in range(num_slides):
            print(mri.size())
            new_mri = mri[0,:,:,indx + i]
            image_numpy = new_mri.numpy() # added Icxel
            rgb_img = torch.from_numpy(np.stack((image_numpy,), axis=0))
            for label in target_labels:
                target = g.attrs[label]
            image_name = "test_image" + str(images_count) + "_" + target + ".png"
            save_image(rgb_img, '/home/icxel/shape_continuum/ProtoPNet/testing_images/' + image_name)
            images[image_name] = target
            images_count = images_count + 1
        subjects_count = subjects_count + 1    


# print(images)


## COUNTING DATASET CLASS INSTANCES

# with h5py.File('/mnt/nas/Users/Sebastian/adni-mri-pet/registered/classification-nomci/mri-pet/1-train.h5', mode="r") as fin:
#     images_count = 0
#     healthy_pts = 0
#     alzheimer_pts = 0
#     for name, g in fin.items():
#         if name == "stats":
#             continue
#         #visits.append((g.attrs["RID"], g.attrs["VISCODE"]))
#         for label in target_labels:
#             target = g.attrs[label]
#             if target == 'CN':
#                 healthy_pts = healthy_pts + 1
#             elif target == 'Dementia':
#                 alzheimer_pts = alzheimer_pts + 1
# print("train healthy ", healthy_pts)
# print("train alzheimers ", alzheimer_pts)

# with h5py.File('/mnt/nas/Users/Sebastian/adni-mri-pet/registered/classification-nomci/mri-pet/1-valid.h5', mode="r") as fin:
#     images_count = 0
#     healthy_pts = 0
#     alzheimer_pts = 0
#     for name, g in fin.items():
#         if name == "stats":
#             continue
#         #visits.append((g.attrs["RID"], g.attrs["VISCODE"]))
#         for label in target_labels:
#             target = g.attrs[label]
#             if target == 'CN':
#                 healthy_pts = healthy_pts + 1
#             elif target == 'Dementia':
#                 alzheimer_pts = alzheimer_pts + 1
# print("valid healthy ", healthy_pts)
# print("valid alzheimers ", alzheimer_pts)