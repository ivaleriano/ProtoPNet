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
    #img_transforms.append(tio.CropOrPad((138, 138, 138)))
    #img_transforms.append(tio.CropOrPad((224, 224, 224)))

    if is_training:
        # img_transforms.append(
        #     RandAffine(
        #         prob=0.5,
        #         rotate_range=(0.218,0.218,0.218), # 12 degrees in radians
        #         translate_range=(12,12,12),
        #         scale_range=(0.05,0.05,0.05),
        #         mode='bilinear'
        #     )
        # )
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
    #img_transform = monai.transforms.Compose(img_transforms)
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
    #def __init__(self, filename: Path, target_labels: Sequence[str], transform=None, target_transform=None, is_training=False) -> None:
    def __init__(self, filepaths, target_labels: Sequence[str], transform=None, target_transform=None, is_training=False) -> None:
        #self.filename = filename
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
        #with h5py.File(self.filename, mode="r") as fin:
            with h5py.File(filename, mode='r') as fin: # added by Icxel
                for name, g in fin.items():
                    if name == "stats":
                        continue
                    visits.append((g.attrs["RID"], g.attrs["VISCODE"]))
                    
                    # for label in target_labels:
                    #     target = g.attrs[label]
                    #     targets[label].append(target)

                    mri = g["MRI"]["T1"]["data"][:]
                    #pet = g["PET"]["FDG"]["data"][:]

                    subject = torch.from_numpy(mri[np.newaxis].astype(np.float32)) # notebook
                    #data.append(subject)
                    #if self.transform is not None:
                    #   subject = self.transform(subject)
                    #img_rescale = ResizeWithPadOrCrop((138,138,138),mode='constant')
                    img_rescale = tio.CropOrPad((138, 138, 138))
                    subject = img_rescale(subject)

                    num_slides = 26
                    indx = subject.numpy().shape
                    indx = indx[-1]//2 - num_slides//2

                    for i in range(num_slides):
                        subject_to_append = subject[0,:,:,indx + i]
                        for label in target_labels:
                            target = g.attrs[label]
                            #if self.target_transform is not None:
                            #   target = self.target_transform[label](target)
                            targets[label].append(target)
                        data.append(subject_to_append.unsqueeze(0))
                        temp_img_indx[data_index] = indx + i
                        temp_subjects[data_index] = subject
                        data_index = data_index + 1
                    assert(len(targets[target_labels[0]]) == len(data))
                # subject = OrderedDict([
                #     ("t1", torch.from_numpy(mri[np.newaxis].astype(np.float32))), # only gray matter
                #     ("fdg", torch.from_numpy(pet[np.newaxis].astype(np.float32))), # blurry
                # ])
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
        # else:
        #     self.data = data
        #     self.targets = targets
        # self.visits = visits

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        #subject = self.data[index]
        subject, indx = self.get_3d_img_to_transform(index)
        data_point = []
        #for key, img in subject.items():
            #if self.transform is not None and key in self.transform:
                #img = self.transform[key](img)
        #if self.transform is not None:
        #    img = self.transform(subject.unsqueeze(-1))
        if self.transform is not None:
            img = self.transform(subject)
        # forIndx = img
        # indx_middle = forIndx.numpy().shape[-1]//2
        # indx = random.randint(0, 20)
        # add_or_minus = random.randint(0,1)
        # if(add_or_minus==1):
        #     img_2d = img[0,:,:,indx_middle + indx]
        # else:
        #     img_2d = img[0,:,:,indx_middle - indx]
        img_2d = img[0,:,:,indx]
        img_3ch = torch.from_numpy(np.stack((img_2d,)*3, axis=0))
        data_point.append(img_3ch)
       # data_point.append(img_2d.unsqueeze(0))
        # print(img.size())
        #img = img[:,:,:,0]
        #img = img.unsqueeze(0)
        #data_point.append(img)
        #data_point.append(subject)
        for label in self.target_labels:
            target = self.targets[label][index]
            if self.target_transform is not None:
                target = self.target_transform[label](target)
            data_point.append(target)

        return tuple(data_point)

    def get_3d_img_to_transform(self, index):
        return self.subjects[index], self.indices[index]

# class ResNet(BaseModel):
#     def __init__(self, in_channels=1, bn_momentum=0.05, n_basefilters=32):
#         super().__init__()
#         self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum, kernel_size=5)
#         self.pool1 = nn.MaxPool3d(2, stride=2)  # downsample
#         self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
#         self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # downsample
#         self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # downsample
#         self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # downsample
#         self.global_pool = nn.AdaptiveAvgPool3d(1)
#         # self.dropout = nn.Dropout(p=0.4)

#     @property
#     def input_names(self) -> Sequence[str]:
#         return ("image",)

#     @property
#     def output_names(self) -> Sequence[str]:
#         return ("feature_map",)

#     def forward(self, image):
#         out = self.conv1(image)
#         out = self.pool1(out)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.block3(out)
#         out = self.block4(out)
#         out = self.global_pool(out)
#         out = out.view(out.size(0), -1)
#         # out = self.dropout(out)

#         return {"feature_map": out}


# class DualResNet(BaseModel):
#     def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.05, n_basefilters=32):
#         super().__init__()
#         self.resnet1 = ResNet(in_channels=in_channels, bn_momentum=bn_momentum, n_basefilters=n_basefilters)
#         self.resnet2 = ResNet(in_channels=in_channels, bn_momentum=bn_momentum, n_basefilters=n_basefilters)
#         self.fc1 = nn.Sequential(
#             nn.Linear(2 * 8 * n_basefilters, 4 * n_basefilters),
#             nn.PReLU(),
#         )
#         self.dropout = nn.Dropout(p=0.4)
#         self.fc2 = nn.Linear(4 * n_basefilters, n_outputs)

#     @property
#     def input_names(self) -> Sequence[str]:
#         return ("t1", "fdg")

#     @property
#     def output_names(self) -> Sequence[str]:
#         return ("logits",)

#     def forward(self, t1, fdg):
#         out1 = self.resnet1(t1)
#         out2 = self.resnet2(fdg)

#         out = torch.cat([out1["feature_map"], out2["feature_map"]], dim=1) # comment for notebook
#         #out = torch.cat([out1["feature_map"]], dim=1)
#         out = self.fc1(out)
#         out = self.dropout(out)
#         out = self.fc2(out)

#         return {"logits": out}


# class AdniMriPetModelFactory(BaseModelFactory):
#     def get_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
#         train_img_transform = get_image_transform(is_training=True)
#         eval_img_transform = get_image_transform(is_training=False)

#         target_labels = ["DX"]
#         if self._task == Task.BINARY_CLASSIFICATION:
#             target_transform_map = DIAGNOSIS_CODES_BINARY
#         elif self._task == Task.MULTI_CLASSIFICATION:
#             target_transform_map = DIAGNOSIS_CODES_MULTICLASS
#         else:
#             raise ValueError(f"{self._task} is not supported")
#         target_transform = {"DX": lambda x: target_transform_map[x]}

#         #output_names = ["t1", "fdg"]
#         output_names = ["t1"]

#         LOG.info("Loading training data from %s", self.args.train_data)
#         train_data = AdniDateset(
#             self.args.train_data,
#             target_labels=target_labels,
#             transform=train_img_transform,
#             target_transform=target_transform,
#         )
#         train_loader = self._make_named_data_loader(train_data, output_names, is_training=True)
#         train_loader.num_workers = 4

#         LOG.info("Loading validation data from %s", self.args.val_data)
#         val_data = AdniDateset(
#             self.args.val_data,
#             target_labels=target_labels,
#             transform=eval_img_transform,
#             target_transform=target_transform,
#         )
#         valid_loader = self._make_named_data_loader(val_data, output_names, is_training=False)

#         LOG.info("Loading test data from %s", self.args.test_data)
#         test_data = AdniDateset(
#             self.args.test_data,
#             target_labels=target_labels,
#             transform=eval_img_transform,
#             target_transform=target_transform,
#         )
#         test_loader = self._make_named_data_loader(test_data, output_names, is_training=False)

#         return train_loader, valid_loader, test_loader

#     def get_model(self) -> BaseModel:
#         model_args = {
#             "in_channels": 1,
#             "n_outputs": self.args.num_classes,
#             "n_basefilters": 8,
#         }
#         return DualResNet(**model_args)


# def create_parser():
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--silent", action="store_true", default=False)

#     g = parser.add_argument_group("Training")
#     g.add_argument("--epoch", type=int, default=200, help="number of epochs for training")
#     g.add_argument("--pretrain", type=Path, help="whether use pretrain model")
#     g.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for training")
#     g.add_argument("--decay_rate", type=float, default=1e-4, help="weight decay")
#     g.add_argument("--optimizer", choices=["Adam", "SGD", "AdamW"], default="Adam", help="type of optimizer")

#     g = parser.add_argument_group("Data")
#     g.add_argument("--batchsize", type=int, default=12, help="input batch size")

#     g.add_argument("--train_data", type=Path, required=True, help="path to training dataset")
#     g.add_argument("--val_data", type=Path, required=True, help="path to validation dataset")
#     g.add_argument("--test_data", type=Path, required=True, help="path to testing dataset")

#     g = parser.add_argument_group("Logging")
#     g.add_argument(
#         "--experiment_name",
#         nargs="?",
#         const=True,  # present, but not followed by a command-line argument
#         default=False,  # not present
#         help="Whether to give the experiment a particular name (default: current date and time).",
#     )

#     parser.set_defaults(task="clf", num_classes=1, shape="vol_with_bg")

#     return parser


# def main(args=None):
#     args = create_parser().parse_args(args=args)

#     logging.basicConfig(level=logging.INFO)

#     torch.manual_seed(20220223)
#     factory = AdniMriPetModelFactory(args)

#     experiment_dir, checkpoints_dir, tb_log_dir = factory.make_directories()

#     factory.write_args(experiment_dir / "experiment_args.json")

#     train_loader, valid_loader, test_loader = factory.get_data()
#     discriminator = factory.get_and_init_model()
#     optimizerD = factory.get_optimizer(filter(lambda p: p.requires_grad, discriminator.parameters()))
#     loss = factory.get_loss()

#     tb_log_dir = experiment_dir / "tensorboard"
#     checkpoints_dir = experiment_dir / "checkpoints"
#     tb_log_dir.mkdir(parents=True, exist_ok=True)
#     checkpoints_dir.mkdir(parents=True, exist_ok=True)

#     train_metrics = factory.get_metrics()
#     train_hooks = [TensorBoardLogger(str(tb_log_dir / "train"), train_metrics)]

#     eval_metrics_tb = factory.get_metrics()
#     eval_hooks = [TensorBoardLogger(str(tb_log_dir / "eval"), eval_metrics_tb)]
#     eval_metrics_cp = factory.get_metrics()
#     eval_hooks.append(
#         CheckpointSaver(discriminator, checkpoints_dir, save_every_n_epochs=1, max_keep=5, metrics=eval_metrics_cp)
#     )

#     dev = torch.device("cuda")
#     train_and_evaluate(
#         model=discriminator,
#         loss=loss,
#         train_data=train_loader,
#         optimizer=optimizerD,
#         scheduler=StepLR(optimizerD, step_size=100, gamma=0.2),
#         num_epochs=args.epoch,
#         eval_data=valid_loader,
#         train_hooks=train_hooks,
#         eval_hooks=eval_hooks,
#         device=dev,
#         progressbar=not args.silent,
#     )

#     return factory


# if __name__ == "__main__":
#      main()
    # main(["--train_data", "../adni-data-preprocessing/classification-nomci/mri-pet/0-train.h5", "--val_data", "../adni-data-preprocessing/classification-nomci/mri-pet/0-valid.h5", "--test_data", "../adni-data-preprocessing/classification-nomci/mri-pet/0-test.h5"])