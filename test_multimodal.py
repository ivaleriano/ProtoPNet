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

from local_analysis_multimodal_aopc2 import LocalAnalysis
from settings_multimodal import img_size_mri,img_size_pet

DIAGNOSIS_CODES_BINARY = {
    "CN": np.array(0, dtype=np.int64),
    "Dementia": np.array(1, dtype=np.int64),
}

def get_mri_image_transform():
    # /1/MRI/T1/data           Dataset {113, 137, 113}
    img_transforms = []
    img_transforms.append(tio.RescaleIntensity(out_min_max=(0, 1)))
    img_transform = tio.Compose(img_transforms)
    return img_transform


def get_pet_image_transform():
    img_transforms = []
    img_transforms.append(tio.RescaleIntensity(out_min_max=(0, 1)))
    img_transform = tio.Compose(img_transforms)
    return img_transform


def get_image_transform():
    return {
        "t1": get_mri_image_transform(),
        "fdg": get_pet_image_transform(),
     }


class AdniDatasetTest(Dataset):
    def __init__(self, filepaths, target_labels: Sequence[str], transform=None, target_transform=None) -> None:
        self.filepaths = filepaths
        self.target_labels = target_labels
        self.transform = transform
        self.target_transform = target_transform
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
        for filename in self.filepaths:
            with h5py.File(filename, mode='r') as fin:
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

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir")
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--base_architecture")
    parser.add_argument("--img_size_mri", type=int, default=138)
    parser.add_argument("--img_size_pet", type=int, default=161)
    parser.add_argument("--prototype_shape", default=(30, 128, 1, 1))
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--prototype_activation_function", default='log')
    parser.add_argument("--add_on_layers_type", default='regular')
    parser.add_argument("--experiment_run", default='001')
    parser.add_argument("--model_dir", nargs=1, type=str)
    parser.add_argument("--model", nargs=1, type=str)
    parser.add_argument("--validation_dir", nargs=1, type=str)
    parser.add_argument("--gpuid", nargs=1, type=str, default='0,1') # python3 main.py -gpuid=0,1,2,3

    return parser

def _make_named_data_loader(batchsize, dataset: Dataset, model_data_names: Sequence[str], is_training: bool = False
) -> NamedDataLoader:
    """Create a NamedDataLoader for the given dataset.

    Args:
        dataset (Dataset):
        The dataset to wrap.
        model_data_names (list of str):
        Should correspond to the names of the first `len(model_data_names)` outputs
        of the dataset and that are fed to model in a forward pass. The names
        of the targets used to compute the loss will be retrieved from :meth:`data_loader_target_names`.
        is_training (bool):
        Whether to enable training mode or not.
    """
    batch_size = batchsize
    if len(dataset) < batch_size:
        if is_training:
            raise RuntimeError(
                "batch size ({:d}) cannot exceed dataset size ({:d})".format(batch_size, len(dataset))
            )
        else:
            batch_size = len(dataset)

    collate_fn = mesh_collate
    kwargs = {"batch_size": batch_size, "collate_fn": collate_fn, "shuffle": is_training, "drop_last": is_training}

    output_names = list(model_data_names) + data_loader_target_names()
    loader = NamedDataLoader(dataset, output_names=output_names, **kwargs)
    return loader

def data_loader_target_names():
    target_names = ["target"]
    return target_names

def main(args):
    args = create_parser().parse_args(args=args)
    eval_img_transform = get_image_transform()
    target_labels = ["DX"]
    target_transform_map = DIAGNOSIS_CODES_BINARY
    target_transform = {"DX": lambda x: target_transform_map[x]}

    output_names = ["t1", "fdg"]
    test_dir = [args.test_dir]
    test_batch_size = int(args.test_batch_size)
    print("Loading test data from", test_dir)
    test_dataset = AdniDatasetTest(
                test_dir,
                target_labels=target_labels,
                transform=eval_img_transform,
                target_transform=target_transform
            )
    test_loader = _make_named_data_loader(
        batchsize=test_batch_size, dataset=test_dataset, 
        model_data_names=output_names, is_training=False)

    model_dir = args.model_dir[0]
    model = args.model[0]
    validation_dir = args.validation_dir[0]
    gpuid = args.gpuid
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'analysis_aopc.log'))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(test_batch_size))

    predictions = []
    predictions_t1 = []
    predictions_fdg = []
    for j, (mri_image, pet_image, label) in enumerate(test_loader):
        if label == 1:
            name = 'Dementia'
        else:
            name = 'CN'
        image_name = "test_image" + str(j) + "_" + name
        save_image(mri_image, '/home/icxel/shape_continuum/ProtoPNet/testing_images/' + image_name + "_mri.png")
        save_image(pet_image, '/home/icxel/shape_continuum/ProtoPNet/testing_images/' + image_name + "_pet.png")

        if label==1:
            img_class = "1"
        elif label==0:
            img_class = "0"
        else:
            print("wrong class")
        args_analysis = ["--gpuid",gpuid,"--modeldir",model_dir,"--model",model,"--imgdir","/home/icxel/shape_continuum/ProtoPNet/testing_images/",
        "--imgmri",image_name + "_mri.png","--imgpet",image_name + "_pet.png","--imgclass",img_class,"--testdir",validation_dir,"--useprevious","false",
        "--numimg",str(j)]
        analysis = LocalAnalysis(args_analysis)
        pred, pred_t1, pred_fdg = analysis.get_aopc_predictions()
        predictions.insert(j, pred)
        predictions_t1.insert(j, pred_t1)
        predictions_fdg.insert(j, pred_fdg)
        os.remove('/home/icxel/shape_continuum/ProtoPNet/testing_images/' + image_name + "_mri.png")
        os.remove('/home/icxel/shape_continuum/ProtoPNet/testing_images/' + image_name + "_pet.png")
    accuracies = []
    correct_in_round = np.zeros_like(predictions[0])
    accuracies_t1 = []
    correct_in_round_t1 = np.zeros_like(predictions_t1[0])
    accuracies_fdg = []
    correct_in_round_fdg = np.zeros_like(predictions_fdg[0])
    for k in range(len(predictions)):
        for l in range(len(predictions[k])):
            if(predictions[k][l] == 1):
                correct_in_round[l] = correct_in_round[l] + 1
            if(predictions_t1[k][l] == 1):
                correct_in_round_t1[l] = correct_in_round_t1[l] + 1
            if(predictions_fdg[k][l] == 1):
                correct_in_round_fdg[l] = correct_in_round_fdg[l] + 1
    log('total images: {0}'.format(len(predictions)))
    log('total subjects: {0}'.format(len(predictions)/22))
    for l in range(len(correct_in_round)):
        accuracies.insert(l, correct_in_round[l]/len(predictions))
        accuracies_t1.insert(l, correct_in_round_t1[l]/len(predictions_t1))
        accuracies_fdg.insert(l, correct_in_round_fdg[l]/len(predictions_fdg))
        log('OVERALL:')
        log('{0}  {1}  {2}'.format(str(l), accuracies[l], correct_in_round[l]))
        log('T1:')
        log('{0}  {1}  {2}'.format(str(l), accuracies_t1[l], correct_in_round_t1[l]))
        log('FDG:')
        log('{0}  {1}  {2}'.format(str(l), accuracies_fdg[l], correct_in_round_fdg[l]))

            #os.system("python3 /home/icxel/shape_continuum/ProtoPNet/local_analysis_multimodal.py " + "--gpuid," + gpuid + ",--modeldir," + model_dir + ",--model," + model + ",--imgdir," + "/home/icxel/shape_continuum/ProtoPNet/testing_images/"
            #+ ",--imgmri," + image_name + "_mri" + ",--imgpet," + image_name + "_pet" + ",--imgclass," + img_class + ",--test_dir," + validation_dir)


if __name__ == "__main__":
    args=["--test_dir",'/mnt/nas/Users/Sebastian/adni-mri-pet/registered/classification-nomci/mri-pet/0-test.h5',"--test_batch_size",'1',"--base_architecture",'resnet18',"--img_size_mri",'138',
    "--img_size_pet",'130',"--prototype_shape",'(30,128,1,1)',"--num_classes",'2',"--prototype_activation_function",'log',"--add_on_layers_type",'regular',"--experiment_run",'012',"--model_dir",'/home/icxel/saved_multimodal_models/resnet18/012/',
    "--model",'70_11push0.8209.pth',"--validation_dir","/mnt/nas/Users/Sebastian/adni-mri-pet/registered/classification-nomci/mri-pet/0-valid.h5"]
    main(args)