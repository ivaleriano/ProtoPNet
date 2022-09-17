from operator import is_
from ast import literal_eval
import os
from pathlib import Path
import shutil
from tokenize import String
from typing import Sequence,Tuple

import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
import model_multimodal
import push_multimodal
import prune
import train_and_test_multimodal as tnt
import save_multimodal as save
from log import create_logger
from preprocess_multimodal import mean, std, preprocess_input_function
from wrappers_multimodal import NamedDataLoader, mesh_collate 


from train_adni_multimodal import get_image_transform
from train_adni_multimodal import DIAGNOSIS_CODES_BINARY
from train_adni_multimodal import AdniDateset
import json

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_epochs", type=int, default=250, help="number of epochs for training")
    parser.add_argument("--num_warm_epochs", type=int, default=0, help="number of warm epochs")
    parser.add_argument("--push_start", type=int, help="to start push epoch")
    parser.add_argument("--push_epochs", help=" every x for push epochs")
    parser.add_argument("--coefs")
    parser.add_argument("--last_layer_optimizer_lr", type=float, default=1e-4)
    parser.add_argument("--warm_optimizer_lrs")
    parser.add_argument("--joint_lr_step_size", type=int, default=5)
    parser.add_argument("--joint_optimizer_lrs")
    parser.add_argument("--train_dir")
    parser.add_argument("--test_dir")
    parser.add_argument("--train_batch_size", type=int, default=100)
    parser.add_argument("--test_batch_size", type=int, default=50)
    parser.add_argument("--train_push_batch_size", type=int, default=100)
    parser.add_argument("--base_architecture")
    parser.add_argument("--img_size_mri", type=int, default=138)
    parser.add_argument("--img_size_pet", type=int, default=130)
    parser.add_argument("--prototype_shape", default=(30, 128, 1, 1))
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--prototype_activation_function", default='log')
    parser.add_argument("--add_on_layers_type", default='regular')
    parser.add_argument("--experiment_run", default='001')
    parser.add_argument('-gpuid', nargs=1, type=str, default='0,1') # python3 main.py -gpuid=0,1,2,3

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
        # if self._task == adni_hdf.Task.SURVIVAL_ANALYSIS:
        #     collate_fn = partial(cox_collate_fn, data_collate=collate_fn)

        kwargs = {"batch_size": batch_size, "collate_fn": collate_fn, "shuffle": is_training, "drop_last": is_training}

        output_names = list(model_data_names) + data_loader_target_names()
        loader = NamedDataLoader(dataset, output_names=output_names, **kwargs)
        return loader

def data_loader_target_names():
    target_names = ["target"]
    return target_names

def main(args=None):
    args = create_parser().parse_args(args=args)
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
    #args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    base_architecture = args.base_architecture
    base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

    experiment_run = args.experiment_run
    model_dir = './saved_multimodal_models/' + base_architecture + '/' + experiment_run + '/'
    makedir(model_dir)
    tmp_dir = 'shape_continuum/ProtoPNet'
    #tmp_dir = ''
    shutil.copy(src=os.path.join('', __file__), dst=model_dir)
    shutil.copy(src=os.path.join(tmp_dir, 'settings_multimodal.py'), dst=model_dir)
    shutil.copy(src=os.path.join(tmp_dir, base_architecture_type + '_features_multimodal.py'), dst=model_dir)
    shutil.copy(src=os.path.join(tmp_dir, 'model_multimodal.py'), dst=model_dir)
    shutil.copy(src=os.path.join(tmp_dir, 'train_and_test_multimodal.py'), dst=model_dir)
    shutil.copy(src=os.path.join(tmp_dir, 'run_main_multimodal.sh'), dst=model_dir)
    # os.getcwd()
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    # load the data
    #from settings import train_dir, test_dir, train_push_dir, \
    #                     train_batch_size, test_batch_size, train_push_batch_size

    normalize = transforms.Normalize(mean=mean,
                                    std=std)

    # all datasets
    train_img_transform = get_image_transform(is_training=True)
    eval_img_transform = get_image_transform(is_training=False)

    target_labels = ["DX"]
    target_transform_map = DIAGNOSIS_CODES_BINARY
    target_transform = {"DX": lambda x: target_transform_map[x]}

    train_dir = args.train_dir.split(',')
    train_batch_size = int(args.train_batch_size)
    output_names = ["t1", "fdg"]
    print("Loading training data from", train_dir)
    train_dataset = AdniDateset(
        train_dir,
        target_labels=target_labels,
        transform=train_img_transform,
        target_transform=target_transform,
        is_training=True
    )
    train_loader = _make_named_data_loader(
        batchsize=train_batch_size, dataset=train_dataset, 
        model_data_names=output_names, is_training=True)
    train_loader.num_workers = 12
    # torch.utils.data.DataLoader(
    #     train_dataset, batch_size=train_batch_size, shuffle=True,
    #     num_workers=12, pin_memory=False)   # original 4 icxel commented for multimodal

    train_push_batch_size = int(args.train_push_batch_size)
    print("Loading training data 2 from", train_dir)
    train_push_dataset = AdniDateset(
        train_dir,
        target_labels=target_labels,
        transform=eval_img_transform,
        target_transform=target_transform,
        is_training=True
    )
    train_push_loader = _make_named_data_loader(
        batchsize=train_push_batch_size, dataset=train_push_dataset, 
        model_data_names=output_names, is_training=True) 
    # torch.utils.data.DataLoader(
    #     train_push_dataset, batch_size=train_push_batch_size, shuffle=True,
    #     num_workers=12, pin_memory=False) # shuffle = False Icxel commented for multimodal

    test_dir = [args.test_dir]
    test_batch_size = int(args.test_batch_size)
    print("Loading test data from", test_dir)
    test_dataset = AdniDateset(
                test_dir,
                target_labels=target_labels,
                transform=eval_img_transform,
                target_transform=target_transform,
                is_training=False
            )
    test_loader = _make_named_data_loader(
        batchsize=test_batch_size, dataset=test_dataset, 
        model_data_names=output_names, is_training=False)

    # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(train_batch_size))

    # construct the model
    img_size = dict()
    img_size["t1"] = int(args.img_size_mri)
    img_size["fdg"] = int(args.img_size_pet)
    prototype_shape = tuple(int(x) for x in args.prototype_shape[1:-1].split(','))
    num_classes = int(args.num_classes)
    prototype_activation_function = args.prototype_activation_function
    add_on_layers_type = args.add_on_layers_type
    ppnet = model_multimodal.construct_DualPPNet(base_architecture=base_architecture,
                                pretrained=False, img_size=img_size,
                                prototype_shape=prototype_shape,
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type=add_on_layers_type)
    #if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    # define optimizer
    #from settings import joint_optimizer_lrs, joint_lr_step_size
    strs = args.joint_optimizer_lrs[1:-1].split(',') # take away section in brackets
    joint_optimizer_lrs = {strs[0].split(':')[0][1:-1]: float(strs[0].split(':')[1]), strs[1].split(':')[0][1:-1]: float(strs[1].split(':')[1]),
    strs[2].split(':')[0][1:-1]: float(strs[2].split(':')[1])} #take away the [1:-1]
    joint_lr_step_size = int(args.joint_lr_step_size)
    joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-2}, # bias are now also being regularized
    {'params': ppnet.fc1.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-2}, # 1e-3 here and above Icxel
    #{'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.5) # 0.1 original icxel

    #from settings import warm_optimizer_lrs
    strs = args.warm_optimizer_lrs[1:-1].split(',')
    warm_optimizer_lrs = {strs[0].split(':')[0][1:-1]: float(strs[0].split(':')[1]), strs[1].split(':')[0][1:-1]: float(strs[1].split(':')[1])} # take [1:-1] away
    warm_optimizer_specs = \
    [{'params': ppnet.fc1.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    #from settings import last_layer_optimizer_lr
    last_layer_optimizer_lr = args.last_layer_optimizer_lr
    last_layer_optimizer_specs = [{'params': ppnet.fc2.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # train the model
    log('start training')
    import copy
    strs = args.coefs[1:-1].split(',')
    coefs = {strs[0].split(':')[0][1:-1]: float(strs[0].split(':')[1]), strs[1].split(':')[0][1:-1]: float(strs[1].split(':')[1]),
    strs[2].split(':')[0][1:-1]: float(strs[2].split(':')[1]), strs[3].split(':')[0][1:-1] : float(strs[3].split(':')[1])}
    num_train_epochs = int(args.num_train_epochs)
    num_warm_epochs = int(args.num_warm_epochs)
    push_start = int(args.push_start)
    push_epochs = [int(x) for x in args.push_epochs[1:-1].split(',')]
    accu_arr = list()
    accu_t1_arr = list()
    accu_fdg_arr =list()
    cross_ent_arr = list()
    cross_ent_t1_arr = list()
    cross_ent_fdg_arr = list()
    cluster_arr = list()
    cluster_t1_arr = list()
    cluster_fdg_arr = list()
    sep_arr = list()
    sep_t1_arr = list()
    sep_fdg_arr =list()
    loss_arr = list()
    loss_t1_arr = list()
    loss_fdg_arr = list()
    accu_val_arr = list()
    accu_val_t1_arr = list()
    accu_val_fdg_arr = list()
    cross_ent_val_arr = list()
    cross_ent_val_t1_arr = list()
    cross_ent_val_fdg_arr = list()
    cluster_val_arr = list()
    cluster_val_t1_arr = list()
    cluster_val_fdg_arr = list()
    sep_val_arr = list()
    sep_val_t1_arr = list()
    sep_val_fdg_arr =list()
    accu_first_push_arr = list()
    accu_first_push_t1_arr = list()
    accu_first_push_fdg_arr = list()
    cross_ent_first_push_arr = list()
    cross_ent_first_push_t1_arr = list()
    cross_ent_first_push_fdg_arr = list()
    cluster_first_push_arr = list()
    cluster_first_push_t1_arr = list()
    cluster_first_push_fdg_arr = list()
    sep_first_push_arr = list()
    sep_first_push_t1_arr = list()
    sep_first_push_fdg_arr =list()
    loss_first_push_arr = list()
    loss_first_push_t1_arr = list()
    loss_first_push_fdg_arr = list()
    accu_push_arr = list()
    accu_push_t1_arr = list()
    accu_push_fdg_arr = list()
    cross_ent_push_arr = list()
    cross_ent_push_t1_arr = list()
    cross_ent_push_fdg_arr = list()
    cluster_push_arr = list()
    cluster_push_t1_arr = list()
    cluster_push_fdg_arr = list()
    sep_push_arr = list()
    sep_push_t1_arr = list()
    sep_push_fdg_arr =list()
    loss_push_arr = list()
    loss_push_t1_arr = list()
    loss_push_fdg_arr = list()
    accu_push_val_arr = list()
    accu_push_val_t1_arr = list()
    accu_push_val_fdg_arr = list()
    cross_ent_val_push_arr = list()
    cross_ent_val_push_t1_arr = list()
    cross_ent_val_push_fdg_arr = list()
    cluster_val_push_arr = list()
    cluster_val_push_t1_arr = list()
    cluster_val_push_fdg_arr = list()
    sep_val_push_arr = list()
    sep_val_push_t1_arr = list()
    sep_val_push_fdg_arr =list()
    loss_val_push_arr = list()
    loss_val_push_t1_arr = list()
    loss_val_push_fdg_arr = list()
    for epoch in range(num_train_epochs):
        log('epoch: \t{0}'.format(epoch))

        if epoch < num_warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            joint_lr_scheduler.step()
            info_train = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)
            accu_arr.append(info_train['sum'])
            accu_t1_arr.append(info_train['t1'])
            accu_fdg_arr.append(info_train['fdg'])
            cross_ent_arr.append(info_train['cross_ent'])
            cross_ent_t1_arr.append(info_train['cross_ent_t1'])
            cross_ent_fdg_arr.append(info_train['cross_ent_fdg'])
            cluster_arr.append(info_train['cluster'])
            cluster_t1_arr.append(info_train['cluster_t1'])
            cluster_fdg_arr.append(['cluster_fdg'])
            sep_arr.append(info_train['separation'])
            sep_t1_arr.append(info_train['separation_t1'])
            sep_fdg_arr.append(info_train['separation_fdg'])
            loss_arr.append(info_train['loss'].item())
            loss_t1_arr.append(info_train['loss_t1'].item())
            loss_fdg_arr.append(info_train['loss_fdg'].item())

        info_val = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=info_val['sum'],
                                    target_accu=0.70, log=log) #changed by Icxel, target_accu = 0.70
        accu_val_arr.append(info_val['sum'])
        accu_val_t1_arr.append(info_val['t1'])
        accu_val_fdg_arr.append(info_val['fdg'])
        cross_ent_val_arr.append(info_val['cross_ent'])
        cross_ent_val_t1_arr.append(info_val['cross_ent_t1'])
        cross_ent_val_fdg_arr.append(info_val['cross_ent_fdg'])
        cluster_val_arr.append(info_val['cluster'])
        cluster_val_t1_arr.append(info_val['cluster_t1'])
        cluster_val_fdg_arr.append(info_val['cluster_fdg'])
        sep_val_arr.append(info_val['separation'])
        sep_val_t1_arr.append(info_val['separation_t1'])
        sep_val_fdg_arr.append(info_val['separation_fdg'])
        if epoch >= push_start and epoch in push_epochs:
            push_multimodal.push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function, #preprocess_input_function, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log)
            info_first_push = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=info_first_push['sum'],
                                        target_accu=0.70, log=log)
            accu_first_push_arr.append(info_first_push['sum'])
            accu_first_push_t1_arr.append(info_first_push['t1'])
            accu_first_push_fdg_arr.append(info_first_push['fdg'])
            cross_ent_first_push_arr.append(info_first_push['cross_ent'])
            cross_ent_first_push_t1_arr.append(info_first_push['cross_ent_t1'])
            cross_ent_first_push_fdg_arr.append(info_first_push['cross_ent_fdg'])
            cluster_first_push_arr.append(info_first_push['cluster'])
            cluster_first_push_t1_arr.append(info_first_push['cluster_t1'])
            cluster_first_push_fdg_arr.append(info_first_push['cluster_fdg'])
            sep_first_push_arr.append(info_first_push['separation'])
            sep_first_push_t1_arr.append(info_first_push['separation_t1'])
            sep_first_push_fdg_arr.append(info_first_push['separation_fdg'])
            loss_first_push_arr.append(info_first_push['loss'])
            loss_first_push_t1_arr.append(info_first_push['loss_t1'])
            loss_first_push_fdg_arr.append(info_first_push['loss_fdg'])
            if prototype_activation_function != 'linear':
                tnt.last_only(model=ppnet_multi, log=log)
                for i in range(20):
                    log('iteration: \t{0}'.format(i))
                    info_push = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                class_specific=class_specific, coefs=coefs, log=log)
                    accu_push_arr.append(info_push['sum'])
                    accu_push_t1_arr.append(info_push['t1'])
                    accu_push_fdg_arr.append(info_push['fdg'])
                    cross_ent_push_arr.append(info_push['cross_ent'])
                    cross_ent_push_t1_arr.append(info_push['cross_ent_t1'])
                    cross_ent_push_fdg_arr.append(info_push['cross_ent_fdg'])
                    cluster_push_arr.append(info_push['cluster'])
                    cluster_push_t1_arr.append(info_push['cluster_t1'])
                    cluster_push_fdg_arr.append(info_push['cluster_fdg'])
                    sep_push_arr.append(info_push['separation'])
                    sep_push_t1_arr.append(info_push['separation_t1'])
                    sep_push_fdg_arr.append(info_push['separation_fdg'])
                    loss_push_arr.append(info_push['loss'].item())
                    loss_push_t1_arr.append(info_push['loss_t1'].item())
                    loss_push_fdg_arr.append(info_push['loss_fdg'].item())
                    info_push_val = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log)
                    accu_push_val_arr.append(info_push_val['sum'])
                    accu_push_val_t1_arr.append(info_push_val['t1'])
                    accu_push_val_fdg_arr.append(info_push_val['fdg'])
                    cross_ent_val_push_arr.append(info_push_val['cross_ent'])
                    cross_ent_val_push_t1_arr.append(info_push_val['cross_ent_t1'])
                    cross_ent_val_push_fdg_arr.append(info_push_val['cross_ent_fdg'])
                    cluster_val_push_arr.append(info_push_val['cluster'])
                    cluster_val_push_t1_arr.append(info_push_val['cluster_t1'])
                    cluster_val_push_fdg_arr.append(info_push_val['cluster_fdg'])
                    sep_val_push_arr.append(info_push_val['separation'])
                    sep_val_push_t1_arr.append(info_push_val['separation_t1'])
                    sep_val_push_fdg_arr.append(info_push_val['separation_fdg'])
                    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=info_push_val['sum'],
                                                target_accu=0.70, log=log)
    log('\taccuracy array: \t{0}'.format(accu_arr))
    log('\tcross ent array: \t{0}'.format(cross_ent_arr))
    log('\tcluster array: \t{0}'.format(cluster_arr))
    log('\tcluster array t1: \t{0}'.format(cluster_t1_arr))
    log('\tcluster array fdg: \t{0}'.format(cluster_fdg_arr))
    log('\tsep array: \t{0}'.format(sep_arr))
    log('\tsep array t1: \t{0}'.format(sep_t1_arr))
    log('\tsep array fdg: \t{0}'.format(sep_fdg_arr))
    log('\tloss array: \t{0}'.format(loss_arr))
    log('\taccuracy val array: \t{0}'.format(accu_val_arr))
    log('\tcross ent val array: \t{0}'.format(cross_ent_val_arr))
    log('\tcluster val array: \t{0}'.format(cluster_val_arr))
    log('\tcluster val array t1: \t{0}'.format(cluster_val_t1_arr))
    log('\tcluster val array fdg: \t{0}'.format(cluster_val_fdg_arr))
    log('\tsep val array: \t{0}'.format(sep_val_arr))
    log('\tsep val t1 array: \t{0}'.format(sep_val_t1_arr))
    log('\tsep val fdg array: \t{0}'.format(sep_val_fdg_arr))
    log('\taccuracy first push array: \t{0}'.format(accu_first_push_arr))
    log('\tcross ent first push array: \t{0}'.format(cross_ent_first_push_arr))
    log('\tcluster first push array: \t{0}'.format(cluster_first_push_arr))
    log('\tcluster first push t1 array: \t{0}'.format(cluster_first_push_t1_arr))
    log('\tcluster first push fdg array: \t{0}'.format(cluster_first_push_fdg_arr))
    log('\tsep first push array: \t{0}'.format(sep_first_push_arr))
    log('\tsep first push array t1: \t{0}'.format(sep_first_push_t1_arr))
    log('\tsep first push array fdg: \t{0}'.format(sep_first_push_fdg_arr))
    log('\tloss first push array: \t{0}'.format(loss_first_push_arr))
    log('\taccuracy push array: \t{0}'.format(accu_push_arr))
    log('\tcross ent push array: \t{0}'.format(cross_ent_push_arr))
    log('\tcluster push array: \t{0}'.format(cluster_push_arr))
    log('\tcluster push array t1: \t{0}'.format(cluster_push_t1_arr))
    log('\tcluster push array fdg: \t{0}'.format(cluster_push_fdg_arr))
    log('\tsep push array: \t{0}'.format(sep_push_arr))
    log('\tsep push array t1: \t{0}'.format(sep_push_t1_arr))
    log('\tsep push array fdg: \t{0}'.format(sep_push_fdg_arr))
    log('\tloss push array: \t{0}'.format(loss_push_arr))
    log('\taccuracy push val array: \t{0}'.format(accu_push_val_arr))
    log('\tcross ent push val array: \t{0}'.format(cross_ent_val_push_arr))
    log('\tcluster push val array: \t{0}'.format(cluster_val_push_arr))
    log('\tcluster push val t1 array: \t{0}'.format(cluster_val_push_t1_arr))
    log('\tcluster push val fdg array: \t{0}'.format(cluster_val_push_fdg_arr))
    log('\tsep push val array: \t{0}'.format(sep_val_push_arr))
    log('\tsep push val array t1: \t{0}'.format(sep_val_push_t1_arr))
    log('\tsep push val array fdg: \t{0}'.format(sep_val_push_fdg_arr))
    logclose()

if __name__ == "__main__":
    args=["--num_train_epochs",'201',"--num_warm_epochs",'0',"--push_start",'10',"--push_epochs",'[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240]',
    "--train_dir","/mnt/nas/Users/Sebastian/adni-mri-pet/registered/classification-nomci/mri-pet/0-train.h5", "--test_dir","/mnt/nas/Users/Sebastian/adni-mri-pet/registered/classification-nomci/mri-pet/0-valid.h5",
    "--coefs",'{"crs_ent":1.3,"clst":0.6,"sep":-0.18,"l1":1e-4}',"--last_layer_optimizer_lr",'5e-4', "--warm_optimizer_lrs",'{"add_on_layers":3e-3,"prototype_vectors":3e-3}',
    "--joint_lr_step_size",'50', "--joint_optimizer_lrs",'{"features":1e-3,"add_on_layers":1e-3,"prototype_vectors":1e-3}', "--train_push_batch_size",'150',
    "--test_batch_size",'75',"--train_batch_size",'150',"--experiment_run",'012',"--add_on_layers_type",'regular',"--prototype_activation_function",'log',"--num_classes",'2',
    "--prototype_shape",'(30,128,1,1)',"--img_size_mri",'138',"--img_size_pet","130","--base_architecture",'resnet18']
    main(args)

    

    