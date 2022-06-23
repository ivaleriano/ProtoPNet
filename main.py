from operator import is_
from ast import literal_eval
import os
from pathlib import Path
import shutil
from tokenize import String

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

from train_adni_mri import get_image_transform
from train_adni_mri import DIAGNOSIS_CODES_BINARY
from train_adni_mri import AdniDateset
import json
# book keeping namings and code
#from settings import base_architecture, img_size, prototype_shape, num_classes, \
#                     prototype_activation_function, add_on_layers_type, experiment_run
#from settings import train_dir, test_dir, train_push_dir, \
#                     train_batch_size, test_batch_size, train_push_batch_size
#from settings import joint_optimizer_lrs, joint_lr_step_size
#from settings import warm_optimizer_lrs
#from settings import last_layer_optimizer_lr
#from settings import coefs
#from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

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
    parser.add_argument("--img_size", type=int, default=139)
    parser.add_argument("--prototype_shape", default=(30, 128, 1, 1))
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--prototype_activation_function", default='log')
    parser.add_argument("--add_on_layers_type", default='regular')
    parser.add_argument("--experiment_run", default='001')
    parser.add_argument('-gpuid', nargs=1, type=str, default='0,1') # python3 main.py -gpuid=0,1,2,3

    return parser

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
    model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
    makedir(model_dir)
    tmp_dir = 'shape_continuum/ProtoPNet'
    #tmp_dir = ''
    shutil.copy(src=os.path.join('', __file__), dst=model_dir)
    shutil.copy(src=os.path.join(tmp_dir, 'settings.py'), dst=model_dir)
    shutil.copy(src=os.path.join(tmp_dir, base_architecture_type + '_features.py'), dst=model_dir)
    shutil.copy(src=os.path.join(tmp_dir, 'model.py'), dst=model_dir)
    shutil.copy(src=os.path.join(tmp_dir, 'train_and_test.py'), dst=model_dir)
    shutil.copy(src=os.path.join(tmp_dir, 'run_main.sh'), dst=model_dir)
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
    print("Loading training data from", train_dir)
    train_dataset = AdniDateset(
        train_dir,
        target_labels=target_labels,
        transform=train_img_transform,
        target_transform=target_transform,
        is_training=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=12, pin_memory=False)   # original 4 icxel

    train_push_batch_size = int(args.train_push_batch_size)
    print("Loading training data 2 from", train_dir)
    train_push_dataset = AdniDateset(
        train_dir,
        target_labels=target_labels,
        transform=eval_img_transform,
        target_transform=target_transform,
        is_training=True
    )
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=True,
        num_workers=12, pin_memory=False) # shuffle = False Icxel

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
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True,
        num_workers=12, pin_memory=False) # shuffle = False Icxel

    # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(train_batch_size))

    # construct the model
    img_size = int(args.img_size)
    prototype_shape = tuple(int(x) for x in args.prototype_shape[1:-1].split(',')) #take [1:-1] away
    num_classes = int(args.num_classes)
    prototype_activation_function = args.prototype_activation_function
    add_on_layers_type = args.add_on_layers_type
    ppnet = model.construct_PPNet(base_architecture=base_architecture,
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
    {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-2}, # 1e-3 here and above Icxel
    {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.5) # 0.1 original icxel

    #from settings import warm_optimizer_lrs
    strs = args.warm_optimizer_lrs[1:-1].split(',')
    warm_optimizer_lrs = {strs[0].split(':')[0][1:-1]: float(strs[0].split(':')[1]), strs[1].split(':')[0][1:-1]: float(strs[1].split(':')[1])} # take [1:-1] away
    warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

    #from settings import last_layer_optimizer_lr
    last_layer_optimizer_lr = args.last_layer_optimizer_lr
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # weighting of different training losses
    #from settings import coefs

    # number of training epochs, number of warm epochs, push start epoch, push epochs
    #from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

    # train the model
    log('start training')
    import copy
    strs = args.coefs[1:-1].split(',')
    coefs = {strs[0].split(':')[0][1:-1]: float(strs[0].split(':')[1]), strs[1].split(':')[0][1:-1]: float(strs[1].split(':')[1]),
    strs[2].split(':')[0][1:-1]: float(strs[2].split(':')[1]), strs[3].split(':')[0][1:-1] : float(strs[3].split(':')[1])} # take away [1:-1]
    num_train_epochs = int(args.num_train_epochs)
    num_warm_epochs = int(args.num_warm_epochs)
    push_start = int(args.push_start)
    push_epochs = [int(x) for x in args.push_epochs[1:-1].split(',')]
    for epoch in range(num_train_epochs):
        log('epoch: \t{0}'.format(epoch))

        if epoch < num_warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            joint_lr_scheduler.step()
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log)

        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                    target_accu=0.80, log=log) #changed by Icxel, target_accu = 0.70

        if epoch >= push_start and epoch in push_epochs:
            push.push_prototypes(
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
            accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                        target_accu=0.80, log=log)

            if prototype_activation_function != 'linear':
                tnt.last_only(model=ppnet_multi, log=log)
                for i in range(20):
                    log('iteration: \t{0}'.format(i))
                    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                class_specific=class_specific, coefs=coefs, log=log)
                    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log)
                    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                                target_accu=0.80, log=log)
    
    logclose()

if __name__ == "__main__":
    #main()
    args=["--num_train_epochs",'201',"--num_warm_epochs",'0',"--push_start",'10',"--push_epochs",'[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240]',
    "--train_dir","/mnt/nas/Users/Sebastian/adni-mri-pet/registered/classification-nomci/mri-pet/0-train.h5", "--test_dir","/mnt/nas/Users/Sebastian/adni-mri-pet/registered/classification-nomci/mri-pet/0-valid.h5",
    "--coefs",'{"crs_ent":1,"clst":0.8,"sep":-0.08,"l1":1e-4}',"--last_layer_optimizer_lr",'5e-4', "--warm_optimizer_lrs",'{"add_on_layers":3e-3,"prototype_vectors":3e-3}',
    "--joint_lr_step_size",'50', "--joint_optimizer_lrs",'{"features":1e-3,"add_on_layers":1e-3,"prototype_vectors":1e-3}', "--train_push_batch_size",'150',
    "--test_batch_size",'75',"--train_batch_size",'150',"--experiment_run",'020',"--add_on_layers_type",'regular',"--prototype_activation_function",'log',"--num_classes",'2',
    "--prototype_shape",'(30,128,1,1)',"--img_size",'138',"--base_architecture",'resnet18']
    main(args)

    
    # IDEAS
    # increase step size for epochs before decreasing lr oooor change gamma to 0.95 and then increase lr a bit
    # monitor specifically  cross entropy
    # plot the logits/probabilities predicted -> all 1 or 0  (negative)
    # pruning without reoptimizing last layer
    # check weights for last layer
    # train for 50-60 epochs
    # maybe run in parallel 
    # add PET and MRI (different splits so performance might be a bit different)

    

    # change datasets
    # compare to baseline
    # multimodal without protopnet -----> multimodal with protopnet
    