##### MODEL AND DATA LOADING
#from ctypes.wintypes import HICON
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image
from torch.autograd import Variable
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import re

import os
import copy

from helpers import makedir, find_high_activation_crop
import model
import push
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function
from train_adni_multimodal import AdniDateset
from train_adni_multimodal import get_image_transform
from train_adni_multimodal import DIAGNOSIS_CODES_BINARY

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpuid", nargs=1, type=str, default='0')
parser.add_argument("--modeldir", nargs=1, type=str)
parser.add_argument("--model", nargs=1, type=str)
parser.add_argument("--imgdir", nargs=1, type=str)
parser.add_argument("--imgmri", nargs=1, type=str)
parser.add_argument("--imgpet", nargs=1, type=str)
parser.add_argument("--imgclass", nargs=1, type=int, default=-1)
parser.add_argument("--testdir", nargs=1, type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

# specify the test image to be analyzed
test_image_dir = args.imgdir[0] #'./local_analysis/Painted_Bunting_Class15_0081/'
test_image_name_mri = args.imgmri[0] #'Painted_Bunting_0081_15230.jpg'
test_image_name_pet = args.imgpet[0]
test_image_label = args.imgclass[0] #15

test_image_path_mri = os.path.join(test_image_dir, test_image_name_mri)
test_image_path_pet = os.path.join(test_image_dir, test_image_name_pet)

# load the model
check_test_accu = False

load_model_dir = args.modeldir[0] #'./saved_models/vgg19/003/'
load_model_name = args.model[0] #'10_18push0.7822.pth'
model_base_architecture = load_model_dir.split('/')[2]
experiment_run = '/'.join(load_model_dir.split('/')[3:])

save_analysis_path = os.path.join(test_image_dir, model_base_architecture,
                                  experiment_run, load_model_name)
makedir(save_analysis_path)

log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str)

log('load model from ' + load_model_path)
log('model base architecture: ' + model_base_architecture)
log('experiment run: ' + experiment_run)

ppnet = torch.load(load_model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

img_size = ppnet_multi.module.img_size
prototype_shape = ppnet.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = True

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# load the test data and check test accuracy
#from settings import test_dir
test_dir=args.test_dir
if check_test_accu:
    test_batch_size = 100

    print(img_size)
    eval_img_transform = get_image_transform(is_training=False)

    target_labels = ["DX"]
    target_transform_map = DIAGNOSIS_CODES_BINARY
    target_transform = {"DX": lambda x: target_transform_map[x]}
    test_dataset = AdniDateset(
                test_dir,
                target_labels=target_labels,
                transform=eval_img_transform,
                target_transform=target_transform,
                is_training=False
            )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)
    log('test set size: {0}'.format(len(test_loader.dataset)))

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=print)

##### SANITY CHECK
# confirm prototype class identity
load_img_dir = os.path.join(load_model_dir, 'img')

prototype_info = np.load(os.path.join(load_img_dir, 'epoch-' + epoch_number_str, 'bb'+epoch_number_str+'.npy'))
prototype_img_identity = prototype_info[:, -1]

log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
log('Their class identities are: ' + str(prototype_img_identity))

# confirm prototype connects most strongly to its own class
prototype_max_connection = torch.argmax(ppnet.fc2.weight, dim=0)
prototype_max_connection = prototype_max_connection.cpu().numpy()
if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
    log('All prototypes connect most strongly to their respective classes.')
else:
    log('WARNING: Not all prototypes connect most strongly to their respective classes.')

##### HELPER FUNCTIONS FOR PLOTTING
def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
    
    plt.imsave(fname, undo_preprocessed_img)
    #plt.imsave(fname, np.stack((undo_preprocessed_img,), axis=-1)) #added by icxel)
    return undo_preprocessed_img

def save_prototype(fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))
    #plt.axis('off')
    plt.imsave(fname, p_img)
    
def save_prototype_self_activation(fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                    'prototype-img-original_with_self_act'+str(index)+'.png'))
    #plt.axis('off')
    plt.imsave(fname, p_img)

def save_prototype_original_img_with_bbox(fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    #plt.imshow(p_img_rgb)
    #plt.axis('off')
    plt.imsave(fname, p_img_rgb)

def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    #plt.imshow(img_rgb_float)
    #plt.axis('off')
    plt.imsave(fname, img_rgb_float)

# ICXEL - preprocessing already done when creating test images
preprocess_transforms = []
preprocess_transforms.append(tio.RescaleIntensity(out_min_max=(0, 1)))
preprocess = tio.Compose(preprocess_transforms)
img_pil_mri = Image.open(test_image_path_mri)
img_pil_pet = Image.open(test_image_path_pet)
img_2d_mri = preprocess(torch.from_numpy(np.transpose(np.asarray(img_pil_mri), (2,0,1))).unsqueeze(-1))
img_2d_pet = preprocess(torch.from_numpy(np.transpose(np.asarray(img_pil_pet), (2,0,1))).unsqueeze(-1))
image_new_name = "test_image_toTensor.png"
save_image(img_2d_mri[:,:,:,0], os.path.join(save_analysis_path, "mri_" + image_new_name))
save_image(img_2d_pet[:,:,:,0], os.path.join(save_analysis_path, "pet_" + image_new_name))
#img_tensor = preprocess(img_pil) commented by Icxel

#img_tensor = img_tensor[0,:,:].unsqueeze(0) # added Icxel
img_variable_mri = Variable(img_2d_mri[:,:,:,0].unsqueeze(0)) # Icxel, original img_tensor.unsqueeze(0)
img_variable_pet = Variable(img_2d_pet[:,:,:,0].unsqueeze(0)) # Icxel, original img_tensor.unsqueeze(0)

images_test_mri = img_variable_mri.cuda()
images_test_pet = img_variable_pet.cuda()
images_test_input = {"t1": images_test_mri, "pet": images_test_pet}
labels_test = torch.tensor([test_image_label])
#logits, min_distances = ppnet_multi(images_test_input)
logits_mri, min_distances_mri = ppnet_multi.ppnet1(images_test_mri)
logits_pet, min_distances_pet = ppnet_multi.ppnet2(images_test_pet)
conv_output_mri, distances_mri = ppnet.ppnet1.push_forward(images_test_mri)
conv_output_pet, distances_pet = ppnet.ppnet2.push_forward(images_test_pet)
prototype_activations_mri = ppnet.ppnet1.distance_2_similarity(min_distances_mri)
prototype_activations_pet = ppnet.ppnet2.distance_2_similarity(min_distances_pet)
prototype_activation_patterns_mri = ppnet.ppnet1.distance_2_similarity(distances_mri)
prototype_activation_patterns_pet = ppnet.ppnet2.distance_2_similarity(distances_pet)
if ppnet.prototype_activation_function == 'linear':
    prototype_activations_mri = prototype_activations_mri + max_dist
    prototype_activations_pet = prototype_activations_pet + max_dist
    prototype_activation_patterns_mri = prototype_activation_patterns_mri + max_dist
    prototype_activation_patterns_pet = prototype_activation_patterns_pet + max_dist

tables_mri = []
for i in range(logits_mri.size(0)):
    tables_mri.append((torch.argmax(logits_mri, dim=1)[i].item(), labels_test[i].item()))
    log(str(i) + ' ' + str(tables_mri[-1]))

tables_pet = []
for i in range(logits_pet.size(0)):
    tables_pet.append((torch.argmax(logits_pet, dim=1)[i].item(), labels_test[i].item()))
    log(str(i) + ' ' + str(tables_pet[-1]))

idx = 0
predicted_cls_mri = tables_mri[idx][0]
correct_cls_mri = tables_mri[idx][1]
log('Predicted MRI: ' + str(predicted_cls_mri))
log('Actual: ' + str(correct_cls_mri))
original_img_mri = save_preprocessed_img(os.path.join(save_analysis_path, 'original_img.png'),
                                     images_test_mri, idx)
predicted_cls_pet = tables_pet[idx][0]
correct_cls_pet = tables_pet[idx][1]
log('Predicted PET: ' + str(predicted_cls_pet))
log('Actual: ' + str(correct_cls_pet))
original_img_pet = save_preprocessed_img(os.path.join(save_analysis_path, 'original_img.png'),
                                     images_test_pet, idx)

##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))

log('Most activated 10 prototypes of this image MRI:')
array_act_mri, sorted_indices_act_mri = torch.sort(prototype_activations_mri[idx])
for i in range(1,9):
    log('top {0} activated prototype for this image MRI:'.format(i))
    save_prototype(os.path.join(save_analysis_path, 'most_activated_prototypes_mri',
                                'top-%d_activated_prototype.png' % i),
                   start_epoch_number, sorted_indices_act_mri[-i].item())
    save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes_mri',
                                                             'top-%d_activated_prototype_in_original_pimg.png' % i),
                                          epoch=start_epoch_number,
                                          index=sorted_indices_act_mri[-i].item(),
                                          bbox_height_start=prototype_info[sorted_indices_act_mri[-i].item()][1],
                                          bbox_height_end=prototype_info[sorted_indices_act_mri[-i].item()][2],
                                          bbox_width_start=prototype_info[sorted_indices_act_mri[-i].item()][3],
                                          bbox_width_end=prototype_info[sorted_indices_act_mri[-i].item()][4],
                                          color=(0, 255, 255))
    save_prototype_self_activation(os.path.join(save_analysis_path, 'most_activated_prototypes_mri',
                                                'top-%d_activated_prototype_self_act.png' % i),
                                   start_epoch_number, sorted_indices_act_mri[-i].item())
    log('prototype index mri: {0}'.format(sorted_indices_act_mri[-i].item()))
    log('prototype class identity mri: {0}'.format(prototype_img_identity[sorted_indices_act_mri[-i].item()]))
    if prototype_max_connection[sorted_indices_act_mri[-i].item()] != prototype_img_identity[sorted_indices_act_mri[-i].item()]:
        log('prototype connection identity mri: {0}'.format(prototype_max_connection[sorted_indices_act_mri[-i].item()]))
    log('activation value (similarity score) mri: {0}'.format(array_act_mri[-i]))
    log('last layer connection with predicted class mri: {0}'.format(ppnet.ppnet1.last_layer.weight[predicted_cls_mri][sorted_indices_act_mri[-i].item()]))
    
    activation_pattern_mri = prototype_activation_patterns_mri[idx][sorted_indices_act_mri[-i].item()].detach().cpu().numpy()
    upsampled_activation_pattern_mri = cv2.resize(activation_pattern_mri, dsize=(img_size["t1"], img_size["t1"]),
                                              interpolation=cv2.INTER_CUBIC)

log('Most activated 10 prototypes of this image PET:')
array_act_pet, sorted_indices_act_pet = torch.sort(prototype_activations_pet[idx])
for i in range(1,9):
    log('top {0} activated prototype for this image PET:'.format(i))
    save_prototype(os.path.join(save_analysis_path, 'most_activated_prototypes_pet',
                                'top-%d_activated_prototype.png' % i),
                   start_epoch_number, sorted_indices_act_pet[-i].item())
    save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes_pet',
                                                             'top-%d_activated_prototype_in_original_pimg.png' % i),
                                          epoch=start_epoch_number,
                                          index=sorted_indices_act_pet[-i].item(),
                                          bbox_height_start=prototype_info[sorted_indices_act_pet[-i].item()][1],
                                          bbox_height_end=prototype_info[sorted_indices_act_pet[-i].item()][2],
                                          bbox_width_start=prototype_info[sorted_indices_act_pet[-i].item()][3],
                                          bbox_width_end=prototype_info[sorted_indices_act_pet[-i].item()][4],
                                          color=(0, 255, 255))
    save_prototype_self_activation(os.path.join(save_analysis_path, 'most_activated_prototypes_pet',
                                                'top-%d_activated_prototype_self_act.png' % i),
                                   start_epoch_number, sorted_indices_act_pet[-i].item())
    log('prototype index pet: {0}'.format(sorted_indices_act_pet[-i].item()))
    log('prototype class identity pet: {0}'.format(prototype_img_identity[sorted_indices_act_pet[-i].item()]))
    if prototype_max_connection[sorted_indices_act_pet[-i].item()] != prototype_img_identity[sorted_indices_act_pet[-i].item()]:
        log('prototype connection identity pet: {0}'.format(prototype_max_connection[sorted_indices_act_pet[-i].item()]))
    log('activation value (similarity score) pet: {0}'.format(array_act_pet[-i]))
    log('last layer connection with predicted class pet: {0}'.format(ppnet.last_layer.weight[predicted_cls_pet][sorted_indices_act_pet[-i].item()]))
    
    activation_pattern_pet = prototype_activation_patterns_pet[idx][sorted_indices_act_pet[-i].item()].detach().cpu().numpy()
    upsampled_activation_pattern_pet = cv2.resize(activation_pattern_pet, dsize=(img_size["fdg"], img_size["fdg"]),
                                              interpolation=cv2.INTER_CUBIC)
    
    # show the most highly activated patch of the image by this prototype
    high_act_patch_indices_mri = find_high_activation_crop(upsampled_activation_pattern_mri)
    high_act_patch_mri = original_img_mri[high_act_patch_indices_mri[0]:high_act_patch_indices_mri[1],
                                  high_act_patch_indices_mri[2]:high_act_patch_indices_mri[3], :]
    log('most highly activated patch of the chosen image by this prototype mri:')
    #plt.axis('off')
    plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes_mri',
                            'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                            high_act_patch_mri)
               #np.stack((high_act_patch,), axis=-1)) #added by icxel)
    log('most highly activated patch by this prototype shown in the original image:')
    imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes_mri',
                            'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                     img_rgb=original_img_mri,
                     bbox_height_start=high_act_patch_indices_mri[0],
                     bbox_height_end=high_act_patch_indices_mri[1],
                     bbox_width_start=high_act_patch_indices_mri[2],
                     bbox_width_end=high_act_patch_indices_mri[3], color=(0, 255, 255))

    high_act_patch_indices_pet = find_high_activation_crop(upsampled_activation_pattern_pet)
    high_act_patch_pet = original_img_pet[high_act_patch_indices_pet[0]:high_act_patch_indices_pet[1],
                                  high_act_patch_indices_pet[2]:high_act_patch_indices_pet[3], :]
    log('most highly activated patch of the chosen image by this prototype pet:')
    #plt.axis('off')
    plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes_pet',
                            'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                            high_act_patch_pet)
               #np.stack((high_act_patch,), axis=-1)) #added by icxel)
    log('most highly activated patch by this prototype shown in the original image pet:')
    imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes_pet',
                            'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                     img_rgb=original_img_pet,
                     bbox_height_start=high_act_patch_indices_pet[0],
                     bbox_height_end=high_act_patch_indices_pet[1],
                     bbox_width_start=high_act_patch_indices_pet[2],
                     bbox_width_end=high_act_patch_indices_pet[3], color=(0, 255, 255))
    
    # show the image overlayed with prototype activation map
    rescaled_activation_pattern_mri = upsampled_activation_pattern_mri - np.amin(upsampled_activation_pattern_mri)
    rescaled_activation_pattern_mri = rescaled_activation_pattern_mri / np.amax(rescaled_activation_pattern_mri)
    heatmap_mri = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern_mri), cv2.COLORMAP_JET)
    heatmap_mri = np.float32(heatmap_mri) / 255
    heatmap_mri = heatmap_mri[...,::-1]
    overlayed_img_mri = 0.5 * original_img_mri + 0.3 * heatmap_mri
    log('prototype activation map of the chosen image mri:')
    #plt.axis('off')
    plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes_mri',
                            'prototype_activation_map_by_top-%d_prototype.png' % i),
               overlayed_img_mri)

    rescaled_activation_pattern_pet = upsampled_activation_pattern_pet - np.amin(upsampled_activation_pattern_pet)
    rescaled_activation_pattern_pet = rescaled_activation_pattern_pet / np.amax(rescaled_activation_pattern_pet)
    heatmap_pet = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern_pet), cv2.COLORMAP_JET)
    heatmap_pet = np.float32(heatmap_pet) / 255
    heatmap_pet = heatmap_pet[...,::-1]
    overlayed_img_pet = 0.5 * original_img_pet + 0.3 * heatmap_pet
    log('prototype activation map of the chosen image pet:')
    #plt.axis('off')
    plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes_pet',
                            'prototype_activation_map_by_top-%d_prototype.png' % i),
               overlayed_img_pet)
    log('--------------------------------------------------------------')

##### PROTOTYPES FROM TOP-k CLASSES
k = 2
log('Prototypes from top-%d classes mri:' % k)
topk_logits_mri, topk_classes_mri = torch.topk(logits_mri[idx], k=k)
for i,c in enumerate(topk_classes_mri.detach().cpu().numpy()):
    makedir(os.path.join(save_analysis_path, 'top-%d_class_prototypes_mri' % (i+1)))

    log('top %d predicted class mri: %d' % (i+1, c))
    log('logit of the class mri: %f' % topk_logits_mri[i])
    class_prototype_indices_mri = np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
    class_prototype_activations_mri = prototype_activations_mri[idx][class_prototype_indices_mri]
    _, sorted_indices_cls_act_mri = torch.sort(class_prototype_activations_mri)

    prototype_cnt = 1
    for j in reversed(sorted_indices_cls_act_mri.detach().cpu().numpy()):
        prototype_index_mri = class_prototype_indices_mri[j]
        save_prototype(os.path.join(save_analysis_path, 'top-%d_class_prototypes_mri' % (i+1),
                                    'top-%d_activated_prototype.png' % prototype_cnt),
                       start_epoch_number, prototype_index_mri)
        save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes_mri' % (i+1),
                                                                 'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
                                              epoch=start_epoch_number,
                                              index=prototype_index_mri,
                                              bbox_height_start=prototype_info[prototype_index_mri][1],
                                              bbox_height_end=prototype_info[prototype_index_mri][2],
                                              bbox_width_start=prototype_info[prototype_index_mri][3],
                                              bbox_width_end=prototype_info[prototype_index_mri][4],
                                              color=(0, 255, 255))
        save_prototype_self_activation(os.path.join(save_analysis_path, 'top-%d_class_prototypes_mri' % (i+1),
                                                    'top-%d_activated_prototype_self_act.png' % prototype_cnt),
                                       start_epoch_number, prototype_index_mri)
        log('prototype index mri: {0}'.format(prototype_index_mri))
        log('prototype class identity mri: {0}'.format(prototype_img_identity[prototype_index_mri]))
        if prototype_max_connection[prototype_index_mri] != prototype_img_identity[prototype_index_mri]:
            log('prototype connection identity mri: {0}'.format(prototype_max_connection[prototype_index_mri]))
        log('activation value (similarity score) mri: {0}'.format(prototype_activations_mri[idx][prototype_index_mri]))
        log('last layer connection mri: {0}'.format(ppnet.last_layer.weight[c][prototype_index_mri]))
        
        activation_pattern_mri = prototype_activation_patterns_mri[idx][prototype_index_mri].detach().cpu().numpy()
        upsampled_activation_pattern_mri = cv2.resize(activation_pattern_mri, dsize=(img_size["t1"], img_size["t1"]),
                                                  interpolation=cv2.INTER_CUBIC)
        
        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices_mri = find_high_activation_crop(upsampled_activation_pattern_mri)
        high_act_patch_mri = original_img_mri[high_act_patch_indices_mri[0]:high_act_patch_indices_mri[1],
                                      high_act_patch_indices_mri[2]:high_act_patch_indices_mri[3], :]
        log('most highly activated patch of the chosen image by this prototype mri:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes_mri' % (i+1),
                                'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
                                high_act_patch_mri)
                   #np.stack((high_act_patch,), axis=-1))#added by icxel,), axis=-1) #added by icxel)
        log('most highly activated patch by this prototype shown in the original image mri:')
        imsave_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes_mri' % (i+1),
                                            'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
                         img_rgb=original_img_mri,
                         bbox_height_start=high_act_patch_indices_mri[0],
                         bbox_height_end=high_act_patch_indices_mri[1],
                         bbox_width_start=high_act_patch_indices_mri[2],
                         bbox_width_end=high_act_patch_indices_mri[3], color=(0, 255, 255))
        
        # show the image overlayed with prototype activation map
        rescaled_activation_pattern_mri = upsampled_activation_pattern_mri - np.amin(upsampled_activation_pattern_mri)
        rescaled_activation_pattern_mri = rescaled_activation_pattern_mri / np.amax(rescaled_activation_pattern_mri)
        heatmap_mri = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern_mri), cv2.COLORMAP_JET)
        heatmap_mri = np.float32(heatmap_mri) / 255
        heatmap_mri = heatmap_mri[...,::-1]
        overlayed_img_mri = 0.5 * original_img_mri + 0.3 * heatmap_mri
        log('prototype activation map of the chosen image mri:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes_mri' % (i+1),
                                'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt),
                   overlayed_img_mri)
        log('--------------------------------------------------------------')
        prototype_cnt += 1
    log('***************************************************************')

if predicted_cls_mri == correct_cls_mri:
    log('Prediction MRI is correct.')
else:
    log('Prediction MRI is wrong.')


k = 2
log('Prototypes from top-%d classes pet:' % k)
topk_logits_pet, topk_classes_pet = torch.topk(logits_pet[idx], k=k)
for i,c in enumerate(topk_classes_pet.detach().cpu().numpy()):
    makedir(os.path.join(save_analysis_path, 'top-%d_class_prototypes_pet' % (i+1)))

    log('top %d predicted class pet: %d' % (i+1, c))
    log('logit of the class pet: %f' % topk_logits_pet[i])
    class_prototype_indices_pet = np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
    class_prototype_activations_pet = prototype_activations_pet[idx][class_prototype_indices_pet]
    _, sorted_indices_cls_act = torch.sort(class_prototype_activations_pet)

    prototype_cnt = 1
    for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
        prototype_index_pet = class_prototype_indices_pet[j]
        save_prototype(os.path.join(save_analysis_path, 'top-%d_class_prototypes_pet' % (i+1),
                                    'top-%d_activated_prototype.png' % prototype_cnt),
                       start_epoch_number, prototype_index_pet)
        save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes_pet' % (i+1),
                                                                 'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
                                              epoch=start_epoch_number,
                                              index=prototype_index_pet,
                                              bbox_height_start=prototype_info[prototype_index_pet][1],
                                              bbox_height_end=prototype_info[prototype_index_pet][2],
                                              bbox_width_start=prototype_info[prototype_index_pet][3],
                                              bbox_width_end=prototype_info[prototype_index_pet][4],
                                              color=(0, 255, 255))
        save_prototype_self_activation(os.path.join(save_analysis_path, 'top-%d_class_prototypes_pet' % (i+1),
                                                    'top-%d_activated_prototype_self_act.png' % prototype_cnt),
                                       start_epoch_number, prototype_index_pet)
        log('prototype index pet: {0}'.format(prototype_index_pet))
        log('prototype class identity pet: {0}'.format(prototype_img_identity[prototype_index_pet]))
        if prototype_max_connection[prototype_index_pet] != prototype_img_identity[prototype_index_pet]:
            log('prototype connection identity pet: {0}'.format(prototype_max_connection[prototype_index_pet]))
        log('activation value (similarity score) pet: {0}'.format(prototype_activations_pet[idx][prototype_index_pet]))
        log('last layer connection pet: {0}'.format(ppnet.last_layer.weight[c][prototype_index_pet]))
        
        activation_pattern_pet = prototype_activation_patterns_pet[idx][prototype_index_pet].detach().cpu().numpy()
        upsampled_activation_pattern_pet = cv2.resize(activation_pattern_pet, dsize=(img_size["fdg"], img_size["fdg"]),
                                                  interpolation=cv2.INTER_CUBIC)
        
        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices_pet = find_high_activation_crop(upsampled_activation_pattern_pet)
        high_act_patch_pet = original_img_pet[high_act_patch_indices_pet[0]:high_act_patch_indices_pet[1],
                                      high_act_patch_indices_pet[2]:high_act_patch_indices_pet[3], :]
        log('most highly activated patch of the chosen image by this prototype pet:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes_pet' % (i+1),
                                'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
                                high_act_patch_pet)
                   #np.stack((high_act_patch,), axis=-1))#added by icxel,), axis=-1) #added by icxel)
        log('most highly activated patch by this prototype shown in the original image:')
        imsave_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                            'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
                         img_rgb=original_img_pet,
                         bbox_height_start=high_act_patch_indices_pet[0],
                         bbox_height_end=high_act_patch_indices_pet[1],
                         bbox_width_start=high_act_patch_indices_pet[2],
                         bbox_width_end=high_act_patch_indices_pet[3], color=(0, 255, 255))
        
        # show the image overlayed with prototype activation map
        rescaled_activation_pattern_pet = upsampled_activation_pattern_pet - np.amin(upsampled_activation_pattern_pet)
        rescaled_activation_pattern_pet = rescaled_activation_pattern_pet / np.amax(rescaled_activation_pattern_pet)
        heatmap_pet = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern_pet), cv2.COLORMAP_JET)
        heatmap_pet = np.float32(heatmap_pet) / 255
        heatmap_pet = heatmap_pet[...,::-1]
        overlayed_img_pet = 0.5 * original_img_pet + 0.3 * heatmap_pet
        log('prototype activation map of the chosen image pet:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes_pet' % (i+1),
                                'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt),
                   overlayed_img_pet)
        log('--------------------------------------------------------------')
        prototype_cnt += 1
    log('***************************************************************')

if predicted_cls_pet == correct_cls_pet:
    log('Prediction PET is correct.')
else:
    log('Prediction PET is wrong.')

logclose()


# tmux -> shortcuts (next thing to do is for tmux, run even if detached)