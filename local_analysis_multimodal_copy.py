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

class LocalAnalysis():
    def __init__(self, args):
        self.args = self.create_parser().parse_args(args=args)
        self.load()
        self.analyze()

    def create_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--gpuid", nargs=1, type=str, default='0')
        parser.add_argument("--modeldir", nargs=1, type=str)
        parser.add_argument("--model", nargs=1, type=str)
        parser.add_argument("--imgdir", nargs=1, type=str)
        parser.add_argument("--imgmri", nargs=1, type=str)
        parser.add_argument("--imgpet", nargs=1, type=str)
        parser.add_argument("--imgclass", nargs=1, type=int, default=-1)
        parser.add_argument("--testdir", nargs=1, type=str)
        return parser
        #args = parser.parse_args()

    def load(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpuid[0]
        #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

        # specify the test image to be analyzed
        test_image_dir = self.args.imgdir[0] #'./local_analysis/Painted_Bunting_Class15_0081/'
        test_image_name_mri = self.args.imgmri[0] #'Painted_Bunting_0081_15230.jpg'
        test_image_name_pet = self.args.imgpet[0]
        self.test_image_label = self.args.imgclass[0] #15

        self.test_image_path_mri = os.path.join(test_image_dir, test_image_name_mri)
        self.test_image_path_pet = os.path.join(test_image_dir, test_image_name_pet)

        # load the model
        check_test_accu = False

        load_model_dir = self.args.modeldir[0] #'./saved_models/vgg19/003/'
        load_model_name = self.args.model[0] #'10_18push0.7822.pth'
        model_base_architecture = load_model_dir.split('/')[4]
        experiment_run = '/'.join(load_model_dir.split('/')[5:])

        self.save_analysis_path = os.path.join(test_image_dir, model_base_architecture,
                                        experiment_run, load_model_name)
        makedir(self.save_analysis_path)

        self.log, self.logclose = create_logger(log_filename=os.path.join(self.save_analysis_path, 'local_analysis.log'))

        load_model_path = os.path.join(load_model_dir, load_model_name)
        epoch_number_str = re.search(r'\d+', load_model_name).group(0)
        self.start_epoch_number = int(epoch_number_str)

        self.log('load model from ' + load_model_path)
        self.log('model base architecture: ' + model_base_architecture)
        self.log('experiment run: ' + experiment_run)

        self.ppnet = torch.load(load_model_path)
        self.ppnet = self.ppnet.cuda()
        self.ppnet_multi = torch.nn.DataParallel(self.ppnet)

        self.img_size = self.ppnet_multi.module.img_size
        prototype_shape = self.ppnet.prototype_shape
        self.max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

        class_specific = True

        normalize = transforms.Normalize(mean=mean,
                                        std=std)

        # load the test data and check test accuracy
        #from settings import test_dir
        test_dir=self.args.testdir
        if check_test_accu:
            test_batch_size = 100

            print(self.img_size)
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
            self.log('test set size: {0}'.format(len(test_loader.dataset)))

            accu = tnt.test(model=self.ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=print)

        ##### SANITY CHECK
        # confirm prototype class identity
        self.load_img_dir = os.path.join(load_model_dir, 'img')

        self.prototype_info = np.load(os.path.join(self.load_img_dir, 'epoch-' + epoch_number_str, 'bb'+epoch_number_str+'.npy'))
        self.prototype_img_identity = self.prototype_info[:, -1]

        self.log('Prototypes are chosen from ' + str(len(set(self.prototype_img_identity))) + ' number of classes.')
        self.log('Their class identities are: ' + str(self.prototype_img_identity))

        # confirm prototype connects most strongly to its own class
        self.prototype_max_connection = torch.argmax(self.ppnet.fc2.weight, dim=0)
        self.prototype_max_connection = self.prototype_max_connection.cpu().numpy()
        if np.sum(self.prototype_max_connection == self.prototype_img_identity) == self.ppnet.num_prototypes:
            self.log('All prototypes connect most strongly to their respective classes.')
        else:
            self.log('WARNING: Not all prototypes connect most strongly to their respective classes.')

    ##### HELPER FUNCTIONS FOR PLOTTING
    def save_preprocessed_img(self, fname, preprocessed_imgs, index=0):
        img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
        undo_preprocessed_img = undo_preprocess_input_function(img_copy)
        print('image index {0} in batch'.format(index))
        undo_preprocessed_img = undo_preprocessed_img[0]
        undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
        undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
    
        plt.imsave(fname, undo_preprocessed_img)
        #plt.imsave(fname, np.stack((undo_preprocessed_img,), axis=-1)) #added by icxel)
        return undo_preprocessed_img

    def save_prototype(self, fname, epoch, index, img_type):
        count = 0
        print(index)
        for file in sorted(os.listdir(os.path.join(self.load_img_dir, 'epoch-'+str(epoch)))):
            filename = os.fsdecode(file)
            if img_type in filename and 'original' not in filename:
                if count == index:
                    index_str = filename.split('-')[1].split('_')[0][3:]
                count += 1
        p_img = plt.imread(os.path.join(self.load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+index_str+ '_' + img_type + '.png'))
        #plt.axis('off')
        plt.imsave(fname, p_img)
    
    def save_prototype_self_activation(self, fname, epoch, index, img_type):
        count = 0
        for file in sorted(os.listdir(os.path.join(self.load_img_dir, 'epoch-'+str(epoch)))):
            filename = os.fsdecode(file)
            if img_type in filename and 'original_with_self_act' in filename:
                if count == index:
                    index_str = filename.split('-')[2].split('_')[5].split('.')[0]
                count += 1
        p_img = plt.imread(os.path.join(self.load_img_dir, 'epoch-'+str(epoch),
                                    'prototype-img-original_with_self_act_' + img_type + '_' + index_str +'.png'))
        #plt.axis('off')
        plt.imsave(fname, p_img)

    def save_prototype_original_img_with_bbox(self, fname, epoch, index,
                                            bbox_height_start, bbox_height_end,
                                            bbox_width_start, bbox_width_end, img_type, color=(0, 255, 255)):
        count = 0
        for file in sorted(os.listdir(os.path.join(self.load_img_dir, 'epoch-'+str(epoch)))):
            filename = os.fsdecode(file)
            if img_type in filename and 'original_with_self_act' not in filename and 'original' in filename:
                if count == index:
                    index_str = filename.split('-')[2].split('_')[2].split('.')[0]
                count += 1
        p_img_bgr = cv2.imread(os.path.join(self.load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original_'+ img_type + '_' + index_str +'.png'))
        cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
        p_img_rgb = p_img_bgr[...,::-1]
        p_img_rgb = np.float32(p_img_rgb) / 255
        #plt.imshow(p_img_rgb)
        #plt.axis('off')
        plt.imsave(fname, p_img_rgb)

    def imsave_with_bbox(self, fname, img_rgb, bbox_height_start, bbox_height_end,
                        bbox_width_start, bbox_width_end, color=(0, 255, 255)):
        img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
        cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                    color, thickness=2)
        img_rgb_uint8 = img_bgr_uint8[...,::-1]
        img_rgb_float = np.float32(img_rgb_uint8) / 255
        #plt.imshow(img_rgb_float)
        #plt.axis('off')
        plt.imsave(fname, img_rgb_float)

    def analyze(self):
        # ICXEL - preprocessing already done when creating test images
        preprocess_transforms = []
        preprocess_transforms.append(tio.RescaleIntensity(out_min_max=(0, 1)))
        preprocess = tio.Compose(preprocess_transforms)
        img_pil_mri = Image.open(self.test_image_path_mri)
        img_pil_pet = Image.open(self.test_image_path_pet)
        img_2d_mri = preprocess(torch.from_numpy(np.transpose(np.asarray(img_pil_mri), (2,0,1))).unsqueeze(-1))
        img_2d_pet = preprocess(torch.from_numpy(np.transpose(np.asarray(img_pil_pet), (2,0,1))).unsqueeze(-1))
        image_new_name = "test_image_toTensor.png"
        save_image(img_2d_mri[:,:,:,0], os.path.join(self.save_analysis_path, "mri_" + image_new_name))
        save_image(img_2d_pet[:,:,:,0], os.path.join(self.save_analysis_path, "pet_" + image_new_name))
        #img_tensor = preprocess(img_pil) commented by Icxel

        #img_tensor = img_tensor[0,:,:].unsqueeze(0) # added Icxel
        img_variable_mri = Variable(img_2d_mri[:,:,:,0].unsqueeze(0)) # Icxel, original img_tensor.unsqueeze(0)
        img_variable_pet = Variable(img_2d_pet[:,:,:,0].unsqueeze(0)) # Icxel, original img_tensor.unsqueeze(0)

        images_test_mri = img_variable_mri.cuda()
        images_test_pet = img_variable_pet.cuda()
        images_test_input = {"t1": images_test_mri, "pet": images_test_pet}
        labels_test = torch.tensor([self.test_image_label])
        #logits, min_distances = self.ppnet_multi(images_test_input)
        logits_mri, min_distances_mri = self.ppnet_multi.module.ppnet1(images_test_mri)
        logits_pet, min_distances_pet = self.ppnet_multi.module.ppnet2(images_test_pet)
        conv_output_mri, distances_mri = self.ppnet.ppnet1.push_forward(images_test_mri)
        conv_output_pet, distances_pet = self.ppnet.ppnet2.push_forward(images_test_pet)
        prototype_activations_mri = self.ppnet.ppnet1.distance_2_similarity(min_distances_mri)
        prototype_activations_pet = self.ppnet.ppnet2.distance_2_similarity(min_distances_pet)
        prototype_activation_patterns_mri = self.ppnet.ppnet1.distance_2_similarity(distances_mri)
        prototype_activation_patterns_pet = self.ppnet.ppnet2.distance_2_similarity(distances_pet)
        if self.ppnet.ppnet1.prototype_activation_function == 'linear':
            prototype_activations_mri = prototype_activations_mri + self.max_dist
            prototype_activation_patterns_mri = prototype_activation_patterns_mri + self.max_dist
        if self.ppnet.ppnet2.prototype_activation_function == 'linear':
            prototype_activations_pet = prototype_activations_pet + self.max_dist
            prototype_activation_patterns_pet = prototype_activation_patterns_pet + self.max_dist

        tables_mri = []
        for i in range(logits_mri.size(0)):
            tables_mri.append((torch.argmax(logits_mri, dim=1)[i].item(), labels_test[i].item()))
            self.log(str(i) + ' ' + str(tables_mri[-1]))

        tables_pet = []
        for i in range(logits_pet.size(0)):
            tables_pet.append((torch.argmax(logits_pet, dim=1)[i].item(), labels_test[i].item()))
            self.log(str(i) + ' ' + str(tables_pet[-1]))

        idx_mri = 0
        predicted_cls_mri = tables_mri[idx_mri][0]
        correct_cls_mri = tables_mri[idx_mri][1]
        self.log('Predicted MRI: ' + str(predicted_cls_mri))
        self.log('Actual: ' + str(correct_cls_mri))
        original_img_mri = self.save_preprocessed_img(os.path.join(self.save_analysis_path, 'original_img.png'),
                                            images_test_mri, idx_mri)

        idx_pet = 0                                  
        predicted_cls_pet = tables_pet[idx_pet][0]
        correct_cls_pet = tables_pet[idx_pet][1]
        self.log('Predicted PET: ' + str(predicted_cls_pet))
        self.log('Actual: ' + str(correct_cls_pet))
        original_img_pet = self.save_preprocessed_img(os.path.join(self.save_analysis_path, 'original_img.png'),
                                     images_test_pet, idx_pet)

        ##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
        makedir(os.path.join(self.save_analysis_path, 'most_activated_prototypes_mri'))
        makedir(os.path.join(self.save_analysis_path, 'most_activated_prototypes_pet'))

        self.log('Most activated 10 prototypes of this image MRI:')
        array_act_mri, sorted_indices_act_mri = torch.sort(prototype_activations_mri[idx_mri])
        print('len mri ',len(sorted_indices_act_mri))
        for i in range(1,9):
            self.log('top {0} activated prototype for this image MRI:'.format(i))
            self.save_prototype(os.path.join(self.save_analysis_path, 'most_activated_prototypes_mri',
                                        'top-%d_activated_prototype.png' % i),
                        self.start_epoch_number, sorted_indices_act_mri[-i].item(), 't1')
            self.save_prototype_original_img_with_bbox(fname=os.path.join(self.save_analysis_path, 'most_activated_prototypes_mri',
                                                                    'top-%d_activated_prototype_in_original_pimg.png' % i),
                                                epoch=self.start_epoch_number,
                                                index=sorted_indices_act_mri[-i].item(),
                                                bbox_height_start=self.prototype_info[sorted_indices_act_mri[-i].item()][1],
                                                bbox_height_end=self.prototype_info[sorted_indices_act_mri[-i].item()][2],
                                                bbox_width_start=self.prototype_info[sorted_indices_act_mri[-i].item()][3],
                                                bbox_width_end=self.prototype_info[sorted_indices_act_mri[-i].item()][4],
                                                img_type='t1',
                                                color=(0, 255, 255))
            self.save_prototype_self_activation(os.path.join(self.save_analysis_path, 'most_activated_prototypes_mri',
                                                        'top-%d_activated_prototype_self_act.png' % i),
                                        self.start_epoch_number, sorted_indices_act_mri[-i].item(), 't1')
            self.log('prototype index mri: {0}'.format(sorted_indices_act_mri[-i].item()))
            self.log('prototype class identity mri: {0}'.format(self.prototype_img_identity[sorted_indices_act_mri[-i].item()]))
            if self.prototype_max_connection[sorted_indices_act_mri[-i].item()] != self.prototype_img_identity[sorted_indices_act_mri[-i].item()]:
                self.log('prototype connection identity mri: {0}'.format(self.prototype_max_connection[sorted_indices_act_mri[-i].item()]))
            self.log('activation value (similarity score) mri: {0}'.format(array_act_mri[-i]))
            self.log('last layer connection with predicted class mri: {0}'.format(self.ppnet.ppnet1.last_layer.weight[predicted_cls_mri][sorted_indices_act_mri[-i].item()]))
    
            activation_pattern_mri = prototype_activation_patterns_mri[idx_mri][sorted_indices_act_mri[-i].item()].detach().cpu().numpy()
            upsampled_activation_pattern_mri = cv2.resize(activation_pattern_mri, dsize=(self.img_size["t1"], self.img_size["t1"]),
                                                    interpolation=cv2.INTER_CUBIC)

        self.log('Most activated 10 prototypes of this image PET:')
        array_act_pet, sorted_indices_act_pet = torch.sort(prototype_activations_pet[idx_pet])
        print('len pet ',len(sorted_indices_act_pet))
        for i in range(1,9):
            self.log('top {0} activated prototype for this image PET:'.format(i))
            self.save_prototype(os.path.join(self.save_analysis_path, 'most_activated_prototypes_pet',
                                        'top-%d_activated_prototype.png' % i),
                        self.start_epoch_number, sorted_indices_act_pet[-i].item(),'fdg')
            self.save_prototype_original_img_with_bbox(fname=os.path.join(self.save_analysis_path, 'most_activated_prototypes_pet',
                                                                    'top-%d_activated_prototype_in_original_pimg.png' % i),
                                                epoch=self.start_epoch_number,
                                                index=sorted_indices_act_pet[-i].item(),
                                                bbox_height_start=self.prototype_info[sorted_indices_act_pet[-i].item()][1],
                                                bbox_height_end=self.prototype_info[sorted_indices_act_pet[-i].item()][2],
                                                bbox_width_start=self.prototype_info[sorted_indices_act_pet[-i].item()][3],
                                                bbox_width_end=self.prototype_info[sorted_indices_act_pet[-i].item()][4],
                                                img_type='fdg',
                                                color=(0, 255, 255))
            self.save_prototype_self_activation(os.path.join(self.save_analysis_path, 'most_activated_prototypes_pet',
                                                        'top-%d_activated_prototype_self_act.png' % i),
                                        self.start_epoch_number, sorted_indices_act_pet[-i].item(), 'fdg')
            self.log('prototype index pet: {0}'.format(sorted_indices_act_pet[-i].item()))
            self.log('prototype class identity pet: {0}'.format(self.prototype_img_identity[sorted_indices_act_pet[-i].item()]))
            if self.prototype_max_connection[sorted_indices_act_pet[-i].item()] != self.prototype_img_identity[sorted_indices_act_pet[-i].item()]:
                self.log('prototype connection identity pet: {0}'.format(self.prototype_max_connection[sorted_indices_act_pet[-i].item()]))
            self.log('activation value (similarity score) pet: {0}'.format(array_act_pet[-i]))
            self.log('last layer connection with predicted class pet: {0}'.format(self.ppnet.fc2.weight[predicted_cls_pet][sorted_indices_act_pet[-i].item()]))
    
            activation_pattern_pet = prototype_activation_patterns_pet[idx_pet][sorted_indices_act_pet[-i].item()].detach().cpu().numpy()
            upsampled_activation_pattern_pet = cv2.resize(activation_pattern_pet, dsize=(self.img_size["fdg"], self.img_size["fdg"]),
                                                    interpolation=cv2.INTER_CUBIC)
    
            # show the most highly activated patch of the image by this prototype
            high_act_patch_indices_mri = find_high_activation_crop(upsampled_activation_pattern_mri)
            high_act_patch_mri = original_img_mri[high_act_patch_indices_mri[0]:high_act_patch_indices_mri[1],
                                  high_act_patch_indices_mri[2]:high_act_patch_indices_mri[3], :]
            self.log('most highly activated patch of the chosen image by this prototype mri:')
            #plt.axis('off')
            plt.imsave(os.path.join(self.save_analysis_path, 'most_activated_prototypes_mri',
                                    'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                                    high_act_patch_mri)
                    #np.stack((high_act_patch,), axis=-1)) #added by icxel)
            self.log('most highly activated patch by this prototype shown in the original image:')
            self.imsave_with_bbox(fname=os.path.join(self.save_analysis_path, 'most_activated_prototypes_mri',
                                    'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                            img_rgb=original_img_mri,
                            bbox_height_start=high_act_patch_indices_mri[0],
                            bbox_height_end=high_act_patch_indices_mri[1],
                            bbox_width_start=high_act_patch_indices_mri[2],
                            bbox_width_end=high_act_patch_indices_mri[3], color=(0, 255, 255))

            high_act_patch_indices_pet = find_high_activation_crop(upsampled_activation_pattern_pet)
            high_act_patch_pet = original_img_pet[high_act_patch_indices_pet[0]:high_act_patch_indices_pet[1],
                                        high_act_patch_indices_pet[2]:high_act_patch_indices_pet[3], :]
            self.log('most highly activated patch of the chosen image by this prototype pet:')
            #plt.axis('off')
            plt.imsave(os.path.join(self.save_analysis_path, 'most_activated_prototypes_pet',
                                    'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                                    high_act_patch_pet)
                    #np.stack((high_act_patch,), axis=-1)) #added by icxel)
            self.log('most highly activated patch by this prototype shown in the original image pet:')
            self.imsave_with_bbox(fname=os.path.join(self.save_analysis_path, 'most_activated_prototypes_pet',
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
            self.log('prototype activation map of the chosen image mri:')
            #plt.axis('off')
            plt.imsave(os.path.join(self.save_analysis_path, 'most_activated_prototypes_mri',
                                    'prototype_activation_map_by_top-%d_prototype.png' % i),
                    overlayed_img_mri)

            rescaled_activation_pattern_pet = upsampled_activation_pattern_pet - np.amin(upsampled_activation_pattern_pet)
            rescaled_activation_pattern_pet = rescaled_activation_pattern_pet / np.amax(rescaled_activation_pattern_pet)
            heatmap_pet = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern_pet), cv2.COLORMAP_JET)
            heatmap_pet = np.float32(heatmap_pet) / 255
            heatmap_pet = heatmap_pet[...,::-1]
            overlayed_img_pet = 0.5 * original_img_pet + 0.3 * heatmap_pet
            self.log('prototype activation map of the chosen image pet:')
            #plt.axis('off')
            plt.imsave(os.path.join(self.save_analysis_path, 'most_activated_prototypes_pet',
                                    'prototype_activation_map_by_top-%d_prototype.png' % i),
                    overlayed_img_pet)
            self.log('--------------------------------------------------------------')

        ##### PROTOTYPES FROM TOP-k CLASSES
        k = 2
        self.log('Prototypes from top-%d classes mri:' % k)
        topk_logits_mri, topk_classes_mri = torch.topk(logits_mri[idx_mri], k=k)
        for i,c in enumerate(topk_classes_mri.detach().cpu().numpy()):
            makedir(os.path.join(self.save_analysis_path, 'top-%d_class_prototypes_mri' % (i+1)))

            self.log('top %d predicted class mri: %d' % (i+1, c))
            self.log('logit of the class mri: %f' % topk_logits_mri[i])
            class_prototype_indices_mri = np.nonzero(self.ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
            class_prototype_activations_mri = prototype_activations_mri[idx_mri][class_prototype_indices_mri]
            _, sorted_indices_cls_act_mri = torch.sort(class_prototype_activations_mri)

            prototype_cnt = 1
            for j in reversed(sorted_indices_cls_act_mri.detach().cpu().numpy()):
                prototype_index_mri = class_prototype_indices_mri[j]
                self.save_prototype(os.path.join(self.save_analysis_path, 'top-%d_class_prototypes_mri' % (i+1),
                                            'top-%d_activated_prototype.png' % prototype_cnt),
                            self.start_epoch_number, prototype_index_mri, 't1')
                self.save_prototype_original_img_with_bbox(fname=os.path.join(self.save_analysis_path, 'top-%d_class_prototypes_mri' % (i+1),
                                                                        'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
                                                    epoch=self.start_epoch_number,
                                                    index=prototype_index_mri,
                                                    bbox_height_start=self.prototype_info[prototype_index_mri][1],
                                                    bbox_height_end=self.prototype_info[prototype_index_mri][2],
                                                    bbox_width_start=self.prototype_info[prototype_index_mri][3],
                                                    bbox_width_end=self.prototype_info[prototype_index_mri][4],
                                                    img_type = 't1',
                                                    color=(0, 255, 255))
                self.save_prototype_self_activation(os.path.join(self.save_analysis_path, 'top-%d_class_prototypes_mri' % (i+1),
                                                            'top-%d_activated_prototype_self_act.png' % prototype_cnt),
                                            self.start_epoch_number, prototype_index_mri,'t1')
                self.log('prototype index mri: {0}'.format(prototype_index_mri))
                self.log('prototype class identity mri: {0}'.format(self.prototype_img_identity[prototype_index_mri]))
                if self.prototype_max_connection[prototype_index_mri] != self.prototype_img_identity[prototype_index_mri]:
                    self.log('prototype connection identity mri: {0}'.format(self.prototype_max_connection[prototype_index_mri]))
                self.log('activation value (similarity score) mri: {0}'.format(prototype_activations_mri[idx_mri][prototype_index_mri]))
                self.log('last layer connection mri: {0}'.format(self.ppnet.fc2.weight[c][prototype_index_mri]))
        
                activation_pattern_mri = prototype_activation_patterns_mri[idx_mri][prototype_index_mri].detach().cpu().numpy()
                upsampled_activation_pattern_mri = cv2.resize(activation_pattern_mri, dsize=(self.img_size["t1"], self.img_size["t1"]),
                                                        interpolation=cv2.INTER_CUBIC)
        
                # show the most highly activated patch of the image by this prototype
                high_act_patch_indices_mri = find_high_activation_crop(upsampled_activation_pattern_mri)
                high_act_patch_mri = original_img_mri[high_act_patch_indices_mri[0]:high_act_patch_indices_mri[1],
                                            high_act_patch_indices_mri[2]:high_act_patch_indices_mri[3], :]
                self.log('most highly activated patch of the chosen image by this prototype mri:')
                #plt.axis('off')
                plt.imsave(os.path.join(self.save_analysis_path, 'top-%d_class_prototypes_mri' % (i+1),
                                        'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
                                        high_act_patch_mri)
                        #np.stack((high_act_patch,), axis=-1))#added by icxel,), axis=-1) #added by icxel)
                self.log('most highly activated patch by this prototype shown in the original image mri:')
                self.imsave_with_bbox(fname=os.path.join(self.save_analysis_path, 'top-%d_class_prototypes_mri' % (i+1),
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
                self.log('prototype activation map of the chosen image mri:')
                #plt.axis('off')
                plt.imsave(os.path.join(self.save_analysis_path, 'top-%d_class_prototypes_mri' % (i+1),
                                        'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt),
                        overlayed_img_mri)
                self.log('--------------------------------------------------------------')
                prototype_cnt += 1
            self.log('***************************************************************')

        if predicted_cls_mri == correct_cls_mri:
            self.log('Prediction MRI is correct.')
        else:
            self.log('Prediction MRI is wrong.')


        k = 2
        self.log('Prototypes from top-%d classes pet:' % k)
        topk_logits_pet, topk_classes_pet = torch.topk(logits_pet[idx_pet], k=k)
        for i,c in enumerate(topk_classes_pet.detach().cpu().numpy()):
            makedir(os.path.join(self.save_analysis_path, 'top-%d_class_prototypes_pet' % (i+1)))

            self.log('top %d predicted class pet: %d' % (i+1, c))
            self.log('logit of the class pet: %f' % topk_logits_pet[i])
            class_prototype_indices_pet = np.nonzero(self.ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
            class_prototype_activations_pet = prototype_activations_pet[idx_pet][class_prototype_indices_pet]
            _, sorted_indices_cls_act = torch.sort(class_prototype_activations_pet)

            prototype_cnt = 1
            for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
                prototype_index_pet = class_prototype_indices_pet[j]
                self.save_prototype(os.path.join(self.save_analysis_path, 'top-%d_class_prototypes_pet' % (i+1),
                                            'top-%d_activated_prototype.png' % prototype_cnt),
                            self.start_epoch_number, prototype_index_pet, 'fdg')
                self.save_prototype_original_img_with_bbox(fname=os.path.join(self.save_analysis_path, 'top-%d_class_prototypes_pet' % (i+1),
                                                                        'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
                                                    epoch=self.start_epoch_number,
                                                    index=prototype_index_pet,
                                                    bbox_height_start=self.prototype_info[prototype_index_pet][1],
                                                    bbox_height_end=self.prototype_info[prototype_index_pet][2],
                                                    bbox_width_start=self.prototype_info[prototype_index_pet][3],
                                                    bbox_width_end=self.prototype_info[prototype_index_pet][4],
                                                    img_type = 'fdg',
                                                    color=(0, 255, 255))
                self.save_prototype_self_activation(os.path.join(self.save_analysis_path, 'top-%d_class_prototypes_pet' % (i+1),
                                                            'top-%d_activated_prototype_self_act.png' % prototype_cnt),
                                            self.start_epoch_number, prototype_index_pet, 'fdg')
                self.log('prototype index pet: {0}'.format(prototype_index_pet))
                self.log('prototype class identity pet: {0}'.format(self.prototype_img_identity[prototype_index_pet]))
                if self.prototype_max_connection[prototype_index_pet] != self.prototype_img_identity[prototype_index_pet]:
                    self.log('prototype connection identity pet: {0}'.format(self.prototype_max_connection[prototype_index_pet]))
                self.log('activation value (similarity score) pet: {0}'.format(prototype_activations_pet[idx_pet][prototype_index_pet]))
                self.log('last layer connection pet: {0}'.format(self.ppnet.fc2.weight[c][prototype_index_pet]))
        
                activation_pattern_pet = prototype_activation_patterns_pet[idx_pet][prototype_index_pet].detach().cpu().numpy()
                upsampled_activation_pattern_pet = cv2.resize(activation_pattern_pet, dsize=(self.img_size["fdg"], self.img_size["fdg"]),
                                                        interpolation=cv2.INTER_CUBIC)
        
                # show the most highly activated patch of the image by this prototype
                high_act_patch_indices_pet = find_high_activation_crop(upsampled_activation_pattern_pet)
                high_act_patch_pet = original_img_pet[high_act_patch_indices_pet[0]:high_act_patch_indices_pet[1],
                                            high_act_patch_indices_pet[2]:high_act_patch_indices_pet[3], :]
                self.log('most highly activated patch of the chosen image by this prototype pet:')
                #plt.axis('off')
                plt.imsave(os.path.join(self.save_analysis_path, 'top-%d_class_prototypes_pet' % (i+1),
                                        'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
                                        high_act_patch_pet)
                        #np.stack((high_act_patch,), axis=-1))#added by icxel,), axis=-1) #added by icxel)
                self.log('most highly activated patch by this prototype shown in the original image PET:')
                self.imsave_with_bbox(fname=os.path.join(self.save_analysis_path, 'top-%d_class_prototypes_pet' % (i+1),
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
                self.log('prototype activation map of the chosen image pet:')
                #plt.axis('off')
                plt.imsave(os.path.join(self.save_analysis_path, 'top-%d_class_prototypes_pet' % (i+1),
                                        'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt),
                        overlayed_img_pet)
                self.log('--------------------------------------------------------------')
                prototype_cnt += 1
            self.log('***************************************************************')

        if predicted_cls_pet == correct_cls_pet:
            self.log('Prediction PET is correct.')
        else:
            self.log('Prediction PET is wrong.')

        self.logclose()


# tmux -> shortcuts (next thing to do is for tmux, run even if detached)