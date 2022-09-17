##### MODEL AND DATA LOADING
#from ctypes.wintypes import HICON
from xmlrpc.client import boolean
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image
from torch.autograd import Variable
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import MaxPool2d
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
from train_adni_mri import AdniDateset
from train_adni_mri import get_image_transform
from train_adni_mri import DIAGNOSIS_CODES_BINARY

import argparse

class LocalAnalysis():
    def __init__(self, args):
        self.args = self.create_parser().parse_args(args=args)
        self.predicted_cls = 0
        self.correct_cls = 0
        self.load()
        self.analyze()

    def create_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--gpuid", nargs=1, type=str, default='0')
        parser.add_argument("--modeldir", nargs=1, type=str)
        parser.add_argument("--model", nargs=1, type=str)
        parser.add_argument("--imgdir", nargs=1, type=str)
        parser.add_argument("--modality", nargs=1, type=str)
        parser.add_argument("--img", nargs=1, type=str)
        parser.add_argument("--imgclass", nargs=1, type=int, default=-1)
        parser.add_argument("--testdir", nargs=1, type=str)
        parser.add_argument("--useprevious", nargs=1, type=boolean)
        parser.add_argument("--numimg", nargs=1, type=int)
        return parser
        #args = parser.parse_args()

    def load(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpuid[0]
        self.modality = self.args.modality[0]
        # specify the test image to be analyzed
        test_image_dir = self.args.imgdir[0] #'./local_analysis/Painted_Bunting_Class15_0081/'
        test_image_name = self.args.img[0] #'Painted_Bunting_0081_15230.jpg'
        self.test_image_label = self.args.imgclass[0] #15

        self.test_image_path = os.path.join(test_image_dir, test_image_name)

        # for aopc
        self.use_previous = self.args.useprevious[0]
        self.num_img = self.args.numimg[0]

        # load the model
        check_test_accu = False

        load_model_dir = self.args.modeldir[0] #'./saved_models/vgg19/003/'
        load_model_name = self.args.model[0] #'10_18push0.7822.pth'
        model_base_architecture = load_model_dir.split('/')[4]
        experiment_run = '/'.join(load_model_dir.split('/')[5:])

        self.save_analysis_path = os.path.join(test_image_dir, model_base_architecture,
                                        experiment_run, load_model_name, str(self.num_img))
        makedir(self.save_analysis_path)
        self.save_new_prototypes_path = os.path.join(self.save_analysis_path, 'most_activated_prototypes',
                                                                    'new_prototypes') 
        if(not os.path.isdir(self.save_new_prototypes_path)):
            makedir(self.save_new_prototypes_path)

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

        self.img_size = self.ppnet.img_size
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
        # confirm prototypes class identity
        self.load_img_dir = os.path.join(load_model_dir, 'img')

        self.prototype_info = np.load(os.path.join(self.load_img_dir, 'epoch-' + epoch_number_str, 'bb'+epoch_number_str+'.npy'))
        self.prototype_img_identity = self.prototype_info[:, -1]

        self.log('Prototypes are chosen from ' + str(len(set(self.prototype_img_identity))) + ' number of classes.')
        self.log('Their class identities are: ' + str(self.prototype_img_identity))

        # confirm prototype connects most strongly to its own class
        self.prototype_max_connection = torch.argmax(self.ppnet.last_layer.weight, dim=0)
        self.prototype_max_connection = self.prototype_max_connection.cpu().numpy()
        if np.sum(self.prototype_max_connection == self.prototype_img_identity) == self.ppnet.num_prototypes:
            self.log('All prototypes connect most strongly to their respective classes.')
        else:
            self.log('WARNING: Not all prototypes connect most strongly to their respective classes.')

    ##### HELPER FUNCTIONS FOR PLOTTING
    def save_preprocessed_img(self, fname, preprocessed_imgs, index=0):
        img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
        img_copy = img_copy[0]
        img_copy = img_copy.detach().cpu().numpy()
        img_copy = np.transpose(img_copy, [1,2,0])
        self.log('image index {0} in batch'.format(index))
        plt.imsave(fname, img_copy)
    
        return img_copy

    def save_prototype(self, fname, epoch, index):
        p_img = plt.imread(os.path.join(self.load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))
        #plt.axis('off')
        plt.imsave(fname, p_img)
    
    def save_prototype_self_activation(self, fname, epoch, index):
        p_img = plt.imread(os.path.join(self.load_img_dir, 'epoch-'+str(epoch),
                                    'prototype-img-original_with_self_act'+str(index)+'.png'))
        #plt.axis('off')
        plt.imsave(fname, p_img)

    def save_prototype_original_img_with_bbox(self, fname, epoch, index,
                                            bbox_height_start, bbox_height_end,
                                            bbox_width_start, bbox_width_end, color=(0, 255, 255)):
        p_img_bgr = cv2.imread(os.path.join(self.load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
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

    def get_correct_prediction(self, predicted_cls, correct_cls):
        return predicted_cls == correct_cls

    def get_aopc_predictions(self):
        return self.predictions_aopc

    def display_prediction(self, logits, labels_test):
        tables = []
        for i in range(logits.size(0)):
            tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
            self.log(str(i) + ' ' + str(tables[-1]))

        idx = 0
        predicted_cls = tables[idx][0]
        correct_cls = tables[idx][1]
        self.log('Predicted: ' + str(predicted_cls))
        self.log('Actual: ' + str(correct_cls))
        return (predicted_cls, correct_cls)

    def fake_forward(self, aopc_overall_distances, mode):
        logits, min_distances = self.fake_forward_single_mode(aopc_overall_distances)
        img_info = (logits, min_distances)
        info = {mode: img_info}
        return (logits, min_distances, info)

    def fake_forward_single_mode(self, aopc_distances):
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        # global min pooling
        torch_distances = torch.tensor(aopc_distances.copy()).cuda()
        min_distances = -torch.max_pool2d(-torch_distances,
                                      kernel_size=(torch_distances.size()[2],
                                                   torch_distances.size()[3])) #ppnetmulti
        min_distances = min_distances.view(-1, self.ppnet.num_prototypes)
        prototype_activations = self.ppnet.distance_2_similarity(min_distances) #ppnetmulti
        prototype_activations[0,aopc_distances[0,:,0,0]==0] = 0
        logits = self.ppnet.last_layer(prototype_activations) #ppnetmulti
        return logits, min_distances

    def get_prototype_activations_info(self, min_distances, distances):
        prototype_activations = self.ppnet.distance_2_similarity(min_distances) #ppnetmulti
        prototype_activation_patterns = self.ppnet.distance_2_similarity(distances) #ppnetmulti
        if self.ppnet.prototype_activation_function == 'linear': #ppnetmulti
            prototype_activations = prototype_activations + self.max_dist
            prototype_activation_patterns = prototype_activation_patterns + self.max_dist
        return prototype_activations,prototype_activation_patterns

    def analyze(self):
        # ICXEL - preprocessing already done when creating test images
        preprocess_transforms = []
        preprocess_transforms.append(tio.RescaleIntensity(out_min_max=(0, 1)))
        preprocess = tio.Compose(preprocess_transforms)
        img_pil = Image.open(self.test_image_path)
        img_2d = preprocess(torch.from_numpy(np.transpose(np.asarray(img_pil), (2,0,1))).unsqueeze(-1))

        img_variable = Variable(img_2d[:,:,:,0].unsqueeze(0)) # Icxel, original img_tensor.unsqueeze(0)

        images_test = img_variable.cuda()
        labels_test = torch.tensor([self.test_image_label])
        logits, min_distances = self.ppnet_multi(images_test) #ppnetmulti
        _, distances = self.ppnet.push_forward(images_test) #ppnetmulti
        prototype_activations_info = self.get_prototype_activations_info(min_distances, distances)
        prototype_activations = prototype_activations_info[0]
        prototype_activation_patterns = prototype_activations_info[1]

        self.predicted_cls, self.correct_cls = self.display_prediction(logits, labels_test)
        idx = 0
        original_img = self.save_preprocessed_img(os.path.join(self.save_analysis_path, 'original_img'+ str(self.num_img) + '.png'),
                                            images_test, idx)

        aopc_weights = self.ppnet.last_layer.weight.clone().detach().cpu().numpy()
        aopc_overall_distances = list()
        aopc_distances = distances.clone().data.cpu().numpy()
        aopc_distances_to_append = aopc_distances.copy()
        aopc_overall_distances.append(aopc_distances_to_append)

        ##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
        makedir(os.path.join(self.save_analysis_path, 'most_activated_prototypes'))

        self.log('Most activated 10 prototypes of this image:')
        array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
        for i in range(1,len(sorted_indices_act)+1):
            self.log('top {0} activated prototype for this image:'.format(i))
            self.save_prototype(os.path.join(self.save_analysis_path, 'most_activated_prototypes',
                                        'top-%d_activated_prototype.png' % i),
                        self.start_epoch_number, sorted_indices_act[-i].item())
            self.save_prototype_original_img_with_bbox(fname=os.path.join(self.save_analysis_path, 'most_activated_prototypes',
                                                                    'top-%d_activated_prototype_in_original_pimg.png' % i),
                                                epoch=self.start_epoch_number,
                                                index=sorted_indices_act[-i].item(),
                                                bbox_height_start=self.prototype_info[sorted_indices_act[-i].item()][1],
                                                bbox_height_end=self.prototype_info[sorted_indices_act[-i].item()][2],
                                                bbox_width_start=self.prototype_info[sorted_indices_act[-i].item()][3],
                                                bbox_width_end=self.prototype_info[sorted_indices_act[-i].item()][4],
                                                color=(0, 255, 255))
            self.save_prototype_self_activation(os.path.join(self.save_analysis_path, 'most_activated_prototypes',
                                                        'top-%d_activated_prototype_self_act.png' % i),
                                        self.start_epoch_number, sorted_indices_act[-i].item())
            self.log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
            self.log('prototype class identity: {0}'.format(self.prototype_img_identity[sorted_indices_act[-i].item()]))
            if self.prototype_max_connection[sorted_indices_act[-i].item()] != self.prototype_img_identity[sorted_indices_act[-i].item()]:
                self.log('prototype connection identity: {0}'.format(self.prototype_max_connection[sorted_indices_act[-i].item()]))
            self.log('activation value (similarity score): {0}'.format(array_act[-i]))
            self.log('last layer connection with predicted class: {0}'.format(self.ppnet.last_layer.weight[self.predicted_cls][sorted_indices_act[-i].item()])) #ppnetmulti
            activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(self.img_size, self.img_size),
                                                    interpolation=cv2.INTER_CUBIC)
            print(self.img_size)

            # show the most highly activated patch of the image by this prototype
            high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
            high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                  high_act_patch_indices[2]:high_act_patch_indices[3], :]
            self.log('most highly activated patch of the chosen image by this prototype:')
            #plt.axis('off')
            plt.imsave(os.path.join(self.save_analysis_path, 'most_activated_prototypes',
                                    'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                                    high_act_patch)
                    #np.stack((high_act_patch,), axis=-1)) #added by icxel)
            self.log('most highly activated patch by this prototype shown in the original image:')
            self.imsave_with_bbox(fname=os.path.join(self.save_analysis_path, 'most_activated_prototypes',
                                    'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                            img_rgb=original_img,
                            bbox_height_start=high_act_patch_indices[0],
                            bbox_height_end=high_act_patch_indices[1],
                            bbox_width_start=high_act_patch_indices[2],
                            bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

            # show the image overlayed with prototype activation map
            rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
            rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]
            overlayed_img = 0.5 * original_img + 0.3 * heatmap
            self.log('prototype activation map of the chosen image:')
            #plt.axis('off')
            plt.imsave(os.path.join(self.save_analysis_path, 'most_activated_prototypes',
                                    'prototype_activation_map_by_top-%d_prototype.png' % i),
                    overlayed_img)
            aopc_weights[self.predicted_cls][sorted_indices_act[-i].item()] = 0
            aopc_weights[abs(self.predicted_cls-1)][sorted_indices_act[-i].item()] = 0
            aopc_distances[idx,sorted_indices_act[-i].item(),:,:] = 0
            aopc_overall_distances.append(aopc_distances.copy())
            self.log('--------------------------------------------------------------')
        #for i in range(9, len(sorted_indices_act)):
        #    aopc_weights[self.predicted_cls][sorted_indices_act[-i].item()] = 0
        #    aopc_weights[abs(self.predicted_cls-1)][sorted_indices_act[-i].item()] = 0
        #    aopc_distances[idx,sorted_indices_act[-i].item(),:,:] = 0
        #    aopc_overall_distances.append(aopc_distances.copy())

        self.predictions_aopc = []
        for i in range(len(aopc_overall_distances)):
            aopc_logits, aopc_min_distances, aopc_mode_info = self.fake_forward(aopc_overall_distances[i], self.modality)
            aopc_pred, aopc_correct_pred = self.display_prediction(aopc_logits, labels_test)
            if self.get_correct_prediction(aopc_pred, aopc_correct_pred):
                self.predictions_aopc.insert(i,1)
            else:
                self.predictions_aopc.insert(i,0)

        ##### PROTOTYPES FROM TOP-k CLASSES
        k = 2
        self.log('Prototypes from top-%d classes:' % k)
        topk_logits, topk_classes = torch.topk(logits[idx], k=k)
        for i,c in enumerate(topk_classes.detach().cpu().numpy()):
            makedir(os.path.join(self.save_analysis_path, 'top-%d_class_prototypes' % (i+1)))

            self.log('top %d predicted class: %d' % (i+1, c))
            self.log('logit of the class: %f' % topk_logits[i])
            class_prototype_indices = np.nonzero(self.ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0] #ppnetmulti
            class_prototype_activations = prototype_activations[idx][class_prototype_indices]
            _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

            prototype_cnt = 1
            for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
                prototype_index = class_prototype_indices[j]
                self.save_prototype(os.path.join(self.save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                            'top-%d_activated_prototype.png' % prototype_cnt),
                            self.start_epoch_number, prototype_index)
                self.save_prototype_original_img_with_bbox(fname=os.path.join(self.save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                                        'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
                                                    epoch=self.start_epoch_number,
                                                    index=prototype_index,
                                                    bbox_height_start=self.prototype_info[prototype_index][1],
                                                    bbox_height_end=self.prototype_info[prototype_index][2],
                                                    bbox_width_start=self.prototype_info[prototype_index][3],
                                                    bbox_width_end=self.prototype_info[prototype_index][4],
                                                    color=(0, 255, 255))
                self.save_prototype_self_activation(os.path.join(self.save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                            'top-%d_activated_prototype_self_act.png' % prototype_cnt),
                                            self.start_epoch_number, prototype_index)
                self.log('prototype index: {0}'.format(prototype_index))
                self.log('prototype class identity: {0}'.format(self.prototype_img_identity[prototype_index]))
                if self.prototype_max_connection[prototype_index] != self.prototype_img_identity[prototype_index]:
                    self.log('prototype connection identity: {0}'.format(self.prototype_max_connection[prototype_index]))
                self.log('activation value (similarity score): {0}'.format(prototype_activations[idx][prototype_index]))
                self.log('last layer connection: {0}'.format(self.ppnet.last_layer.weight[c][prototype_index]))
        
                activation_pattern = prototype_activation_patterns[idx][prototype_index].detach().cpu().numpy()
                upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(self.img_size, self.img_size),
                                                        interpolation=cv2.INTER_CUBIC)
        
                # show the most highly activated patch of the image by this prototype
                high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
                high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                            high_act_patch_indices[2]:high_act_patch_indices[3], :]
                self.log('most highly activated patch of the chosen image by this prototype:')
                #plt.axis('off')
                plt.imsave(os.path.join(self.save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                        'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
                                        high_act_patch)
                        #np.stack((high_act_patch,), axis=-1))#added by icxel,), axis=-1) #added by icxel)
                self.log('most highly activated patch by this prototype shown in the original image:')
                self.imsave_with_bbox(fname=os.path.join(self.save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                    'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
                                img_rgb=original_img,
                                bbox_height_start=high_act_patch_indices[0],
                                bbox_height_end=high_act_patch_indices[1],
                                bbox_width_start=high_act_patch_indices[2],
                                bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
        
                # show the image overlayed with prototype activation map
                rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
                rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
                heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[...,::-1]
                overlayed_img = 0.5 * original_img + 0.3 * heatmap
                self.log('prototype activation map of the chosen image:')
                #plt.axis('off')
                plt.imsave(os.path.join(self.save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                        'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt),
                        overlayed_img)
                self.log('--------------------------------------------------------------')
                prototype_cnt += 1
            self.log('***************************************************************')

        if self.predicted_cls == self.correct_cls:
            self.log('Prediction is correct.')
        else:
            self.log('Prediction is wrong.')

        self.logclose()


# tmux -> shortcuts (next thing to do is for tmux, run even if detached)


# doesn't use same prototypes (recal, second most activated should now be
# first?) ---> keep deleting first over several testing images

# doesn't use same prototypes (recalc, second most activated should now be first?
# ----> delete +1 prototype )



# set weight = 0 for the highest prototype, then for the second best, 
# and so on but keep 10 top-k the same
# other option would be to choose a random prototype instead of setting weight to 0