import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time

from receptive_field_multimodal import compute_rf_prototype
from helpers import makedir, find_high_activation_crop

# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None,
                    class_to_push='both'):

    prototype_network_parallel.eval()
    if class_to_push == 'both':
        log('\tpush both')
    elif class_to_push == 't1':
        log('\tpush t1')
    elif class_to_push == 'fdg':
        log('\tpush fdg')
    
    global_min_proto_dist_dict = dict()
    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist_dict['t1'] = np.full(n_prototypes, np.inf)
    global_min_proto_dist_dict['fdg'] = np.full(n_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches_dict = dict()
    global_min_fmap_patches_dict['t1'] = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])
    global_min_fmap_patches_dict['fdg'] = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    proto_rf_boxes_dict = dict()
    proto_bound_boxes_dict = dict()
    if save_prototype_class_identity:
        proto_rf_boxes_dict['t1'] = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)
        proto_rf_boxes_dict['fdg'] = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)                                    
        proto_bound_boxes_dict['t1'] = np.full(shape=[n_prototypes, 6],
                                            fill_value=-1)
        proto_bound_boxes_dict['fdg'] = np.full(shape=[n_prototypes, 6],
                                            fill_value=-1)
    else:
        proto_rf_boxes_dict['t1'] = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)
        proto_rf_boxes_dict['fdg'] = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)
        proto_bound_boxes_dict['t1'] = np.full(shape=[n_prototypes, 5],
                                            fill_value=-1)
        proto_bound_boxes_dict['fdg'] = np.full(shape=[n_prototypes, 5],
                                            fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.module.num_classes

    for push_iter, (search_batch_input_t1, search_batch_input_fdg, search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        start_index_of_search_batch = push_iter * search_batch_size
        search_batch_input = {'t1': search_batch_input_t1,
                              'fdg': search_batch_input_fdg}

        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_min_proto_dist_dict,
                                   global_min_fmap_patches_dict,
                                   proto_rf_boxes_dict,
                                   proto_bound_boxes_dict,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '_mri.npy'),
                proto_rf_boxes_dict['t1'])
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '_fdg.npy'),
                proto_rf_boxes_dict['fdg'])
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '_mri.npy'),
                proto_bound_boxes_dict['t1'])
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '_fdg.npy'),
                proto_bound_boxes_dict['fdg'])

    log('\tExecuting push ...')
    prototype_update_mri = np.reshape(global_min_fmap_patches_dict['t1'],
                                  tuple(prototype_shape))
    prototype_update_fdg = np.reshape(global_min_fmap_patches_dict['fdg'],
                                  tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update_mri, dtype=torch.float32).cuda())
    prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update_fdg, dtype=torch.float32).cuda())
    # prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))

# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist_dict, # this will be updated
                               global_min_fmap_patches_dict, # this will be updated
                               proto_rf_boxes_dict, # this will be updated
                               proto_bound_boxes_dict, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        # print('preprocessing input for pushing ...')
        # search_batch = copy.deepcopy(search_batch_input)
        search_batch = preprocess_input_function(search_batch_input)

    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch['t1'] = search_batch['t1'].cuda()
        search_batch['fdg'] = search_batch['fdg'].cuda()
        # this computation currently is not parallelized
        protoL_input_torch = dict()
        proto_dist_torch = dict()
        protoL_input_torch['t1'], proto_dist_torch['t1'] = prototype_network_parallel.module.ppnet1.push_forward(search_batch['t1'])
        protoL_input_torch['fdg'], proto_dist_torch['fdg'] = prototype_network_parallel.module.ppnet2.push_forward(search_batch['fdg'])
    

    protoL_input_ = dict()
    proto_dist_ = dict()
    protoL_input_['t1'] = np.copy(protoL_input_torch['t1'].detach().cpu().numpy())
    proto_dist_['t1'] = np.copy(proto_dist_torch['t1'].detach().cpu().numpy())
    protoL_input_['fdg'] = np.copy(protoL_input_torch['fdg'].detach().cpu().numpy())
    proto_dist_['fdg'] = np.copy(proto_dist_torch['fdg'].detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    for j in range(n_prototypes):
        #if n_prototypes_per_class != None:
        if class_specific:
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = dict()
            proto_dist_j['t1'] = proto_dist_['t1'][class_to_img_index_dict[target_class]][:,j,:,:]
            proto_dist_j['fdg'] = proto_dist_['fdg'][class_to_img_index_dict[target_class]][:,j,:,:]
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j['t1'] = proto_dist_['t1'][:,j,:,:]
            proto_dist_j['fdg'] = proto_dist_['fdg'][:,j,:,:]

        batch_min_proto_dist_j = dict()
        batch_min_proto_dist_j['t1'] = np.amin(proto_dist_j['t1'])
        batch_min_proto_dist_j['fdg'] = np.amin(proto_dist_j['fdg'])
        batch_argmin_proto_dist_j = dict()
        
        img_index_in_batch = dict()
        fmap_height_start_index = dict()
        fmap_width_start_index = dict()
        fmap_height_end_index = dict()
        fmap_width_end_index = dict()
        batch_min_fmap_patch_j = dict()
        rf_prototype_j = dict()
        protoL_rf_info = dict()
        original_img_j = dict()
        original_img_size = dict()
        rf_img_j = dict()
        proto_dist_img_j = dict()
        proto_act_img_j = dict()
        upsampled_act_img_j = dict()
        proto_bound_j = dict()
        proto_img_j = dict()
        rescaled_act_img_j = dict()
        heatmap = dict()
        overlayed_original_img_j = dict()
        overlayed_rf_img_j = dict()

        if batch_min_proto_dist_j['t1'] < global_min_proto_dist_dict['t1'][j] and batch_min_proto_dist_j['fdg'] < global_min_proto_dist_dict['fdg'][j]:
            batch_argmin_proto_dist_j['t1']  = \
                list(np.unravel_index(np.argmin(proto_dist_j['t1'], axis=None),
                                    proto_dist_j['t1'].shape))
            batch_argmin_proto_dist_j['fdg']  = \
                list(np.unravel_index(np.argmin(proto_dist_j['fdg'], axis=None),
                                    proto_dist_j['fdg'].shape))
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''
                batch_argmin_proto_dist_j['t1'][0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j['t1'][0]]
                batch_argmin_proto_dist_j['fdg'][0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j['fdg'][0]]
            
            # retrieve the corresponding feature map patch
            img_index_in_batch['t1'] = batch_argmin_proto_dist_j['t1'][0]
            fmap_height_start_index['t1'] = batch_argmin_proto_dist_j['t1'][1] * prototype_layer_stride
            fmap_height_end_index['t1'] = fmap_height_start_index['t1'] + proto_h
            fmap_width_start_index['t1'] = batch_argmin_proto_dist_j['t1'][2] * prototype_layer_stride
            fmap_width_end_index['t1'] = fmap_width_start_index['t1'] + proto_w

            batch_min_fmap_patch_j['t1'] = protoL_input_['t1'][img_index_in_batch['t1'],
                                                   :,
                                                   fmap_height_start_index['t1']:fmap_height_end_index['t1'],
                                                   fmap_width_start_index['t1']:fmap_width_end_index['t1']]

            global_min_proto_dist_dict['t1'][j] = batch_min_proto_dist_j['t1']
            global_min_fmap_patches_dict['t1'][j] = batch_min_fmap_patch_j['t1']

            img_index_in_batch['fdg'] = batch_argmin_proto_dist_j['fdg'][0]
            fmap_height_start_index['fdg'] = batch_argmin_proto_dist_j['fdg'][1] * prototype_layer_stride
            fmap_height_end_index['fdg'] = fmap_height_start_index['fdg'] + proto_h
            fmap_width_start_index['fdg'] = batch_argmin_proto_dist_j['fdg'][2] * prototype_layer_stride
            fmap_width_end_index['fdg'] = fmap_width_start_index['fdg'] + proto_w

            batch_min_fmap_patch_j['fdg'] = protoL_input_['fdg'][img_index_in_batch['fdg'],
                                                   :,
                                                   fmap_height_start_index['fdg']:fmap_height_end_index['fdg'],
                                                   fmap_width_start_index['fdg']:fmap_width_end_index['fdg']]

            global_min_proto_dist_dict['fdg'][j] = batch_min_proto_dist_j['fdg']
            global_min_fmap_patches_dict['fdg'][j] = batch_min_fmap_patch_j['fdg']

            # get the receptive field boundary of the image patch
            # that generates the representation
            protoL_rf_info['fdg'] = prototype_network_parallel.module.ppnet2.proto_layer_rf_info
            rf_prototype_j['fdg'] = compute_rf_prototype(search_batch['fdg'].size(2), batch_argmin_proto_dist_j['fdg'], protoL_rf_info['fdg'])
            
            protoL_rf_info['t1'] = prototype_network_parallel.module.ppnet1.proto_layer_rf_info
            rf_prototype_j['t1'] = compute_rf_prototype(search_batch['t1'].size(2), batch_argmin_proto_dist_j['t1'], protoL_rf_info['t1'])

            # get the whole image
            original_img_j['t1'] = search_batch_input['t1'][rf_prototype_j['t1'][0]]
            original_img_j['t1'] = original_img_j['t1'].numpy()
            original_img_j['t1'] = np.transpose(original_img_j['t1'], (1, 2, 0))
            original_img_size['t1'] = original_img_j['t1'].shape[0]

            original_img_j['fdg'] = search_batch_input['fdg'][rf_prototype_j['fdg'][0]]
            original_img_j['fdg'] = original_img_j['fdg'].numpy()
            original_img_j['fdg'] = np.transpose(original_img_j['fdg'], (1, 2, 0))
            original_img_size['fdg'] = original_img_j['fdg'].shape[0]

            # crop out the receptive field
            rf_img_j['t1'] = original_img_j['t1'][rf_prototype_j['t1'][1]:rf_prototype_j['t1'][2],
                                      rf_prototype_j['t1'][3]:rf_prototype_j['t1'][4], :]
            rf_img_j['fdg'] = original_img_j['fdg'][rf_prototype_j['fdg'][1]:rf_prototype_j['fdg'][2],
                                      rf_prototype_j['fdg'][3]:rf_prototype_j['fdg'][4], :]

            # save the prototype receptive field information
            proto_rf_boxes_dict['t1'][j, 0] = rf_prototype_j['t1'][0] + start_index_of_search_batch
            proto_rf_boxes_dict['t1'][j, 1] = rf_prototype_j['t1'][1]
            proto_rf_boxes_dict['t1'][j, 2] = rf_prototype_j['t1'][2]
            proto_rf_boxes_dict['t1'][j, 3] = rf_prototype_j['t1'][3]
            proto_rf_boxes_dict['t1'][j, 4] = rf_prototype_j['t1'][4]
            if proto_rf_boxes_dict['t1'].shape[1] == 6 and search_y is not None:
                proto_rf_boxes_dict['t1'][j, 5] = search_y[rf_prototype_j['t1'][0]].item()
            
            proto_rf_boxes_dict['fdg'][j, 0] = rf_prototype_j['fdg'][0] + start_index_of_search_batch
            proto_rf_boxes_dict['fdg'][j, 1] = rf_prototype_j['fdg'][1]
            proto_rf_boxes_dict['fdg'][j, 2] = rf_prototype_j['fdg'][2]
            proto_rf_boxes_dict['fdg'][j, 3] = rf_prototype_j['fdg'][3]
            proto_rf_boxes_dict['fdg'][j, 4] = rf_prototype_j['fdg'][4]
            if proto_rf_boxes_dict['fdg'].shape[1] == 6 and search_y is not None:
                proto_rf_boxes_dict['fdg'][j, 5] = search_y[rf_prototype_j['fdg'][0]].item()

            # find the highly activated region of the original image
            proto_dist_img_j['t1'] = proto_dist_['t1'][img_index_in_batch['t1'], j, :, :]
            if prototype_network_parallel.module.ppnet1.prototype_activation_function == 'log':
                proto_act_img_j['t1'] = np.log((proto_dist_img_j['t1'] + 1) / (proto_dist_img_j['t1'] + prototype_network_parallel.module.ppnet1.epsilon))
            elif prototype_network_parallel.module.ppnet1.prototype_activation_function == 'linear':
                proto_act_img_j['t1'] = max_dist - proto_dist_img_j['t1']
            else:
                proto_act_img_j['t1'] = prototype_activation_function_in_numpy(proto_dist_img_j['t1'])

            upsampled_act_img_j['t1'] = cv2.resize(proto_act_img_j['t1'], dsize=(original_img_size['t1'], original_img_size['t1']),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j['t1'] = find_high_activation_crop(upsampled_act_img_j['t1'])

            proto_dist_img_j['fdg'] = proto_dist_['fdg'][img_index_in_batch['fdg'], j, :, :]
            if prototype_network_parallel.module.ppnet2.prototype_activation_function == 'log':
                proto_act_img_j['fdg'] = np.log((proto_dist_img_j['fdg'] + 1) / (proto_dist_img_j['fdg'] + prototype_network_parallel.module.ppnet2.epsilon))
            elif prototype_network_parallel.module.ppnet2.prototype_activation_function == 'linear':
                proto_act_img_j['fdg'] = max_dist - proto_dist_img_j['fdg']
            else:
                proto_act_img_j['fdg'] = prototype_activation_function_in_numpy(proto_dist_img_j['fdg'])

            upsampled_act_img_j['fdg'] = cv2.resize(proto_act_img_j['fdg'], dsize=(original_img_size['fdg'], original_img_size['fdg']),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j['fdg'] = find_high_activation_crop(upsampled_act_img_j['fdg'])

            # crop out the image patch with high activation as prototype image
            proto_img_j['t1'] = original_img_j['t1'][proto_bound_j['t1'][0]:proto_bound_j['t1'][1],
                                         proto_bound_j['t1'][2]:proto_bound_j['t1'][3], :]
            proto_img_j['fdg'] = original_img_j['fdg'][proto_bound_j['fdg'][0]:proto_bound_j['fdg'][1],
                                         proto_bound_j['fdg'][2]:proto_bound_j['fdg'][3], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes_dict['t1'][j, 0] = proto_rf_boxes_dict['t1'][j, 0]
            proto_bound_boxes_dict['t1'][j, 1] = proto_bound_j['t1'][0]
            proto_bound_boxes_dict['t1'][j, 2] = proto_bound_j['t1'][1]
            proto_bound_boxes_dict['t1'][j, 3] = proto_bound_j['t1'][2]
            proto_bound_boxes_dict['t1'][j, 4] = proto_bound_j['t1'][3]
            if proto_bound_boxes_dict['t1'].shape[1] == 6 and search_y is not None:
                proto_bound_boxes_dict['t1'][j, 5] = search_y[rf_prototype_j['t1'][0]].item()
            
            proto_bound_boxes_dict['fdg'][j, 0] = proto_rf_boxes_dict['fdg'][j, 0]
            proto_bound_boxes_dict['fdg'][j, 1] = proto_bound_j['fdg'][0]
            proto_bound_boxes_dict['fdg'][j, 2] = proto_bound_j['fdg'][1]
            proto_bound_boxes_dict['fdg'][j, 3] = proto_bound_j['fdg'][2]
            proto_bound_boxes_dict['fdg'][j, 4] = proto_bound_j['fdg'][3]
            if proto_bound_boxes_dict['fdg'].shape[1] == 6 and search_y is not None:
                proto_bound_boxes_dict['fdg'][j, 5] = search_y[rf_prototype_j['fdg'][0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                        prototype_self_act_filename_prefix + 'mri_' + str(j) + '.npy'),
                            proto_act_img_j['t1'])
                    np.save(os.path.join(dir_for_saving_prototypes,
                                        prototype_self_act_filename_prefix + 'fdg_'+ str(j) + '.npy'),
                            proto_act_img_j['fdg'])
                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    #img_to_save = np.stack((original_img_j,), axis=-1) #added by icxel
                    if j < 10:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-original_t1_0' + str(j) + '.png'),
                                original_img_j['t1'], #original_img_j, icxel
                                vmin=0.0,
                                vmax=1.0)
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-original_fdg_0' + str(j) + '.png'),
                                original_img_j['fdg'], #original_img_j, icxel
                                vmin=0.0,
                                vmax=1.0)
                    else:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-original_t1_' + str(j) + '.png'),
                                original_img_j['t1'], #original_img_j, icxel
                                vmin=0.0,
                                vmax=1.0)
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-original_fdg_' + str(j) + '.png'),
                                original_img_j['fdg'], #original_img_j, icxel
                                vmin=0.0,
                                vmax=1.0)
                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j['t1'] = upsampled_act_img_j['t1'] - np.amin(upsampled_act_img_j['t1'])
                    rescaled_act_img_j['t1'] = rescaled_act_img_j['t1'] / np.amax(rescaled_act_img_j['t1'])
                    heatmap['t1'] = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j['t1']), cv2.COLORMAP_JET)
                    heatmap['t1'] = np.float32(heatmap['t1']) / 255
                    heatmap['t1'] = heatmap['t1'][...,::-1]
                    overlayed_original_img_j['t1'] = 0.5 * original_img_j['t1'] + 0.3 * heatmap['t1']

                    rescaled_act_img_j['fdg'] = upsampled_act_img_j['fdg'] - np.amin(upsampled_act_img_j['fdg'])
                    rescaled_act_img_j['fdg'] = rescaled_act_img_j['fdg'] / np.amax(rescaled_act_img_j['fdg'])
                    heatmap['fdg'] = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j['fdg']), cv2.COLORMAP_JET)
                    heatmap['fdg'] = np.float32(heatmap['fdg']) / 255
                    heatmap['fdg'] = heatmap['fdg'][...,::-1]
                    overlayed_original_img_j['fdg'] = 0.5 * original_img_j['fdg'] + 0.3 * heatmap['fdg']
                    if j < 10:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-original_with_self_act_t1_0' + str(j) + '.png'),
                                    overlayed_original_img_j['t1'],
                                    vmin=0.0,
                                    vmax=1.0)
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-original_with_self_act_fdg_0' + str(j) + '.png'),
                                    overlayed_original_img_j['fdg'],
                                    vmin=0.0,
                                    vmax=1.0)
                    else:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original_with_self_act_t1_' + str(j) + '.png'),
                               overlayed_original_img_j['t1'],
                               vmin=0.0,
                               vmax=1.0)
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original_with_self_act_fdg_' + str(j) + '.png'),
                               overlayed_original_img_j['fdg'],
                               vmin=0.0,
                               vmax=1.0)
                    
                    # if different from the original (whole) image, save the prototype receptive field as png
                    if ((rf_img_j['t1'].shape[0] != original_img_size['t1'] or rf_img_j['t1'].shape[1] != original_img_size['t1'])
                        and (rf_img_j['fdg'].shape[0] != original_img_size['fdg'] or rf_img_j['fdg'].shape[1] != original_img_size['fdg'])):
                        if j < 10:
                            plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                    prototype_img_filename_prefix + '-receptive_field_t1_0' + str(j) + '.png'),
                                    rf_img_j['t1'],
                                    vmin=0.0,
                                    vmax=1.0)
                            plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                    prototype_img_filename_prefix + '-receptive_field_fdg_0' + str(j) + '.png'),
                                    rf_img_j['fdg'],
                                    vmin=0.0,
                                    vmax=1.0)
                        else:
                            plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field_t1_' + str(j) + '.png'),
                                   rf_img_j['t1'],
                                   vmin=0.0,
                                   vmax=1.0) 
                            plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field_fdg_' + str(j) + '.png'),
                                   rf_img_j['fdg'],
                                   vmin=0.0,
                                   vmax=1.0)
                        overlayed_rf_img_j['t1'] = overlayed_original_img_j['t1'][rf_prototype_j['t1'][1]:rf_prototype_j['t1'][2],
                                                                      rf_prototype_j['t1'][3]:rf_prototype_j['t1'][4]]
                        overlayed_rf_img_j['fdg'] = overlayed_original_img_j['fdg'][rf_prototype_j['fdg'][1]:rf_prototype_j['fdg'][2],
                                                                      rf_prototype_j['fdg'][3]:rf_prototype_j['fdg'][4]]
                        if j < 10:
                            plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                    prototype_img_filename_prefix + '-receptive_field_with_self_act_t1_0' + str(j) + '.png'),
                                    overlayed_rf_img_j['t1'],
                                    vmin=0.0,
                                    vmax=1.0)
                            plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                    prototype_img_filename_prefix + '-receptive_field_with_self_act_fdg_0' + str(j) + '.png'),
                                    overlayed_rf_img_j['fdg'],
                                    vmin=0.0,
                                    vmax=1.0)
                        else:
                            plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field_with_self_act_t1_' + str(j) + '.png'),
                                   overlayed_rf_img_j['t1'],
                                   vmin=0.0,
                                   vmax=1.0)
                            plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field_with_self_act_fdg_' + str(j) + '.png'),
                                   overlayed_rf_img_j['fdg'],
                                   vmin=0.0,
                                   vmax=1.0)
                    
                    # save the prototype image (highly activated region of the whole image)
                    # proto_to_save = np.stack((proto_img_j,), axis=-1) # added icxel
                    if j < 10:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '0' + str(j) + '_t1.png'),
                                proto_img_j['t1'], #proto_img_j, icxel
                                vmin=0.0,
                                vmax=1.0)
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '0' + str(j) + '_fdg.png'),
                                proto_img_j['fdg'], #proto_img_j, icxel
                                vmin=0.0,
                                vmax=1.0)
                    else:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + str(j) + '_t1.png'),
                               proto_img_j['t1'], #proto_img_j, icxel
                               vmin=0.0,
                               vmax=1.0)
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + str(j) + '_fdg.png'),
                               proto_img_j['fdg'], #proto_img_j, icxel
                               vmin=0.0,
                               vmax=1.0)
                
    if class_specific:
        del class_to_img_index_dict
