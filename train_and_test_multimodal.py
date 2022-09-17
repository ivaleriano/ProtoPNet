import time
from typing import OrderedDict
import torch

from helpers import list_of_distances, make_one_hot

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct_t1 = 0
    n_correct_fdg = 0
    n_correct_sum = 0
    n_batches = 0
    total_cross_entropy_t1 = 0
    total_cross_entropy_fdg = 0
    total_cross_entropy_sum = 0
    total_cluster_cost_t1 = 0
    total_cluster_cost_fdg = 0
    total_cluster_cost_sum = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost_t1 = 0
    total_separation_cost_fdg = 0
    total_separation_cost_sum = 0
    total_avg_separation_cost_sum = 0
    loss_sum = 0
    loss_t1_sum = 0
    loss_fdg_sum = 0

    for i, (mri_image, pet_image, label) in enumerate(dataloader):
        input1 = mri_image.cuda()
        input2 = pet_image.cuda()
        target = label.cuda()
        
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            output, min_distances, output_t1,min_distances_t1,output_fdg,min_distances_fdg = model(input1,input2)
            #output_t1,min_distances_t1 = output['t1'][0], output['t1'][1]
            #output_fdg, min_distances_fdg = output['fdg'][0], output['fdg'][1]

            # compute loss
            cross_entropy_t1 = torch.nn.functional.cross_entropy(output_t1, target)
            cross_entropy_fdg = torch.nn.functional.cross_entropy(output_fdg, target)
            cross_entropy_sum = torch.nn.functional.cross_entropy(output, target)
            # below commented for resnet multimodal icxel
            if class_specific:
                max_dist = (model.module.prototype_shape[1]*2
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])
                max_dist_single_mode = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                # below commented for multimodal icxel
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
                inverted_distances_sum, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                inverted_distances_t1, _ = torch.max((max_dist_single_mode - min_distances_t1) * prototypes_of_correct_class[:,:30], dim=1)
                inverted_distances_fdg, _ = torch.max((max_dist_single_mode - min_distances_fdg) * prototypes_of_correct_class[:,30:], dim=1)
                cluster_cost_t1 = torch.mean(max_dist_single_mode - inverted_distances_t1)
                cluster_cost_fdg = torch.mean(max_dist_single_mode - inverted_distances_fdg)
                cluster_cost_sum = torch.mean(max_dist - inverted_distances_sum)

                # calculate separation cost
                # below commented for multimodal icxel
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes_t1, _ = \
                    torch.max((max_dist_single_mode - min_distances_t1) * prototypes_of_wrong_class[:,:30], dim=1)
                inverted_distances_to_nontarget_prototypes_fdg, _ = \
                    torch.max((max_dist_single_mode - min_distances_fdg) * prototypes_of_wrong_class[:,30:], dim=1)
                inverted_distances_to_nontarget_prototypes_sum, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost_t1 = torch.mean(max_dist_single_mode - inverted_distances_to_nontarget_prototypes_t1)
                separation_cost_fdg = torch.mean(max_dist_single_mode - inverted_distances_to_nontarget_prototypes_fdg)
                separation_cost_sum = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes_sum)

                # calculate avg cluster cost
                # below commented for multimodal icxel
                avg_separation_cost_sum = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost_sum = torch.mean(avg_separation_cost_sum)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.fc2.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.fc2.weight.norm(p=1) 

            # commented for multimodal icxel
            else:
                min_distances_t1, _ = torch.min(min_distances_t1, dim=1)
                min_distances_fdg, _ = torch.min(min_distances_fdg, dim=1)
                min_distances, _ = torch.min(min_distances, dim=1)
                cluster_cost_t1 = torch.mean(min_distances_t1)
                cluster_cost_fdg = torch.mean(min_distances_fdg)
                cluster_cost_sum = torch.mean(min_distances)
                l1 = model.module.fc2.weight.norm(p=1)

            # evaluation statistics
            _, predicted_t1 = torch.max(output_t1.data, 1)
            _, predicted_fdg = torch.max(output_fdg.data, 1)
            _, predicted_sum = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct_t1 += (predicted_t1 == target).sum().item()
            n_correct_fdg += (predicted_fdg == target).sum().item()
            n_correct_sum += (predicted_sum == target).sum().item()

            n_batches += 1
            total_cross_entropy_t1 += cross_entropy_t1.item()
            total_cross_entropy_fdg += cross_entropy_fdg.item()
            total_cross_entropy_sum += cross_entropy_sum.item()
            total_cluster_cost_t1 += cluster_cost_t1.item()  #commented for multimodal icxel
            total_cluster_cost_fdg += cluster_cost_fdg.item()  #commented for multimodal icxel
            total_cluster_cost_sum += cluster_cost_sum.item()  #commented for multimodal icxel
            total_separation_cost_t1 += separation_cost_t1.item() #commented for multimodal icxel
            total_separation_cost_fdg += separation_cost_fdg.item() #commented for multimodal icxel
            total_separation_cost_sum += separation_cost_sum.item() #commented for multimodal icxel
            total_avg_separation_cost_sum += avg_separation_cost_sum.item() #commented for multimodal icxel

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss_t1 = (coefs['crs_ent'] * cross_entropy_t1
                          + coefs['clst'] * cluster_cost_t1
                          + coefs['sep'] * separation_cost_t1
                          + coefs['l1'] * l1)  #commented for multimodal icxel
                    loss_fdg = (coefs['crs_ent'] * cross_entropy_fdg
                          + coefs['clst'] * cluster_cost_fdg
                          + coefs['sep'] * separation_cost_fdg
                          + coefs['l1'] * l1)  #commented for multimodal icxel
                    total_loss = (coefs['crs_ent'] * cross_entropy_sum
                          + coefs['clst'] * cluster_cost_sum
                          + coefs['sep'] * separation_cost_sum
                          + coefs['l1'] * l1)  #commented for multimodal icxel
                else:
                    loss_t1 = cross_entropy_t1 + 0.8 * cluster_cost_t1 - 0.08 * separation_cost_t1 + 1e-4 * l1 #commented for multimodal icxel
                    loss_fdg = cross_entropy_fdg + 0.8 * cluster_cost_fdg - 0.08 * separation_cost_fdg + 1e-4 * l1 #commented for multimodal icxel
                    total_loss = cross_entropy_sum + 0.8 * cluster_cost_sum - 0.08 * separation_cost_sum + 1e-4 * l1 #commented for multimodal icxel
            else:
                if coefs is not None:
                    loss_t1 = (coefs['crs_ent'] * cross_entropy_t1
                          + coefs['clst'] * cluster_cost_t1
                          + coefs['l1'] * l1) #commented for multimodal icxel
                    loss_fdg = (coefs['crs_ent'] * cross_entropy_fdg
                          + coefs['clst'] * cluster_cost_fdg
                          + coefs['l1'] * l1) #commented for multimodal icxel
                    total_loss = (coefs['crs_ent'] * cross_entropy_sum
                          + coefs['clst'] * cluster_cost_sum
                          + coefs['l1'] * l1)  #commented for multimodal icxel
                else:
                    loss_t1 = cross_entropy_t1 + 0.8 * cluster_cost_t1 + 1e-4 * l1 #commented for multimodal icxel
                    loss_fdg = cross_entropy_fdg + 0.8 * cluster_cost_fdg + 1e-4 * l1 #commented for multimodal icxel
                    total_loss = cross_entropy_sum + 0.8 * cluster_cost_sum + 1e-4 * l1 #commented for multimodal icxel
            loss_sum = loss_sum + total_loss
            loss_t1_sum = loss_t1_sum + loss_t1
            loss_fdg_sum = loss_fdg_sum + loss_fdg
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        del input1
        del input2
        del target
        del output
        del output_t1
        del output_fdg
        del predicted_t1
        del predicted_fdg
        del predicted_sum
        del min_distances_t1
        del min_distances_fdg
        del min_distances

    end = time.time()

    '''log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent fdg: \t{0}'.format(total_cross_entropy_fdg / n_batches))
    log('\tcluster fdg: \t{0}'.format(total_cluster_cost_fdg / n_batches)) #commented for multimodal icxel
    if class_specific:   #commented multimodal icxel
       log('\tseparation fdg:\t{0}'.format(total_separation_cost_fdg / n_batches)) #commented multimodal icxel
       log('\tavg separation fdg:\t{0}'.format(total_avg_separation_cost_fdg / n_batches)) #commented multimodal icxel
    log('\taccu fdg: \t\t{0}%'.format(n_correct_fdg / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.module.fc2.weight.norm(p=1).item())) #commented mulimodal icxel   '''

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy_sum / n_batches))
    log('\tcross ent t1: \t{0}'.format(total_cross_entropy_t1 / n_batches))
    log('\tcross ent fdg: \t{0}'.format(total_cross_entropy_fdg / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost_sum / n_batches)) #commented for multimodal 
    log('\tcluster t1: \t{0}'.format(total_cluster_cost_t1 / n_batches)) #commented for multimodal icxel
    log('\tcluster fdg: \t{0}'.format(total_cluster_cost_fdg / n_batches)) #commented for multimodal icxel
    if class_specific:   #commented multimodal icxel
       log('\tseparation:\t{0}'.format(total_separation_cost_sum / n_batches)) #commented multimodal icxel
       log('\tseparation t1:\t{0}'.format(total_separation_cost_t1 / n_batches)) #commented multimodal icxel
       log('\tseparation fdg:\t{0}'.format(total_separation_cost_fdg / n_batches)) #commented multimodal icxel
       log('\tavg separation:\t{0}'.format(total_avg_separation_cost_sum / n_batches)) #commented multimodal icxel
    log('\taccu: \t\t{0}%'.format(n_correct_sum / n_examples * 100))
    log('\taccu t1: \t\t{0}%'.format(n_correct_t1 / n_examples * 100))
    log('\taccu fdg: \t\t{0}%'.format(n_correct_fdg / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.module.fc2.weight.norm(p=1).item())) #commented mulimodal icxel 
    log('\tloss: \t\t{0}%'.format(loss_sum / n_batches * 100))
    log('\tloss t1: \t\t{0}%'.format(loss_t1_sum / n_batches * 100))
    log('\tloss fdg: \t\t{0}%'.format(loss_fdg_sum / n_batches * 100))
    
    return {
        'sum': n_correct_sum / n_examples,
        'cross_ent': total_cross_entropy_sum / n_batches,
        'cluster': total_cluster_cost_sum / n_batches,
        'separation':total_separation_cost_sum / n_batches,
        'loss': loss_sum / n_batches,
        't1': n_correct_t1 / n_examples, 
        'cross_ent_t1': total_cross_entropy_t1 / n_batches,
        'cluster_t1': total_cluster_cost_t1 / n_batches,
        'separation_t1':total_separation_cost_t1 / n_batches,
        'loss_t1': loss_t1_sum / n_batches,
        'fdg': n_correct_fdg / n_examples,
        'cross_ent_fdg': total_cross_entropy_fdg / n_batches,
        'cluster_fdg': total_cluster_cost_fdg / n_batches,
        'separation_fdg':total_separation_cost_fdg / n_batches,
        'loss_fdg': loss_fdg_sum / n_batches,
    }


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)


def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.fc1.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False #commented multimodal icxel
    for p in model.module.fc2.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.fc1.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True #commented multimodal icxel
    for p in model.module.fc2.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.fc1.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True #commented multimodal icxel
    for p in model.module.fc2.parameters():
        p.requires_grad = True
    
    log('\tjoint')


    # optuna optimizer instead of grid search - check it out (return performance)
    # hyperband -parallelization (training on 2 gpu's) ---->>>>> memory issue ()
        # do optimization only on validation set (not test)
    # compare with baseline
    # validation - effect on prototypes?
