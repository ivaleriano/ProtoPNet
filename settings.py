base_architecture = 'densenet169'
img_size = 139
prototype_shape = (30, 128, 1, 1) #(2000,128,1,1)
num_classes = 2
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '001'

data_path = '/mnt/nas/Users/Sebastian/adni-mri-pet/classification-nomci/mri-pet'
#train_dir = data_path + '/0-train.h5'
#train_dir = [data_path + '/0-train.h5', data_path + '/1-train.h5', data_path + '/2-train.h5', data_path + '/3-train.h5', data_path + '/4-train.h5']
train_dir = [data_path + '/0-train.h5', data_path + '/1-train.h5']
#test_dir = data_path + '/0-test.h5'
test_dir = [data_path + '/0-test.h5', data_path + '/1-test.h5', data_path + '/2-test.h5', data_path + '/3-test.h5', data_path + '/4-test.h5']
#test_dir = [data_path + '/0-test.h5', data_path + '/1-test.h5', data_path + '/2-test.h5', data_path + '/3-test.h5',]
train_push_dir = data_path + '/0-train.h5'
train_batch_size = 100
test_batch_size = 50
train_push_batch_size = 100

joint_optimizer_lrs = {'features': 1e-4, # changed by Icxel 1e-4
                       'add_on_layers': 3e-4, #3e-3
                       'prototype_vectors': 3e-4} #3e-3
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-4, #3e-3
                      'prototype_vectors': 3e-4} #3e-3

last_layer_optimizer_lr = 1e-4 # originally 1e-4 changed by Icxel

coefs = {
    'crs_ent': 1,
    'clst': 0.8, # originally 0.8
    'sep': -0.08, # originally -0.08
    'l1': 1e-4, #1e-4
}

num_train_epochs = 200
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
