base_architecture='resnet18'
img_size_mri=138
img_size_pet = 161
prototype_shape=(30,128,1,1) #(2000,128,1,1)
num_classes=2
prototype_activation_function='log'
add_on_layers_type='regular'

experiment_run='018'

data_path = '/mnt/nas/Users/Sebastian/adni-mri-pet/registered/classification-nomci/mri-pet'
train_dir = [data_path + '/1-train.h5']
#train_dir = [data_path + '/0-train.h5', data_path + '/1-train.h5', data_path + '/2-train.h5', data_path + '/3-train.h5', data_path + '/4-train.h5']
test_dir = [data_path + '/1-valid.h5']
#test_dir = [data_path + '/0-test.h5', data_path + '/1-test.h5', data_path + '/2-test.h5', data_path + '/3-test.h5', data_path + '/4-test.h5']
train_push_dir = data_path + '/1-train.h5'
train_batch_size=150
test_batch_size=75
train_push_batch_size=150
# changed by Icxel 1e-4 #3e-3  #3e-3
joint_optimizer_lrs={"features":5e-4,"add_on_layers":5e-4,"prototype_vectors":5e-4}
joint_lr_step_size=50

#3e-3 #3e-3
warm_optimizer_lrs={"add_on_layers":3e-3,"prototype_vectors":3e-3} #3e-3

last_layer_optimizer_lr=1e-3 # originally 1e-4 changed by Icxel

# originally 1, 0.8, -0.08, 1e-4
coefs={"crs_ent":1,"clst":0.8,"sep":-0.08,"l1":1e-4}
num_train_epochs=201
num_warm_epochs=0

push_start=10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
