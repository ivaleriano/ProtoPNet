base_architecture='resnet18'
img_size=138
prototype_shape=(30,128,1,1) #(2000,128,1,1)
num_classes=2
prototype_activation_function='log'
add_on_layers_type='regular'

experiment_run='001'

train_batch_size=150
test_batch_size=75
train_push_batch_size=150
# changed by Icxel 1e-4 #3e-3  #3e-3
joint_optimizer_lrs={{"features":1e-4,"add_on_layers":3e-3,"prototype_vectors":3e-3}}
joint_lr_step_size=50

#3e-3 #3e-3
warm_optimizer_lrs={"add_on_layers":3e-3,"prototype_vectors":3e-3} #3e-3

last_layer_optimizer_lr=1e-4 # originally 1e-4 changed by Icxel

# originally 1, 0.8, -0.08, 1e-4
coefs={"crs_ent":1,"clst":0.8,"sep":-0.08,"l1":1e-4}
num_train_epochs=201
num_warm_epochs=0

push_start=10
push_epochs=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240]

DATA_DIR="/mnt/nas/Users/Sebastian/adni-mri-pet/registered/classification-nomci/mri-pet"

for fold in $(seq 1)
do
python shape_continuum/ProtoPNet/main_multimodal.py \
    --base_architecture "${base_architecture}" \
    --img_size ${img_size} \
    --prototype_shape "${prototype_shape}" \
    --num_classes ${num_classes} \
    --prototype_activation_function "${prototype_activation_function}" \
    --experiment_run "${experiment_run}_${fold}" \
    --train_batch_size ${train_batch_size} \
    --test_batch_size ${test_batch_size} \
    --train_push_batch_size ${train_push_batch_size} \
    --joint_optimizer_lrs ${joint_optimizer_lrs} \
    --joint_lr_step_size ${joint_lr_step_size} \
    --warm_optimizer_lrs ${warm_optimizer_lrs} \
    --last_layer_optimizer_lr ${last_layer_optimizer_lr} \
    --coefs ${coefs} \
    --num_train_epochs ${num_train_epochs} \
    --num_warm_epochs ${num_warm_epochs} \
    --push_start ${push_start} \
    --push_epochs ${push_epochs} \
    --train_dir "${DATA_DIR}/${fold}-train.h5" \
    --test_dir "${DATA_DIR}/${fold}-valid.h5"
done