import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from typing import List, Sequence, Tuple

from resnet_features_multimodal import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features

from receptive_field_multimodal import compute_proto_layer_rf_info_v2

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

class DualResNet(nn.Module):
    def __init__(self, features, img_size, prototype_shape, proto_layer_rf_info, num_classes, init_weights=True, prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):
        super().__init__()
        self.ppnet1 = resnet18_features(num_classes=num_classes)
        self.ppnet2 = resnet18_features(num_classes=num_classes)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.features = features
        self.num_prototypes = prototype_shape[0] 
        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')
        print(first_add_on_layer_in_channels)
        self.fc1 = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels*2, out_channels=prototype_shape[1],
                    kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=prototype_shape[1], out_channels=prototype_shape[1],
                    kernel_size=1),
                nn.Sigmoid()
                )
        self.fc2 = nn.Linear(19200, num_classes,
                                    bias=False) # do not use bias
        #self.fc1 = nn.Sequential(
        #    nn.Linear(2 * 8 * n_basefilters, 4 * n_basefilters),
        #    nn.PReLU(),
        #)
        #self.dropout = nn.Dropout(p=0.4)
        #self.fc2 = nn.Linear(4 * n_basefilters, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("t1", "fdg")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, t1, fdg):
        out1 = self.ppnet1(t1)
        out2 = self.ppnet2(fdg)

        out = torch.cat([out1["feature_map"], out2["feature_map"]], dim=1) # comment for notebook
        out = self.global_pool(out)
        out = self.fc1(out)
        out = self.fc2(out)
        #out = torch.cat([out1["feature_map"]], dim=1)
        #out = self.fc1(out)
       # out = self.dropout(out)
        #out = self.fc2(out)

        return {"logits": out}

class DualPPNet(nn.Module):
    def __init__(self, features, img_size, prototype_shape, proto_layer_rf_info, num_classes, init_weights=True, prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):
        super().__init__()
        self.ppnet1 = PPNet(features, img_size, prototype_shape, proto_layer_rf_info["t1"], num_classes,
                            init_weights, prototype_activation_function, add_on_layers_type)
        self.ppnet2 = PPNet(features, img_size, prototype_shape, proto_layer_rf_info["fdg"], num_classes,
                            init_weights, prototype_activation_function, add_on_layers_type)
        #self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.prototype_vectors = nn.Parameter(torch.rand(prototype_shape),
                                              requires_grad=True)   # commented for baseline icxel
        self.features = features
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0] 
        self.num_classes = num_classes
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes) #commented for baseline Icxel
        num_prototypes_per_class = self.num_prototypes // self.num_classes #commented for baseline Icxel
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1 #commented for baseline Icxel

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')
        #print(first_add_on_layer_in_channels)

        self.fc1 = nn.Sequential(
               nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=prototype_shape[1],
                   kernel_size=1),
               nn.ReLU(),
               nn.Conv2d(in_channels=prototype_shape[1], out_channels=prototype_shape[1],
                   kernel_size=1),
               nn.Sigmoid()
               )
        self.fc2 = nn.Linear(self.num_prototypes, num_classes,
                                   bias=False) # do not use bias (= last_layer)
        
        #self.fc1 = nn.Sequential(
        #    nn.Linear(2 * 8 * n_basefilters, 4 * n_basefilters),
        #    nn.PReLU(),
        #)
        #self.dropout = nn.Dropout(p=0.4)
        #self.fc2 = nn.Linear(4 * n_basefilters, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("t1", "fdg")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, t1, fdg):
        out1, min_distances1 = self.ppnet1(t1)
        out2, min_distances2 = self.ppnet2(fdg)
        out = torch.cat([out1, out2], dim=1)
        min_distances = min_distances1 + min_distances2
        #out = torch.cat([out1["feature_map"], out2["feature_map"]], dim=1) # comment for notebook
        #out = torch.cat([out1, out2], dim=1) # comment for notebook  - lATEST
        #out = self.global_pool(out)
        #out = self.fc1(out)
        #out = self.fc2(out)
        #out = torch.cat([out1["feature_map"]], dim=1)
        #out = self.fc1(out)
       # out = self.dropout(out)
        #out = self.fc2(out)

        return (out, min_distances)
        #return {"logits": out}
       # return {'t1':(out1,min_distances1),'fdg':(out2, min_distances2)}
     
    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances
        
    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)
        distances = self._l2_convolution(conv_output)
        return conv_output, distances

class PPNet(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape  #commented for baseline Icxel
        self.num_prototypes = prototype_shape[0]  # commmented for baseline Icxel
        self.num_classes = num_classes
        self.epsilon = 1e-4
        
        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function #commented for baseline Icxel

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0) #commented for baseline Icxel
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes) #commented for baseline Icxel

        num_prototypes_per_class = self.num_prototypes // self.num_classes #commented for baseline Icxel
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1 #commented for baseline Icxel

        self.proto_layer_rf_info = proto_layer_rf_info #commented for baseline Icxel

        # this has to be named features to allow the precise loading
        self.features = features

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)   # commented for baseline icxel

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)          # commented for baseline icxel

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x):
        distances = self.prototype_distances(x)
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        # global min pooling
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)
        return logits, min_distances

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)
        distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)

        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...],
                                 requires_grad=False)
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size mri: {},\n'
            '\timg_size pet: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info mri: {},\n'
            '\tproto_layer_rf_info pet: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size["t1"],
                          self.img_size["fdg"],
                          self.prototype_shape,
                          self.proto_layer_rf_info["t1"],
                          self.proto_layer_rf_info["fdg"],
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)



def construct_DualResNet(base_architecture, pretrained=False, img_size= {"t1": 138, "fdg": 161},#224,
                    prototype_shape=(2000, 512, 1, 1), num_classes=200,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = dict()
    proto_layer_rf_info["t1"] = compute_proto_layer_rf_info_v2(img_size=img_size["t1"],
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    proto_layer_rf_info["fdg"] = compute_proto_layer_rf_info_v2(img_size=img_size["fdg"],
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])

    return DualResNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)

def construct_DualPPNet(base_architecture, pretrained=False, img_size = {"t1":138, "fdg":161}, #img_size=224,
                    prototype_shape=(2000, 512, 1, 1), num_classes=200,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = dict()
    proto_layer_rf_info["t1"] = compute_proto_layer_rf_info_v2(img_size=img_size["t1"],
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    proto_layer_rf_info["fdg"] = compute_proto_layer_rf_info_v2(img_size=img_size["fdg"],
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])                                                
                                                         
    return DualPPNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)