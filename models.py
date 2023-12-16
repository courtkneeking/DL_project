import torch
import pandas as pd
from torchvision import models
from utils import xavier_init, he_init, prepare

def load_models_using(device, num_classes):    
    # Load a AlexNet model
    a_net = models.alexnet()
    a_net, alex_optimizer = prepare(a_net, xavier_init, device, num_classes)

    # Load VGG16
    vgg_16 = models.vgg16()
    vgg_16, vgg16_optimizer = prepare(vgg_16, xavier_init, device, num_classes)

    # Load VGG19
    vgg_19 = models.vgg19()
    vgg_19, vgg19_optimizer = prepare(vgg_19, xavier_init, device, num_classes)

    # Load ResNet18
    resnet_18 = models.resnet18()
    resnet_18, res18_optimizer = prepare(resnet_18, he_init, device, num_classes)

    # Load ResNet50
    resnet_50 = models.resnet50()
    resnet_50, res50_optimizer = prepare(resnet_50, he_init, device, num_classes)

    # Load ResNet101
    resnet_101 = models.resnet101()
    resnet_101, res101_optimizer = prepare(resnet_101, he_init, device, num_classes)
    
    deep_models = [
        ['AlexNet', 'VGG16', 'VGG19', 'ResNet18', 'ResNet50', 'ResNet101'],
        [a_net, vgg_16, vgg_19, resnet_18, resnet_50, resnet_101],
        [alex_optimizer, vgg16_optimizer, vgg19_optimizer, res18_optimizer, res50_optimizer, res101_optimizer]
    ]

    return deep_models

def get_model_results(model, name, columns):
    data = pd.DataFrame()
    for result in model:
        r = pd.DataFrame(result)
        data = pd.concat([data, r], axis=1)
        
    data.columns = [f"{name}_{x}" for x in columns]
    return data