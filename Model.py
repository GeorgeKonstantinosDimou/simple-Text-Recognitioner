import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

import Backbone
import Recognizer


def rn18_hook(module, input, output):
    global features
    features = output

class ScratchModel(nn.Module):
    def __init__(self, params):
        super(ScratchModel, self).__init__()
        self.flag = True
        self.conv_layer = Backbone.ResNet(params, 18)
        self.rec = Recognizer.Recognizer(params, self.flag)

    def forward(self, image, target=None):
        conv_f = self.conv_layer(image)
        out = self.rec(conv_f, target)
        return out

class ModelPreTrained(nn.Module):
    def __init__(self, params):
        super(ModelPreTrained, self).__init__()
        self.flag = False
        
        self.conv = resnet18(weights = ResNet18_Weights.DEFAULT)
        self.conv.layer1[1].bn2.register_forward_hook(rn18_hook)
        #for param in model_conv.parameters():
        #param.requires_grad = False
        # for name, param in self.conv.named_parameters():
        #     print(name, param.requires_grad)
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 30))
        self.fc = nn.Linear(64, params['feature_layers'])
        
        #self.conv = model_conv
        self.rec = Recognizer.Recognizer(params, self.flag)

    def forward(self, image, target=None):
        #conv_f = self.conv(image)
        _ = self.conv(image)
        #print(features.shape)
        x = self.pool(features).squeeze().permute(0, 2, 1) #[batch_size, seq_length, feature_maps]
        x = self.fc(x)
        #print(features.shape, x.shape)
        #print(conv_f)
        out = self.rec(x.permute(0, 2, 1), target) #[batch_size, feature_maps, seq_length]
        return out
    
    def fc_params(self):
        return self.conv.fc.parameters()
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x