import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None,
                 stride=1):  # identity_downsample is a ConvLayer which we might have to change depending on the number of channels
        super().__init__()
        self.expansion = 4  # by default the ResNet architecture has output 4*times the number of inputs
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # After each block we add the identity
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNetSmallBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None,
                 stride=1):  # identity_downsample is a ConvLayer which we might have to change depending on the number of channels
        super().__init__()
        self.expansion = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #self.identity_downsample = identity_downsample
        self.stride = stride
        

    def forward(self, x):
        # identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # print(identity.shape)
        #print(x.shape)
        # After each block we add the identity
        # if self.identity_downsample is not None:
        #     identity = self.identity_downsample(identity)
        # x += identity
        
        x = self.relu(x) 
        return x


class ResNet(nn.Module):
    def __init__(self, params, depth, img_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        block = ResNetBlock if depth >= 50 else ResNetSmallBlock

        # This is the initialization layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 1, out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, 1,  out_channels=128, stride=1)
        self.layer3 = self._make_layer(block, 1, out_channels=256, stride=1)
        #self.layer4 = self._make_layer(block, 1, out_channels=256, stride=1)
        
        self.maxpool2 = nn.AdaptiveMaxPool2d(output_size = (1, 30))
        
        #self.linear = nn.Linear(params['feature_layers'], params['hidden_size'])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.maxpool2(x)
        #print(x.shape)
        #x = self.layer4(x)
        
        #x = self.linear(x)
        
        return x

    def _make_layer(self, block, blocks, out_channels, stride):
        identify_downsample = None
        layers = []

        # if stride != 1 or self.in_channels != out_channels * 4:
        #     identify_downsample = nn.Sequential(
        #         nn.Conv2d(self.in_channels, out_channels * 1, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(out_channels * 1))

        layers.append(block(self.in_channels, out_channels, identify_downsample, stride))
        self.in_channels = out_channels * 1

        # for i in range(1 - blocks):
        #     layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)  # this unpacks the list so that pytorch knows they will come after each other


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.out_channels = 64
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = [6, 2], stride = [6, 2])
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn = nn.BatchNorm2d(self.out_channels)
        
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(self.out_channels * 2)
 
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3)
        
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn3 = nn.BatchNorm2d(self.out_channels * 2)
        
        self.conv6 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn4 = nn.BatchNorm2d(self.out_channels * 4)
        
        self.conv7 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.bn5 = nn.BatchNorm2d(self.out_channels * 4)
        
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.shape)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu(x)
        #print(x.shape)
        
        x = self.conv4(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # #print(x.shape)
        
        x = self.conv5(x)
        x = self.bn3(x)
        x = self.relu(x)
        #print(x.shape)
        
        # x = self.conv6(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        
        # x = self.conv7(x)
        # x = self.bn5(x)
        # x = self.relu(x)
        
        return x