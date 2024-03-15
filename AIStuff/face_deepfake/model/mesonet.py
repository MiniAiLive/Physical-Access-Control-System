import torch
from torch import nn


class Meso4(nn.Module):
    """ define the mesonet 4 """
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
        #flatten: x = x.view(x.size(0), -1)
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16*8*8, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        """ forward the model with the input image """
        x = self.conv1(x) #(8, 256, 256)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x) #(8, 128, 128)
        x = self.conv2(x) #(8, 128, 128)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x) #(8, 64, 64)
        x = self.conv3(x) #(16, 64, 64)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling1(x) #(16, 32, 32)
        x = self.conv4(x) #(16, 32, 32)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling2(x) #(16, 8, 8)
        x = x.view(x.size(0), -1) #(Batch, 16*8*8)
        x = self.dropout(x)
        x = self.fc1(x) #(Batch, 16)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MesoInception4(nn.Module):
    """ define the MesoInception4 model """
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        #InceptionLayer1
        self.inception1_conv1 = nn.Conv2d(3, 1, 1, padding=0, bias=False)
        self.inception1_conv2_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.inception1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.inception1_conv3_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.inception1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.inception1_conv4_1 = nn.Conv2d(3, 2, 1, padding=0, bias=False)
        self.inception1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.inception1_bn = nn.BatchNorm2d(11)
        # InceptionLayer2
        self.inception2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.inception2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.inception2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.inception2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.inception2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.inception2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.inception2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.inception2_bn = nn.BatchNorm2d(12)
        # Normal Layer
        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16*8*8, 16)
        self.fc2 = nn.Linear(16, num_classes)

    #InceptionLayer
    def InceptionLayer1(self, x):
        """ define the InceptionLayer1 module """
        x1 = self.inception1_conv1(x)
        x2 = self.inception1_conv2_1(x)
        x2 = self.inception1_conv2_2(x2)
        x3 = self.inception1_conv3_1(x)
        x3 = self.inception1_conv3_2(x3)
        x4 = self.inception1_conv4_1(x)
        x4 = self.inception1_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.inception1_bn(y)
        y = self.maxpooling1(y)

        return y

    def InceptionLayer2(self, x):
        """ define the InceptionLayer2 module """
        x1 = self.inception2_conv1(x)
        x2 = self.inception2_conv2_1(x)
        x2 = self.inception2_conv2_2(x2)
        x3 = self.inception2_conv3_1(x)
        x3 = self.inception2_conv3_2(x3)
        x4 = self.inception2_conv4_1(x)
        x4 = self.inception2_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.inception2_bn(y)
        y = self.maxpooling1(y)

        return y

    def forward(self, x):
        """ forward the model with the input image """
        x = self.InceptionLayer1(x) #(Batch, 11, 128, 128)
        x = self.InceptionLayer2(x) #(Batch, 12, 64, 64)

        x = self.conv1(x) #(Batch, 16, 64 ,64)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x) #(Batch, 16, 32, 32)

        x = self.conv2(x) #(Batch, 16, 32, 32)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling2(x) #(Batch, 16, 8, 8)

        x = x.view(x.size(0), -1) #(Batch, 16*8*8)
        x = self.dropout(x)
        x = self.fc1(x) #(Batch, 16)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
