import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class FFN(nn.Module):
    """A simple feed-forward network with 3 layers"""
    def __init__(self, num_in, num_out=10):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(num_in, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_out)

    def forward(self, x):
        x = x.view(x.shape[0] * x.shape[1], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class CNN1(nn.Module):

    def __init__(self,
                 input_shape=(32, 32),
                 in_channels=3,
                 ch1=8,
                 ch2=16,
                 ch3=64,
                 lin1=128,
                 drop1=0.1,  # 0.5,
                 drop2=0,  # 0.3,# 0.5,
                 batch_norm=True,
                 dropout=True,):

        super(CNN1, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=ch1, kernel_size=3)
        bn1 = nn.BatchNorm2d(ch1)
        conv2 = nn.Conv2d(in_channels=ch1, out_channels=ch2, kernel_size=3)
        bn2 = nn.BatchNorm2d(ch2)
        conv3 = nn.Conv2d(in_channels=ch2, out_channels=ch3, kernel_size=3)
        bn3 = nn.BatchNorm2d(ch3)

        if batch_norm:
            self.conv_layers = nn.Sequential(OrderedDict([
                ('conv1', conv1),
                ('bn1', bn1),
                ('relu1', nn.ReLU()),
                ('pool1', nn.MaxPool2d(2)),
                ('conv2', conv2),
                ('bn2', bn2),
                ('relu2', nn.ReLU()),
                ('pool2', nn.MaxPool2d(2)),
                ('conv3', conv3),
                ('bn3', bn3),
                ('relu3', nn.ReLU()),
            ])
            )
        else:
            self.conv_layers = nn.Sequential(OrderedDict([
                ('conv1', conv1),
                ('relu1', nn.ReLU()),
                ('pool1', nn.MaxPool2d(2)),
                ('conv2', conv2),
                ('relu2', nn.ReLU()),
                ('pool2', nn.MaxPool2d(2)),
                ('conv3', conv3),
                ('relu3', nn.ReLU()),
            ])
            )

        linear_shape = self.calculate_linear_shape()

        linear1 = nn.Linear(linear_shape, lin1)
        linear2 = nn.Linear(lin1, 10)
        if dropout:
            self.linear_layers = nn.Sequential(OrderedDict([
                ('dropout1', nn.Dropout(0.5)),
                ('linear1', linear1),
                ('relu1', nn.ReLU()),
                ('dropout2', nn.Dropout(0.1)),
                ('linear2', linear2),
            ]))
        else:
            self.linear_layers = nn.Sequential(OrderedDict([
                ('linear1', linear1),
                ('relu1', nn.ReLU()),
                ('linear2', linear2),
            ]))

    def forward(self, input_img):

        x = self.conv_layers(input_img)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x

    def calculate_linear_shape(self):
        test = self.conv_layers(torch.rand(1, self.in_channels, self.input_shape[0], self.input_shape[1]))
        size = test.size()
        m = 1
        for i in size:
            m *= i
        return m
