import torch.nn as nn
from torch.nn import Sequential
import torch
import numpy as np

class Flatten(nn.Module):
    """
    Implement a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return x.view(x.size()[0],-1)




def all_cnn_module():
    """
    Create a nn.Sequential model containing all of the layers of the All-CNN-C as specified in the paper.
    https://arxiv.org/pdf/1412.6806.pdf
    Use a AvgPool2d to pool and then your Flatten layer as your final layers.
    You should have a total of exactly 23 layers of types:
    - nn.Dropout
    - nn.Conv2d
    - nn.ReLU
    - nn.AvgPool2d
    - Flatten
    :return: a nn.Sequential model
    """
    Layer_23 = nn.Sequential(nn.Dropout(p=0.2),
                             nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, padding=1, stride=1),
                             nn.ReLU(),
                             nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, stride=1),
                             nn.ReLU(),
                             nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, stride=2),
                             nn.ReLU(),
                             nn.Dropout(p=0.5),
                             nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, padding=1, stride=1),
                             nn.ReLU(),
                             nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, stride=1),
                             nn.ReLU(),
                             nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, stride=2),
                             nn.ReLU(),
                             nn.Dropout(p=0.5),
                             nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1),
                             nn.ReLU(),
                             nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1),
                             nn.ReLU(),
                             nn.Conv2d(in_channels=192, out_channels=10, kernel_size=1, stride=1),
                             nn.ReLU(),
                             nn.AvgPool2d(kernel_size=6),
                             Flatten())
    return Layer_23