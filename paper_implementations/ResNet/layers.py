import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# Define a layer that applies a lambda function to its input
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        # Apply the lambda function to the input
        return self.lambd(x)


# Define a basic convolutional block used in the ResNet model
class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, option='A'):
        super(BasicConvBlock, self).__init__()

        # Define the main sequence of layers for the block
        self.features = nn.Sequential(OrderedDict([
            # First convolutional layer followed by batch normalization and ReLU activation
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('act1', nn.ReLU()),
            # Second convolutional layer followed by batch normalization
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(out_channels))
        ]))

        # Initialize the shortcut connection
        self.shortcut = nn.Sequential()

        # Check if the shortcut connection is needed (for downsampling or channel mismatch)
        if stride != 1 or in_channels != out_channels:
            if option == 'A':
                # Option A: Use zero-padding and downsampling
                pad_to_add = (out_channels - in_channels) // 2  # Calculate padding needed to match dimensions
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, pad_to_add, pad_to_add, 0, 0)))
            elif option == 'B':
                # Option B: Use a 1x1 convolution to match dimensions
                self.shortcut = nn.Sequential(OrderedDict([
                    ('s_conv1',
                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)),
                    ('s_bn1', nn.BatchNorm2d(out_channels))
                ]))

    def forward(self, x):
        # Forward pass through the main convolutional layers
        out = self.features(x)
        # Add the shortcut connection
        out += self.shortcut(x)
        # Apply ReLU activation to the result
        out = F.relu(out)
        return out
