import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, block_type, num_blocks):
        super(ResNet, self).__init__()

        # Initial number of input channels
        self.in_channels = 16

        # Initial convolutional layer with Batch Normalization
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16)

        # Define three layers with varying number of blocks
        self.block1 = self.__build_layer(block_type, 16, num_blocks[0],
                                         starting_stride=1)
        self.block2 = self.__build_layer(block_type, 32, num_blocks[1],
                                         starting_stride=2)
        self.block3 = self.__build_layer(block_type, 64, num_blocks[2],
                                         starting_stride=2)

        # Adaptive pooling to ensure the output size is 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Final fully connected layer for classification
        self.linear = nn.Linear(64, 10)

    # Helper function to create layers
    def __build_layer(self, block_type, out_channels, num_blocks, starting_stride):
        # Create a list of strides to handle downsampling in the first block
        strides_list_for_current_block = [starting_stride] + [1] * (num_blocks - 1)
        layers = []

        # Build each block in the layer
        for stride in strides_list_for_current_block:
            layers.append(block_type(self.in_channels, out_channels, stride))
            self.in_channels = out_channels  # Update input channels for next block

        # Return a sequential container of layers
        return nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the initial convolutional layer and activation
        out = F.relu(self.bn0(self.conv0(x)))

        # Forward pass through each set of residual blocks
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        # Apply global average pooling
        out = self.avgpool(out)

        # Flatten and pass through the fully connected layer
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out
