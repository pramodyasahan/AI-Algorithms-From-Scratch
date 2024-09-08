from layers import *
from utils import softmax


class AlexNet:
    """
    AlexNet Model Class
    """

    def __init__(self, num_classes=10):
        self.conv1 = ConvLayer(3, 96, kernel_size=11, stride=4, padding=2)
        self.relu1 = ReLULayer()
        self.pool1 = PoolingLayer(pool_size=3, stride=2)

        self.conv2 = ConvLayer(96, 256, kernel_size=5, stride=1, padding=2)
        self.relu2 = ReLULayer()
        self.pool2 = PoolingLayer(pool_size=3, stride=2)

        self.conv3 = ConvLayer(256, 384, kernel_size=3, stride=1, padding=1)
        self.relu3 = ReLULayer()

        self.conv4 = ConvLayer(384, 384, kernel_size=3, stride=1, padding=1)
        self.relu4 = ReLULayer()

        self.conv5 = ConvLayer(384, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = ReLULayer()
        self.pool3 = PoolingLayer(pool_size=3, stride=2)

        self.fc1 = FullyConnectedLayer(256 * 6 * 6, 4096)
        self.relu6 = ReLULayer()
        self.dropout1 = DropoutLayer()

        self.fc2 = FullyConnectedLayer(4096, 4096)
        self.relu7 = ReLULayer()
        self.dropout2 = DropoutLayer()
        self.fc3 = FullyConnectedLayer(4096, num_classes)

    def apply_layers(self, x, layers):
        for layer in layers:
            x = layer.forward(x)
        return x

    def forward(self, input):
        # Sequentially apply layers using the helper function
        x = self.apply_layers(input, [self.conv1, self.relu1, self.pool1])
        x = self.apply_layers(x, [self.conv2, self.relu2, self.pool2])
        x = self.apply_layers(x, [self.conv3, self.relu3])
        x = self.apply_layers(x, [self.conv4, self.relu4])
        x = self.apply_layers(x, [self.conv5, self.relu5, self.pool3])

        x = x.flatten()  # Flatten for fully connected layers

        x = self.apply_layers(x, [self.fc1, self.relu6, self.dropout1])
        x = self.apply_layers(x, [self.fc2, self.relu7, self.dropout2])
        output = softmax(self.fc3.forward(x))

        return output
