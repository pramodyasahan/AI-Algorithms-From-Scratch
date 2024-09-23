import numpy as np


# Helper functions
def conv2d(input, kernel, stride=1, padding=0):
    batch_size, in_depth, in_height, in_width = input.shape
    out_channels, kernel_height, kernel_width = kernel.shape[0], kernel.shape[2], kernel.shape[3]

    # Apply padding
    input_padded = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    # Calculate output dimensions
    out_height = (in_height + 2 * padding - kernel_height) // stride + 1
    out_width = (in_width + 2 * padding - kernel_width) // stride + 1
    output = np.zeros((batch_size, out_channels, out_height, out_width))

    # Perform the convolution
    for n in range(batch_size):
        for i in range(out_height):
            for j in range(out_width):
                for c in range(out_channels):
                    region = input_padded[n, :, i * stride:i * stride + kernel_height,
                             j * stride:j * stride + kernel_width]
                    output[n, c, i, j] = np.sum(region * kernel[c])

    return output


def max_pooling(input, pool_size, stride):
    batch_size, in_depth, in_height, in_width = input.shape
    out_height = (in_height - pool_size) // stride + 1
    out_width = (in_width - pool_size) // stride + 1
    output = np.zeros((batch_size, in_depth, out_height, out_width))

    for n in range(batch_size):
        for i in range(out_height):
            for j in range(out_width):
                region = input[n, :, i * stride:i * stride + pool_size, j * stride:j * stride + pool_size]
                output[n, :, i, j] = np.max(region, axis=(1, 2))

    return output


def relu(x):
    return np.maximum(0, x)


def fully_connected(input, weights, bias):
    return np.dot(input, weights.T) + bias.T


# Gradient helpers
def conv2d_backward(d_output, input, kernel, stride=1, padding=0):
    batch_size, in_depth, in_height, in_width = input.shape
    d_input = np.zeros_like(input)
    d_kernels = np.zeros_like(kernel)

    input_padded = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    d_input_padded = np.pad(d_input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    out_channels, out_height, out_width = d_output.shape[1:]

    for n in range(batch_size):
        for i in range(out_height):
            for j in range(out_width):
                for c in range(out_channels):
                    region = input_padded[n, :, i * stride:i * stride + kernel.shape[2],
                             j * stride:j * stride + kernel.shape[3]]
                    d_kernels[c] += region * d_output[n, c, i, j]
                    d_input_padded[n, :, i * stride:i * stride + kernel.shape[2],
                    j * stride:j * stride + kernel.shape[3]] += kernel[c] * d_output[n, c, i, j]

    if padding > 0:
        d_input = d_input_padded[:, :, padding:-padding, padding:-padding]
    else:
        d_input = d_input_padded

    return d_input, d_kernels


def max_pooling_backward(d_output, input, pool_size, stride):
    batch_size, in_depth, in_height, in_width = input.shape
    d_input = np.zeros_like(input)
    out_height, out_width = d_output.shape[2:]

    for n in range(batch_size):
        for i in range(out_height):
            for j in range(out_width):
                for c in range(in_depth):
                    region = input[n, c, i * stride:i * stride + pool_size, j * stride:j * stride + pool_size]
                    max_index = np.unravel_index(np.argmax(region, axis=None), region.shape)
                    d_input[n, c, i * stride + max_index[0], j * stride + max_index[1]] = d_output[n, c, i, j]

    return d_input


def relu_backward(d_output, input):
    return d_output * (input > 0)


# Layer classes
class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # He initialization
        self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2. / in_channels)

    def forward(self, input):
        self.input = input
        return conv2d(input, self.kernels, self.stride, self.padding)

    def backward(self, d_output, learning_rate):
        d_input, d_kernels = conv2d_backward(d_output, self.input, self.kernels, self.stride, self.padding)
        self.kernels -= learning_rate * d_kernels
        return d_input


class PoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        return max_pooling(input, self.pool_size, self.stride)

    def backward(self, d_output):
        return max_pooling_backward(d_output, self.input, self.pool_size, self.stride)


class ReLULayer:
    def forward(self, input):
        self.input = input
        return relu(input)

    def backward(self, d_output):
        return relu_backward(d_output, self.input)


class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        # He initialization
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        return fully_connected(input, self.weights, self.bias)

    def backward(self, d_output, learning_rate):
        d_weights = np.dot(d_output.T, self.input)
        d_bias = np.sum(d_output, axis=0, keepdims=True).T

        d_input = np.dot(d_output, self.weights)

        # Update weights and biases
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias

        return d_input


class FlattenLayer:
    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, d_output):
        return d_output.reshape(self.input_shape)


class SimpleCNN:

    def __init__(self):
        self.layers = [
            # First Convolutional Block
            ConvLayer(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            ReLULayer(),
            PoolingLayer(pool_size=2, stride=2),  # Reduces 32x32 -> 16x16

            # Second Convolutional Block
            ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            ReLULayer(),
            PoolingLayer(pool_size=2, stride=2),  # Reduces 16x16 -> 8x8

            # Third Convolutional Block
            ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            ReLULayer(),
            PoolingLayer(pool_size=2, stride=2),  # Reduces 8x8 -> 4x4

            # Flatten Layer
            FlattenLayer(),

            # Fully Connected Layers
            FullyConnectedLayer(input_size=128 * 4 * 4, output_size=256),  # First FC Layer
            ReLULayer(),
            FullyConnectedLayer(input_size=256, output_size=128),  # Second FC Layer
            ReLULayer(),
            FullyConnectedLayer(input_size=128, output_size=10)  # Output Layer for 10 classes
        ]

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, d_output, learning_rate):
        for layer in reversed(self.layers):
            if isinstance(layer, (ConvLayer, FullyConnectedLayer)):
                d_output = layer.backward(d_output, learning_rate)
            else:
                d_output = layer.backward(d_output)
