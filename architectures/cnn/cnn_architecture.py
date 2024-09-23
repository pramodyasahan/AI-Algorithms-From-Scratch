import numpy as np


# Forward and backward helper functions
def conv2d(input, kernel, stride=1, padding=0):
    in_depth, in_width, in_height = input.shape
    out_channels, kernel_height, kernel_width = kernel.shape[0], kernel.shape[2], kernel.shape[3]

    input_padded = np.pad(input, ((0, 0), (padding, padding), (padding, padding)), mode='constant')

    out_height = (in_height + 2 * padding - kernel_height) // stride + 1
    out_width = (in_width + 2 * padding - kernel_width) // stride + 1
    output = np.zeros((out_channels, out_height, out_width))

    for i in range(out_height):
        for j in range(out_width):
            for c in range(out_channels):
                region = input_padded[:, i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width]
                output[c, i, j] = np.sum(region * kernel[c])

    return output


def max_pooling(input, pool_size, stride):
    in_depth, in_height, in_width = input.shape
    out_height = (in_height - pool_size) // stride + 1
    out_width = (in_width - pool_size) // stride + 1
    output = np.zeros((in_depth, out_height, out_width))

    for i in range(out_height):
        for j in range(out_width):
            region = input[:, i * stride:i * stride + pool_size, j * stride:j * stride + pool_size]
            output[:, i, j] = np.max(region, axis=(1, 2))

    return output


def relu(x):
    return np.maximum(0, x)


def fully_connected(input, weights, bias):
    return np.dot(weights, input) + bias


# Gradient helpers
def conv2d_backward(d_output, input, kernel, stride=1, padding=0):
    d_input = np.zeros_like(input)
    d_kernels = np.zeros_like(kernel)
    input_padded = np.pad(input, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    d_input_padded = np.pad(d_input, ((0, 0), (padding, padding), (padding, padding)), mode='constant')

    out_channels, out_height, out_width = d_output.shape

    for i in range(out_height):
        for j in range(out_width):
            for c in range(out_channels):
                region = input_padded[:, i * stride:i * stride + kernel.shape[2],
                         j * stride:j * stride + kernel.shape[3]]
                d_kernels[c] += region * d_output[c, i, j]
                d_input_padded[:, i * stride:i * stride + kernel.shape[2], j * stride:j * stride + kernel.shape[3]] += \
                    kernel[c] * d_output[c, i, j]

    if padding > 0:
        d_input = d_input_padded[:, padding:-padding, padding:-padding]
    else:
        d_input = d_input_padded

    return d_input, d_kernels


def max_pooling_backward(d_output, input, pool_size, stride):
    d_input = np.zeros_like(input)
    in_depth, in_height, in_width = input.shape
    out_height, out_width = d_output.shape[1], d_output.shape[2]

    for i in range(out_height):
        for j in range(out_width):
            for c in range(in_depth):
                region = input[c, i * stride:i * stride + pool_size, j * stride:j * stride + pool_size]
                max_index = np.unravel_index(np.argmax(region, axis=None), region.shape)
                d_input[c, i * stride + max_index[0], j * stride + max_index[1]] = d_output[c, i, j]

    return d_input


def relu_backward(d_output, input):
    return d_output * (input > 0)


class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01

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
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        return fully_connected(input, self.weights, self.bias)

    def backward(self, d_output, learning_rate):
        d_weights = np.dot(d_output, self.input.T)
        d_bias = d_output

        d_input = np.dot(self.weights.T, d_output)

        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias

        return d_input


class DropoutLayer:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, input, training=True):
        if training:
            self.mask = (np.random.rand(*input.shape) > self.dropout_rate).astype(float)
            output = input * self.mask / (1 - self.dropout_rate)
        else:
            output = input
        return output

    def backward(self, d_output):
        return d_output * self.mask
