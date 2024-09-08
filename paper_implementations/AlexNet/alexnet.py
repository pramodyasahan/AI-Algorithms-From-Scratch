import numpy as np


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
