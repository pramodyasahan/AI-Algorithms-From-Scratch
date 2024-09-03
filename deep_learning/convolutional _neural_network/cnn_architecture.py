import numpy as np


class ConvolutionalLayer:
    def __init__(self, num_filters, filter_size):
        self.last_input = None
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / 9

    def apply_filters(self, image):
        h, w = image.shape

        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                image_region = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield image_region, i, j

    def forward(self, x):
        self.last_input = x
        h, w = x.shape
        output = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))

        for image_region, i, j in self.apply_filters(x):
            output[i, j] = np.sum(image_region * self.filters, axis=(1, 2))
        return output
