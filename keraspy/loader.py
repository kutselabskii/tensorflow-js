import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import tensorflow as tf
import numpy as np


class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, batch_size, count, offset=0, *args, **kwargs):
        self.batch_size = batch_size
        self.count = count
        self.offset = offset
        self.path = "D:/Git/NDDS Generated/TestCapturer"
        self.cache = {}

    def __len__(self):
        return self.count // self.batch_size

    def __getitem__(self, index):
        starter = self.batch_size * index

        X = []
        y = []

        for i in range(self.batch_size):
            number = starter + i
            X.append(self.load_image(number, ''))
            y.append(self.load_image(number, '.cs'))

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        pass

    def load_image(self, number, subtype):
        datadir = Path(self.path)
        strnumber = '0' * (6 - len(str(number))) + str(number)
        path = str(datadir.joinpath(f"{strnumber}{subtype}.png"))
        if path in self.cache.keys():
            return self.cache[path]
        image = Image.open(path).convert('RGB')
        if subtype != '':
            image = ImageOps.grayscale(image)
            data = np.asarray(image).astype('float32') / 255
        else:
            data = np.asarray(image).astype('float32')
        self.cache[path] = data
        return data
