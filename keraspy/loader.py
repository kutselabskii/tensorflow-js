import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import to_categorical


class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, batch_size, count, img_size, offset=0, *args, **kwargs):
        self.batch_size = batch_size
        self.count = count
        self.offset = offset
        self.cache = {}
        # self.img_size = (img_size[1], img_size[0])
        self.img_size = (img_size[1], img_size[0])

    def __len__(self):
        return self.count // self.batch_size

    def __getitem__(self, index):
        starter = self.offset + self.batch_size * index

        X = []
        y = []

        for i in range(self.batch_size):
            number = starter + i

            if number < 900:
                X.append(self.load_image(number, 'Originals'))
                y.append(self.load_image(number, 'Masks'))
            else:
                X.append(self.load_image(number - 900, 'AugmentedOriginals'))
                y.append(self.load_image(number - 900, 'AugmentedMasks'))

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        pass

    def load_image(self, number, folder):
        datadir = Path("D:/Git/SofaDataset")
        path = str(datadir.joinpath(folder).joinpath(f"{number}.jpg"))
        if path in self.cache.keys():
            return self.cache[path]
        image = Image.open(path).convert('RGB').resize(self.img_size)
        if folder != 'Originals' and folder != 'AugmentedOriginals':
            image = ImageOps.grayscale(image)
            data = np.asarray(image).astype('float32') / 255
            # f = lambda x: to_categorical(x, num_classes=2)
            # data = np.apply_along_axis(f, 1, data)
        else:
            data = np.asarray(image).astype('float32') / 255
        self.cache[path] = data
        return data
