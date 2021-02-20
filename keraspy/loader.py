import numpy as np
from PIL import Image
from pathlib import Path


path = "D:/Git/NDDS Generated/TestCapturer"


def sync_shuffle(a, b):
    p = np.random.permutation(len(a))
    return np.array(a)[p], np.array(b)[p]


def load_image(number, height, width, subtype, datadir):
    strnumber = '0' * (6 - len(str(number))) + str(number)
    path = str(datadir.joinpath(f"{strnumber}{subtype}.png"))
    image = Image.open(path).resize((height, width)).convert('RGB')
    data = np.asarray(image).astype('float32')
    return data


def load_dataset(count, height, width):
    X, Y = [], []
    datadir = Path(path)

    for i in range(count):
        print(f"{i}/{count}")
        x = load_image(i, height, width, '', datadir)
        y = load_image(i, height, width, '.cs', datadir)
        X.append(x)
        Y.append(y)
    X, Y = sync_shuffle(X, Y)

    divider = round(len(X) / 100 * 90)
    x_train = X[:divider]
    y_train = Y[:divider]
    x_test = X[divider:]
    y_test = Y[divider:]

    return x_train, y_train, x_test, y_test
