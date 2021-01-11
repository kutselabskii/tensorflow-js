import numpy as np
from PIL import Image
from pathlib import Path

datadir = Path(__file__).resolve().parent.joinpath("dataset")

gestures_count = 11
images_count = 120


def sync_shuffle(a, b):
    p = np.random.permutation(len(a))
    return np.array(a)[p], np.array(b)[p]


def load_image(gesture, number):
    path = str(datadir.joinpath(f"{gesture}/{number}.png"))
    image = Image.open(path).resize((320, 240))
    data = np.asarray(image) / 255
    return data, gesture - 1


def load_dataset():
    X, Y = [], []

    for i in range(1, gestures_count + 1):
        for j in range(1, images_count + 1):
            x, y = load_image(i, j)
            X.append(x)
            Y.append(y)
    X, Y = sync_shuffle(X, Y)

    divider = round(len(X) / 100 * 90)
    x_train = X[:divider]
    y_train = Y[:divider]
    x_test = X[divider:]
    y_test = Y[divider:]

    return x_train, y_train, x_test, y_test
