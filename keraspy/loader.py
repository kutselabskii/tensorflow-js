import numpy as np
from PIL import Image
from pathlib import Path


def sync_shuffle(a, b):
    p = np.random.permutation(len(a))
    return np.array(a)[p], np.array(b)[p]


def load_image(gesture, number, height, width, datadir, format):
    path = str(datadir.joinpath(f"{gesture}/{number}{format}"))
    image = Image.open(path).resize((height, width))
    data = np.asarray(image) / 255
    return data, gesture - 1


def load_dataset(gestures_count, images_count, height, width, folder, format):
    X, Y = [], []
    datadir = Path(__file__).resolve().parent.joinpath(folder)

    for i in range(1, gestures_count + 1):
        print(f"{i}/{gestures_count + 1}")
        for j in range(1, images_count + 1):
            try:
                x, y = load_image(i, j, height, width, datadir, format)
                X.append(x)
                Y.append(y)
            except Exception:
                print(f'Skipped {i}/{j}')
                continue
    X, Y = sync_shuffle(X, Y)

    divider = round(len(X) / 100 * 90)
    x_train = X[:divider]
    y_train = Y[:divider]
    x_test = X[divider:]
    y_test = Y[divider:]

    return x_train, y_train, x_test, y_test
