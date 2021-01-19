import numpy as np
from PIL import Image
from pathlib import Path
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


def sync_shuffle(a, b):
    p = np.random.permutation(len(a))
    return np.array(a)[p], np.array(b)[p]


def load_image(gesture, number, height, width, datadir, format):
    path = str(datadir.joinpath(f"{gesture}/{number}{format}"))
    image = Image.open(path).resize((height, width))
    data = np.asarray(image) / 255
    return data, gesture - 1


def load_hand_image(gesture, number, height, width, datadir, format):
    path = str(datadir.joinpath(f"{gesture}/{number}{format}"))
    image = Image.open(path).resize((height, width))
    data = np.asarray(image)
    hd = hands.process(data)
    result = None
    if hd is not None:
        marks = hd.multi_hand_landmarks[0]
        result = np.array([[mark.x, mark.y] for mark in marks.landmark])
    return result, gesture - 1


def load_dataset(gestures_count, images_count, height, width, folder, format):
    X, Y = [], []
    datadir = Path(__file__).resolve().parent.joinpath(folder)

    for i in range(1, gestures_count + 1):
        print(f"{i}/{gestures_count + 1}")
        for j in range(1, images_count + 1):
            try:
                x, y = load_hand_image(i, j, height, width, datadir, format)
                if x is None:
                    continue
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
