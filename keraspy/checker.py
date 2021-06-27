import tensorflow as tf
from PIL import Image
from PIL.ImageOps import invert
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models import ResizeLayer
from datetime import datetime

SIZE_X = 512
SIZE_Y = 512
today = datetime.today().strftime('%Y-%m-%d')
suffix = "linknet_straight_preprocess"
# suffix = "simple_unet_straight"
# suffix = ""


def recolor(image, mask, texture):
    image = image.convert('HSV')
    texture = texture.resize(image.size).convert('HSV')

    data = image.load()
    texdata = texture.load()

    for i in range(image.size[0]):
        for j in range(image.size[1]):
            if mask[j][i] < 0.3:
                xy = (i, j)
                pixel = (texdata[i, j][0], texdata[i, j][1], data[i, j][2])
                image.putpixel(xy, pixel)

    image = image.convert('RGB')
    return image


use_checkpoint = True
offset = 25
amount = 4
confidence_threshold = 0.5

texpath = Path(__file__).resolve().parent.joinpath(f"recolor/texture/{1}.jpg")
texture = Image.open(texpath)

if use_checkpoint:
    # modelpath = f"unused_models/{today}/checkpoint_fast_scnn_binary{suffix}.h5"
    modelpath = f"unused_models/2021-06-22/checkpoint_fast_scnn_binary_iou.h5"
else:
    modelpath = f'unused_models/{today}/fast_scnn_binary{suffix}.h5'
print(modelpath)

model = tf.keras.models.load_model(modelpath, compile=False, custom_objects={"ResizeLayer": ResizeLayer})

fig = plt.figure()
for i in range(amount):
    current = i + offset
    number = '0' * (8 - len(str(current))) + str(current)
    path = f"D:/Git/tensorflow-js/keraspy/sofa/{number}.jpg"
    image = Image.open(path).convert('RGB').resize((SIZE_X, SIZE_Y))
    data = np.array([np.asarray(image).astype('float32')]) / 255
    # data = np.array([np.asarray(image).astype('float32')])
    # data = np.asarray(image)
    # data = tf.keras.applications.mobilenet.preprocess_input(data)
    # data = np.array([data])
    
    res = model.predict(data)[0]
    res[res >= confidence_threshold] = 1
    res[res < confidence_threshold] = 0

    # img = np.squeeze((res * 255).astype(np.uint8), axis=2)
    img = (res * 255).astype(np.uint8)

    recolored = recolor(image, img, texture)

    # f = lambda x: x[1]
    # img = np.apply_along_axis(f, 0, img)

    fig.add_subplot(amount, 1, i + 1)
    plt.imshow(np.hstack((image, recolored)))
    # fig.add_subplot(amount // column_pairs, column_pairs * 3, i * 3 + 2)
    # plt.imshow(img)

fig.tight_layout()
plt.show()
