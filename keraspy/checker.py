import tensorflow as tf
from PIL import Image
from PIL.ImageOps import invert
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def recolor(image, mask, texture):
    image = image.convert('HSV')
    texture = texture.resize(image.size).convert('HSV')

    data = image.load()
    texdata = texture.load()

    for i in range(image.size[0]):
        for j in range(image.size[1]):
            if mask[j][i] > 0:
                xy = (i, j)
                pixel = (texdata[i, j][0], texdata[i, j][1], data[i, j][2])
                image.putpixel(xy, pixel)

    image = image.convert('RGB')
    return image


use_checkpoint = True
offset = 20
amount = 6
column_pairs = 1
confidence_threshold = 0.85

texpath = Path(__file__).resolve().parent.joinpath(f"recolor/texture/{1}.jpg")
texture = Image.open(texpath)

if use_checkpoint:
    modelpath = 'checkpoint_model.h5'
else:
    modelpath = 'segmentation_model.h5'

model = tf.keras.models.load_model(modelpath, compile=False)

fig = plt.figure()
for i in range(amount):
    current = i + offset
    number = '0' * (8 - len(str(current))) + str(current)
    path = f"D:/Git/tensorflow-js/keraspy/sofa/{number}.jpg"
    image = Image.open(path).convert('RGB').resize((480, 480))
    data = np.array([np.asarray(image).astype('float32')])
    res = model.predict(data)[0]
    res[res >= confidence_threshold] = 1
    res[res < confidence_threshold] = 0
    img = np.squeeze((res * 255).astype(np.uint8), axis=2)

    recolored = recolor(image, img, texture)

    fig.add_subplot(amount // column_pairs, column_pairs * 3, i * 3 + 1)
    plt.imshow(image)
    fig.add_subplot(amount // column_pairs, column_pairs * 3, i * 3 + 2)
    plt.imshow(img)
    fig.add_subplot(amount // column_pairs, column_pairs * 3, i * 3 + 3)
    plt.imshow(recolored)

fig.tight_layout()
plt.show()
