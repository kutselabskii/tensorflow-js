import colorsys
from PIL import Image
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


path = Path(__file__).resolve().parent.joinpath('sofa.jpg')
texpath = Path(__file__).resolve().parent.joinpath('texture.jpg')
savepath = Path(__file__).resolve().parent.joinpath('new.jpg')

image = Image.open(path).convert('HSV')
texture = Image.open(texpath).resize(image.size).convert('HSV')

data = image.load()
texdata = texture.load()

for i in range(image.size[0]):
    for j in range(image.size[1]):
        xy = (i, j)
        pixel = (texdata[i, j][0], texdata[i, j][1], data[i, j][2])
        image.putpixel(xy, pixel)

image.convert('RGB').save(savepath)

original_image = mpimg.imread(path)
texture_image = mpimg.imread(texpath)
resulting_image = mpimg.imread(savepath)

fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
imgplot = plt.imshow(original_image)
ax.set_title('Original')
ax = fig.add_subplot(1, 3, 2)
imgplot = plt.imshow(texture_image)
ax.set_title('Texture')
ax = fig.add_subplot(1, 3, 3)
imgplot = plt.imshow(resulting_image)
ax.set_title('Result')
plt.show()
