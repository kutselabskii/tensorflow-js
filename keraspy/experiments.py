import colorsys
from PIL import Image
from pathlib import Path
import numpy as np


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
