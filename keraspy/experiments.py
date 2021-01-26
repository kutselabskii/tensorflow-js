import colorsys
from PIL import Image
from pathlib import Path
import numpy as np


path = Path(__file__).resolve().parent.joinpath('sofa.jpg')
texpath = Path(__file__).resolve().parent.joinpath('texture.jpg')
savepath = Path(__file__).resolve().parent.joinpath('new.jpg')

image = Image.open(path).convert('HSV')
texture = Image.open(texpath).resize(image.size).convert('HSV')

data = np.asarray(image.getdata())
texdata = np.asarray(texture.getdata())

print(data.shape)

for i in range(len(data)):
    data[i] = (texdata[i][0], texdata[i][1], data[i][2])

repainted = Image.fromarray(data, 'HSV').convert('RGB').resize(image.size)
# for i in range(100):
#     print(repainted.getdata()[i])
# quit()
repainted.save(savepath)
