import random

from pathlib import Path

import imageio
import imgaug as ia
from imgaug import augmenters as iaa

IMAGE_COUNT = 2700
random.seed(0)

path = Path("D:/Git/SofaDataset")
originals = path.joinpath("Originals")
masks = path.joinpath("Masks")

result_originals = path.joinpath("AugmentedOriginals")
result_masks = path.joinpath("AugmentedMasks")

for i in range(IMAGE_COUNT):
    istr = str(i % 900) + ".jpg"
    original = imageio.imread(originals.joinpath(istr))
    mask = imageio.imread(masks.joinpath(istr))

    seq = iaa.Sequential()

    if random.randint(0, 1) == 1:
        seq.add(iaa.GaussianBlur(sigma=random.randint(0, 2)))

    if random.randint(0, 1) == 1:
        seq.add(iaa.SigmoidContrast(gain=random.randint(5, 20), cutoff=random.randint(30, 75) / 100, per_channel=True))

    if random.randint(0, 1) == 1:
        seq.add(iaa.HorizontalFlip())

    if random.randint(0, 1) == 1:
        seq.add(iaa.VerticalFlip())

    if random.randint(0, 1) == 1:
        seq.add(iaa.TranslateX(percent=(random.randrange(-40, 40) / 100)))

    if random.randint(0, 1) == 1:
        seq.add(iaa.TranslateY(percent=(random.randrange(-40, 40) / 100)))

    if random.randint(0, 1) == 1:
        seq.add(iaa.Rotate(random.randrange(-60, 60)))

    results = seq(images=[original, mask])

    istr = str(i) + ".jpg"
    imageio.imsave(result_originals.joinpath(istr), results[0])
    imageio.imsave(result_masks.joinpath(istr), results[1])
