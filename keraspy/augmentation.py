import random

from pathlib import Path

import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

IMAGE_COUNT = 8100

ia.seed(1)

path = Path("D:/Git/SofaDataset")
originals = path.joinpath("Originals")
masks = path.joinpath("Masks")

result_originals = path.joinpath("AugmentedOriginals")
result_masks = path.joinpath("AugmentedMasks")

for i in range(IMAGE_COUNT):
    istr = str(i % 900) + ".jpg"
    original = imageio.imread(originals.joinpath(istr))
    mask = imageio.imread(masks.joinpath(istr))
    mask = SegmentationMapsOnImage(mask, shape=mask.shape)

    seq = iaa.SomeOf((0, None), random_order=True)

    seq.add(iaa.Add((-40, 40), per_channel=0.5))
    seq.add(iaa.GaussianBlur(sigma=(0, 2)))
    seq.add(iaa.SigmoidContrast(gain=(5, 20), cutoff=(0.3, 0.75), per_channel=True))
    seq.add(iaa.HorizontalFlip())
    seq.add(iaa.VerticalFlip())
    seq.add(iaa.TranslateX(percent=(-0.7, 0.7), mode='edge'))
    seq.add(iaa.TranslateY(percent=(-0.7, 0.7), mode='edge'))
    seq.add(iaa.Rotate(random.randrange(-60, 60), mode='edge'))
    seq.add(iaa.ScaleX((0.5, 1.5), mode='edge'))
    seq.add(iaa.ScaleY((0.5, 1.5), mode='edge'))
    seq.add(iaa.imgcorruptlike.DefocusBlur(severity=1))

    results_o, results_m = seq(image=original, segmentation_maps=mask)

    istr = str(i) + ".jpg"
    imageio.imsave(result_originals.joinpath(istr), results_o)
    imageio.imsave(result_masks.joinpath(istr), results_m.arr)
