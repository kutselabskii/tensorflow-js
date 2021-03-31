import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from loader import CustomDataset

check_real = True
use_checkpoint = True
real_number = 7

if use_checkpoint:
    modelpath = 'checkpoint_model.h5'
else:
    modelpath = 'segmentation_model.h5'

model = tf.keras.models.load_model(modelpath, compile=False)

if check_real:
    path = f"D:/Git/tensorflow-js/keraspy/sofa/0000000{real_number}.jpg"
    image = Image.open(path).convert('RGB').resize((480, 480))
    data = np.array([np.asarray(image).astype('float32')])
else:
    data = CustomDataset(batch_size=1, count=100)[5][0]

res = model.predict(data)[0]
img = Image.fromarray(np.squeeze((res * 255).astype(np.uint8), axis=2))
img.show()
