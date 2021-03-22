import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from loader import CustomDataset

model = tf.keras.models.load_model('segmentation_model.h5', compile=False)
testing = CustomDataset(batch_size=1, count=100)

res = model.predict(testing[0][0])[0]
img = Image.fromarray(np.squeeze((res * 255).astype(np.uint8), axis=2))
img.show()
