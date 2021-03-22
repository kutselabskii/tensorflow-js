import segmentation_models as sm
import tensorflowjs as tfjs
import tensorflow as tf

import matplotlib.pyplot as plt

from loader import CustomDataset

sm.set_framework('tf.keras')

model = tf.keras.models.load_model('segmentation_model.h5', compile=False)
testing = CustomDataset(batch_size=1, count=100)

res = model.predict(testing[0][0])
plt.plot(res[0])
plt.show()
