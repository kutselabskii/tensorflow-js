import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


use_checkpoint = True
offset = 1
amount = 12
column_pairs = 3

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
    res[res >= 0.75] = 1
    res[res < 0.75] = 0
    img = np.squeeze((res * 255).astype(np.uint8), axis=2)

    fig.add_subplot(amount // column_pairs, column_pairs * 2, i * 2 + 1)
    plt.imshow(image)
    fig.add_subplot(amount // column_pairs, column_pairs * 2, i * 2 + 2)
    plt.imshow(img)

fig.tight_layout()
plt.show()
