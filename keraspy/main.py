from pathlib import Path

import tensorflowjs as tfjs
import tensorflow as tf
from tensorflow import keras

from loader import CustomDataset
import models


modelpath = str(Path(__file__).resolve().parent.joinpath('model'))
save_path = 'fast_scnn.h5'
checkpoint_path = 'checkpoint_fast_scnn.h5'

BACKBONE = 'resnet34'
CLASSES = ['sofa']
LR = 0.001
EPOCHS = 100
BATCH_SIZE = 4
IMG_COUNT = 900
TRAIN_PERCENTAGE = 0.92
IMG_SIZE = (2048, 1024)

train_amount = round(IMG_COUNT * TRAIN_PERCENTAGE)
val_amount = round(IMG_COUNT * (1 - TRAIN_PERCENTAGE))

train_generator = CustomDataset(batch_size=BATCH_SIZE, count=train_amount, img_size=IMG_SIZE)
val_generator = CustomDataset(batch_size=BATCH_SIZE, count=val_amount, img_size=IMG_SIZE, offset=train_amount)

keras.backend.clear_session()
model = models.get_model(IMG_SIZE)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_mean_io_u',
    mode='max',
    save_best_only=True)

print(model.summary())

model.fit(
   train_generator,
   validation_data=val_generator,
   epochs=EPOCHS,
   callbacks=[model_checkpoint_callback]
)

model.save(save_path)
tfjs.converters.save_keras_model(model, modelpath)

test_x, test_y = val_generator[0]
preds = model.evaluate(test_x, test_y)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
