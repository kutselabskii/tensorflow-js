from pathlib import Path

import tensorflowjs as tfjs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model

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
IMG_COUNT = 3600
TRAIN_PERCENTAGE = 0.92
IMG_SIZE = (1024, 512)

train_amount = round(IMG_COUNT * TRAIN_PERCENTAGE)
val_amount = round(IMG_COUNT * (1 - TRAIN_PERCENTAGE))

train_generator = CustomDataset(batch_size=BATCH_SIZE, count=train_amount, img_size=IMG_SIZE)
val_generator = CustomDataset(batch_size=BATCH_SIZE, count=val_amount, img_size=IMG_SIZE, offset=train_amount)

keras.backend.clear_session()
model = models.get_model(IMG_SIZE)

# optimizer = tf.keras.optimizers.SGD(momentum=0.9, lr=0.045)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.MeanIoU(num_classes=2)], run_eagerly=True)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    # monitor='val_mean_io_u',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# print(model.summary())
plot_model(model, to_file=str(Path(__file__).resolve().parent.joinpath('model_plot.png')), show_shapes=True, show_layer_names=True)

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
