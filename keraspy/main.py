from pathlib import Path

import tensorflowjs as tfjs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model

import numpy as np

from loader import CustomDataset
import models

from datetime import datetime

import segmentation_models as sm
sm.set_framework('tf.keras')


suffix = "linknet_straight"

today = datetime.today().strftime('%Y-%m-%d')
modelpath = str(Path(__file__).resolve().parent.joinpath('model'))
save_path = f'unused_models/{today}/fast_scnn_binary{suffix}.h5'
checkpoint_path = f'unused_models/{today}/checkpoint_fast_scnn_binary{suffix}.h5'
path = f'unused_models/{today}/'

BACKBONE = 'resnet34'
CLASSES = ['sofa']
LR = 0.1
EPOCHS = 400
BATCH_SIZE = 64
IMG_COUNT = 3600
TRAIN_PERCENTAGE = 0.92
IMG_SIZE = (128, 128)

train_amount = round(IMG_COUNT * TRAIN_PERCENTAGE)
val_amount = round(IMG_COUNT * (1 - TRAIN_PERCENTAGE))

train_generator = CustomDataset(batch_size=BATCH_SIZE, count=train_amount, img_size=IMG_SIZE)
val_generator = CustomDataset(batch_size=BATCH_SIZE, count=val_amount, img_size=IMG_SIZE, offset=1500)

# keras.backend.clear_session()
# model = models.get_model(IMG_SIZE)
# model = models.get_unet_model(IMG_SIZE)

# optimizer = tf.keras.optimizers.SGD(momentum=0.9, lr=LR)
# optimizer = tf.keras.optimizers.Adamax(learning_rate=LR)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])


# def IoULoss(targets, inputs, smooth=1e-6):
#     #flatten label and prediction tensors
#     inputs = tf.keras.backend.flatten(inputs)
#     targets = tf.keras.backend.flatten(targets)

#     intersection = tf.keras.backend.sum(tf.keras.backend.dot(targets, inputs))
#     total = tf.keras.backend.sum(targets) + tf.keras.backend.sum(inputs)
#     union = total - intersection

#     IoU = (intersection + smooth) / (union + smooth)
#     return 1 - IoU


# def DiceLoss(targets, inputs, smooth=1e-6):
#     #flatten label and prediction tensors
#     inputs = tf.keras.backend.flatten(inputs)
#     targets = tf.keras.backend.flatten(targets)

#     intersection = tf.keras.backend.sum(tf.keras.backend.dot(targets, inputs))
#     dice = (2*intersection + smooth) / (tf.keras.backend.sum(targets) + tf.keras.backend.sum(inputs) + smooth)
#     return 1 - dice

def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=-1)
    sum_ = tf.keras.backend.sum(tf.keras.backend.abs(y_true) + tf.keras.backend.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

# ALPHA = 0.8
# GAMMA = 2
# def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):
#     inputs = tf.keras.backend.flatten(inputs)
#     targets = tf.keras.backend.flatten(targets)

#     BCE = tf.keras.backend.binary_crossentropy(targets, inputs)
#     BCE_EXP = tf.keras.backend.exp(-BCE)
#     focal_loss = tf.keras.backend.mean(alpha * tf.keras.backend.pow((1-BCE_EXP), gamma) * BCE)

#     return focal_loss


# model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.MeanIoU(num_classes=2)], run_eagerly=False)
# model.compile(loss=jaccard_distance, optimizer=optimizer, metrics=[tf.keras.metrics.MeanIoU(num_classes=2)], run_eagerly=False)
# model.compile(loss=FocalLoss, optimizer=optimizer, metrics=[tf.keras.metrics.MeanIoU(num_classes=2)], run_eagerly=False)
# model.compile(loss=DiceLoss, optimizer=optimizer, metrics=[tf.keras.metrics.MeanIoU(num_classes=2)], run_eagerly=False)

preprocess_input = sm.get_preprocessing(BACKBONE)

model = sm.Linknet(
    BACKBONE,
    classes=len(CLASSES),
    encoder_weights=None
)

# total_loss = sm.losses.binary_focal_dice_loss
total_loss = sm.losses.bce_dice_loss
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
model.compile("ADAM", total_loss, metrics)


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    # monitor='val_mean_io_u',
    monitor='val_iou_score',
    # monitor='val_accuracy',
    mode='max',
    save_best_only=True)

print(model.summary())
plot_model(model, to_file=str(Path(__file__).resolve().parent.joinpath('model_plot.png')), show_shapes=True, show_layer_names=True)

history = model.fit(
   train_generator,
   validation_data=val_generator,
   epochs=EPOCHS,
   callbacks=[
       model_checkpoint_callback,
    #    tf.keras.callbacks.EarlyStopping(monitor='val_mean_io_u', patience=14)
    ]
)

model.save(save_path)
tfjs.converters.save_keras_model(model, modelpath)

test_x, test_y = val_generator[0]
preds = model.evaluate(test_x, test_y)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

np.savetxt(path + f"loss_history{suffix}.txt", np.array(history.history["val_loss"]), delimiter=",")
# np.savetxt(path + f"accuracy_history{suffix}.txt", np.array(history.history["val_mean_io_u"]), delimiter=",")
np.savetxt(path + f"accuracy_history{suffix}.txt", np.array(history.history["val_iou_score"]), delimiter=",")