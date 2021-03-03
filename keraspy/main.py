from pathlib import Path

import segmentation_models as sm
import tensorflowjs as tfjs
import tensorflow as tf

from loader import CustomDataset

sm.set_framework('tf.keras')
modelpath = str(Path(__file__).resolve().parent.joinpath('model'))
save_path = 'segmentation_model.h5'
checkpoint_path = 'checkpoint_model.h5'

BACKBONE = 'resnet34'
CLASSES = ['sofa']
LR = 0.0001
EPOCHS = 100
BATCH_SIZE = 8

preprocess_input = sm.get_preprocessing(BACKBONE)

train_generator = CustomDataset(batch_size=BATCH_SIZE, count=BATCH_SIZE*28)
val_generator = CustomDataset(batch_size=BATCH_SIZE, count=BATCH_SIZE*3, offset=BATCH_SIZE*28)

model = sm.Unet(
    BACKBONE,
    classes=len(CLASSES),
    encoder_weights='imagenet'
)

total_loss = sm.losses.binary_focal_dice_loss
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
model.compile("ADAM", total_loss, metrics)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_iou_score',
    mode='max',
    save_best_only=True)

model.fit_generator(
   generator=train_generator,
   validation_data=val_generator,
   # use_multiprocessing=True,
   use_multiprocessing=False,
   workers=2,
   epochs=EPOCHS,
   callbacks=[model_checkpoint_callback]
)

model.save(save_path)
tfjs.converters.save_keras_model(model, modelpath)

test_x, test_y = val_generator[0]
preds = model.evaluate(test_x, test_y)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
