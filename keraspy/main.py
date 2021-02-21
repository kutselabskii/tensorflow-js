from pathlib import Path

import segmentation_models as sm
import tensorflowjs as tfjs

from loader import load_dataset

sm.set_framework('tf.keras')
modelpath = str(Path(__file__).resolve().parent.joinpath('model'))
save_path = 'segmentation_model.h5'

BACKBONE = 'resnet34'
CLASSES = ['sofa']
LR = 0.0001
EPOCHS = 40

preprocess_input = sm.get_preprocessing(BACKBONE)

x_train, y_train, x_val, y_val = load_dataset(1000, 480, 480)

x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

model = sm.Unet(
    BACKBONE,
    classes=len(CLASSES),
    # input_shape=(480, 480, 3),
    encoder_weights='imagenet'
)

total_loss = sm.losses.binary_focal_dice_loss
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
model.compile("ADAM", total_loss, metrics)

model.fit(
   x=x_train,
   y=y_train,
   batch_size=1,
   epochs=40,
   validation_data=(x_val, y_val),
)

preds = model.evaluate(x_val, y_val)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

model.save(save_path)
tfjs.converters.save_keras_model(model, modelpath)
