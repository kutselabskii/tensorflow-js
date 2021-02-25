from pathlib import Path

import segmentation_models as sm
import tensorflowjs as tfjs

from loader import CustomDataset

sm.set_framework('tf.keras')
modelpath = str(Path(__file__).resolve().parent.joinpath('model'))
save_path = 'segmentation_model.h5'

BACKBONE = 'resnet34'
CLASSES = ['sofa']
LR = 0.0001
EPOCHS = 100

preprocess_input = sm.get_preprocessing(BACKBONE)

train_generator = CustomDataset(batch_size=64, count=64*28)
val_generator = CustomDataset(batch_size=64, count=64*3, offset=64*28)

model = sm.Unet(
    BACKBONE,
    classes=len(CLASSES),
    encoder_weights='imagenet'
)

total_loss = sm.losses.binary_focal_dice_loss
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
model.compile("ADAM", total_loss, metrics)

model.fit_generator(
   generator=train_generator,
   validation_data=val_generator,
   use_multiprocessing=True,
   workers=4,
   epochs=EPOCHS
)

model.save(save_path)
tfjs.converters.save_keras_model(model, modelpath)

test_x, test_y = val_generator[0]
preds = model.evaluate(test_x, test_y)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
