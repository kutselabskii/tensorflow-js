from pathlib import Path

import tensorflow as tf
import tensorflowjs as tfjs


modelpath = str(Path(__file__).resolve().parent.joinpath('model'))

model = tf.keras.models.load_model('my_model.h5')
tfjs.converters.save_keras_model(model, modelpath)
