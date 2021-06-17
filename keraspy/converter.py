from pathlib import Path

import tensorflow as tf
import tensorflowjs as tfjs

from models import ResizeLayer


modelpath = str(Path(__file__).resolve().parent.joinpath('fast_scnn_model1'))

model = tf.keras.models.load_model('checkpoint_fast_scnn_layers.h5', custom_objects={"ResizeLayer": ResizeLayer})
#model = tf.keras.models.load_model('fast_scnn_layers.h5', custom_objects={"ResizeLayer": ResizeLayer})
tfjs.converters.save_keras_model(model, modelpath)

# tensorflowjs_converter --input_format tfjs_layers_model --output_format tfjs_graph_model fast_scnn_model1/model.json graph_model