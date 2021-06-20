from pathlib import Path

import tensorflow as tf
import tensorflowjs as tfjs

from models import ResizeLayer
from datetime import datetime


today = datetime.today().strftime('%Y-%m-%d')


modelpath = str(Path(__file__).resolve().parent.joinpath('model'))

model = tf.keras.models.load_model(f'unused_models/{today}/checkpoint_fast_scnn_layers.h5', custom_objects={"ResizeLayer": ResizeLayer})
# model = tf.keras.models.load_model(f'unused_models/{today}/fast_scnn_layers.h5', custom_objects={"ResizeLayer": ResizeLayer})
tfjs.converters.save_keras_model(model, modelpath)

# tensorflowjs_converter --input_format tfjs_layers_model --output_format tfjs_graph_model fast_scnn_model1/model.json graph_model