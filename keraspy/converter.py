from pathlib import Path

import tensorflow as tf
import tensorflowjs as tfjs

from models import ResizeLayer
from datetime import datetime

import segmentation_models as sm


def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=-1)
    sum_ = tf.keras.backend.sum(tf.keras.backend.abs(y_true) + tf.keras.backend.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


today = datetime.today().strftime('%Y-%m-%d')
suffix = "linknet_straight_preprocess"

modelpath = str(Path(__file__).resolve().parent.joinpath('model'))

# model = tf.keras.models.load_model(f'unused_models/{today}/checkpoint_fast_scnn_binary{suffix}.h5', custom_objects={"ResizeLayer": ResizeLayer, "jaccard_distance": jaccard_distance})
# model = tf.keras.models.load_model(f"unused_models/2021-06-22/checkpoint_fast_scnn_binary_iou.h5", custom_objects={"ResizeLayer": ResizeLayer, "jaccard_distance": jaccard_distance})
# model = tf.keras.models.load_model(
#         f'unused_models/{today}/checkpoint_fast_scnn_binary{suffix}.h5', 
#         custom_objects={"binary_crossentropy_plus_dice_loss": sm.losses.bce_dice_loss, "iou_score": sm.metrics.IOUScore, "f1-score": sm.metrics.FScore}
#     )
model = tf.keras.models.load_model(f'unused_models/2021-06-27/fast_scnn_binarylinknet_straight_preprocess_255.h5', custom_objects={"jaccard_distance": jaccard_distance})
tfjs.converters.save_keras_model(model, modelpath)

# tensorflowjs_converter --input_format tfjs_layers_model --output_format tfjs_graph_model fast_scnn_model1/model.json graph_model