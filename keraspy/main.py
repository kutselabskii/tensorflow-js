from pathlib import Path

from keras import applications
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.models import Model, Sequential
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

from loader import load_dataset

modelpath = str(Path(__file__).resolve().parent.joinpath('model'))


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


gestures_count = 11
images_count = 120
height = 224
width = 224
epochs = 50
batch_size = 32
folder = 'dataset'
format = '.png'
save_path = 'my_model.h5'

x_train, y_train, x_test, y_test = load_dataset(gestures_count, images_count, height, width, folder, format)
# print(x_train.shape, y_train.shape)

y_train = convert_to_one_hot(y_train, gestures_count).T
y_test = convert_to_one_hot(y_test, gestures_count).T

base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(height, width, 3))

for layer in base_model.layers[:143]:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(gestures_count, activation='softmax'))

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

preds = model.evaluate(x_test, y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

model.save(save_path)
tfjs.converters.save_keras_model(model, modelpath)
