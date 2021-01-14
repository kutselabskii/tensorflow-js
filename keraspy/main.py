from keras import applications
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
import numpy as np

from loader import load_dataset


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


gestures_count = 11
images_count = 120
height = 196
width = 196
epochs = 100

x_train, y_train, x_test, y_test = load_dataset(gestures_count, images_count, height, width)
# print(x_train.shape, y_train.shape)

y_train = convert_to_one_hot(y_train, gestures_count).T
y_test = convert_to_one_hot(y_test, gestures_count).T

base_model = applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=(height, width, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(gestures_count, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epochs, batch_size=64)

preds = model.evaluate(x_test, y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

model.save('my_model.h5')
