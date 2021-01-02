from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import tensorflowjs as tfjs

from pathlib import Path

datapath = str(Path(__file__).resolve().parent.joinpath('data.csv'))
modelpath = str(Path(__file__).resolve().parent.parent.joinpath('public/model'))

dataset = loadtxt(datapath, delimiter=',')
X = dataset[:, 0:8]
y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=30, batch_size=10)

_, accuracy = model.evaluate(X, y)
print(f'Accuracy: {round(accuracy * 100, 2)}%')

tfjs.converters.save_keras_model(model, modelpath)
