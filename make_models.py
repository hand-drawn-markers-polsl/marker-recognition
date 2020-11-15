import glob
from datetime import datetime

from keras import layers
from keras import models


def save_model(model, name, timestamp=True):
    timestamp = datetime.now().strftime("%y-%m-%d-%H:%M:%S")
    model.save('log/models/model_' + timestamp + ' ' + name)


def load_model(name):
    path = glob.glob('log/models/model*' + name)
    return models.load_model(path[0])


def make_simple_cnn(input_shape=(640, 640, 3)):
    model = models.Sequential()

    model.add(layers.Conv2D(6, (3, 3), activation='relu',
                            input_shape=input_shape))
    model.add(layers.MaxPool2D())

    model.add(layers.Conv2D(6, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(4, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    return model
