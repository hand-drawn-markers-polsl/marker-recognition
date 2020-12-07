import glob
from datetime import datetime

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers


def save_model(model: models.Model, name: str, timestamp=True):
    timestamp = datetime.now().strftime("%y-%m-%d_%H_%M_%S")
    model.save('log/models/model_' + timestamp + '_' + name)


def load_model(name: str) -> models.Model:
    path = glob.glob('log/models/model*' + name)
    return models.load_model(path[0])


def make_simple_cnn(input_shape=(640, 640, 3)) -> models.Model:
    model = models.Sequential()
    l2_regularizer = regularizers.l2(0.001)

    model.add(layers.Conv2D(32, (3, 3), kernel_regularizer=l2_regularizer,
                            activation='relu', input_shape=input_shape))
    model.add(layers.MaxPool2D())

    model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=l2_regularizer,
                            activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), kernel_regularizer=l2_regularizer,
                            activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), kernel_regularizer=l2_regularizer,
                            activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, kernel_regularizer=l2_regularizer,
                           activation='relu'))

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model
