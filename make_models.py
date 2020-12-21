"""Make classification models and some utility functions."""

from datetime import datetime
from pathlib import Path

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers


def save_model(model: models.Model,
               timestamp=True,
               save_dir=Path('data/models')):
    """Save model in given directory under its 'name' property.

    By default adds a timestamp to the name.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%y-%m-%d_%H_%M_%S")
    model.save(save_dir / ('model_' + timestamp + '_' + model.name))


def load_model(name: str, load_dir=Path('data/models')) -> models.Model:
    """Load model with given name form a directory.

    This globs models in given dir for the 'name' string; so the 'name' isn't
    strictly the name of the file. If multiple models match the 'name' it
    returns the first alphabetically.
    """
    path = load_dir.glob('model*' + name)
    return models.load_model(sorted(path)[0])


def make_regularized_cnn(name: str, input_shape=(640, 640, 3)) -> models.Model:
    model = models.Sequential(name=name)
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
