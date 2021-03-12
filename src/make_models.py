"""Make classification models and some utility functions."""

from datetime import datetime
from pathlib import Path

from tensorflow.keras import applications, layers, models, regularizers


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
    """Build simple regularized cnn binary classifier."""
    model = models.Sequential(name=name)
    l2_regularizer = regularizers.l2(0.001)

    model.add(layers.Conv2D(32, (3, 3), kernel_regularizer=l2_regularizer,
                            activation=layers.LeakyReLU(),
                            input_shape=input_shape))
    model.add(layers.MaxPool2D())

    model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=l2_regularizer,
                            activation=layers.LeakyReLU()))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), kernel_regularizer=l2_regularizer,
                            activation=layers.LeakyReLU()))
    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, kernel_regularizer=l2_regularizer,
                           activation=layers.LeakyReLU()))

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def make_vgg16_based_cnn(name: str, input_shape=(640, 640, 3)) -> models.Model:
    """Build cnn binary classifier on top of VGG16 convolutional base."""
    base_model = applications.VGG16(include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = layers.Dropout(0.5)(base_model.layers[-1].output)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Dense(512, kernel_regularizer=layers.l2_regularizer,
                     activation='relu')(x)
    x = layers.Dropout(0.8)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.layers[0].input, outputs=output)

    return model
