"""Data loaders for training and testing binary classifier."""

from functools import partial
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_train_dataset(directory: Path,
                       batch_size=32,
                       img_size=640,
                       validation_split=0.8,
                       seed=42,
                       augmentations: dict = None
                       ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load training data from directory.

    Load data for training and validation in form of data generators.
    This loader infers binary labels from 'true' and 'false' directories with
    data.

    :param directory: Path to training image set root.
    :parm batch_size: Size of batch generated during iteration.
    :param img_size: Image width and height (assumes square image ratio).
    :param validation_split: Split between train and validation data.
    :param seed: Seed for randomly returning data on generation.
    :param augmentation: Dict of Keras augmentation params, directly
        passed to init of 'ImageDataGenerator', refer to its documentation
        to examine list of available options.
    :return: List with two elements: training and validation data generators.
        Images are in range from 0 to 1.
    """
    if augmentations is None:
        augmentations = {}
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        **augmentations
    )

    load_subset = partial(
        datagen.flow_from_directory,
        directory=directory,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        seed=seed,
    )

    train_ds = load_subset(subset="training")
    val_ds = load_subset(subset="validation")
    return train_ds, val_ds


def load_test_dataset(directory: Path,
                      batch_size=32,
                      img_size=640) -> tf.data.Dataset:
    """Load test data from directory.

    Load data for evaluation in form of data generators.
    This loader infers binary labels from 'true' and 'false' directories with
    data. Images are loaded in a fixed order, shuffling on test data is
    disabled.

    :param directory: Path to training image set root.
    :parm batch_size: Size of batch generated during iteration.
    :return: Test data generator. Images are in range from 0 to 1.
    """
    datagen = ImageDataGenerator(rescale=1./255)
    test_ds = datagen.flow_from_directory(
        directory=directory,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    return test_ds


def main():
    """Load training and test data for demo and sanity check."""
    print('Loading train and validation data:')
    load_train_dataset(Path('data/train'))
    print('Loading test data:')
    load_test_dataset(Path('data/test'))


if __name__ == '__main__':
    main()
