from typing import Tuple
from functools import partial
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


def load_all_dataset(directory: str,
                     batch_size=16,
                     img_size=(640, 640),
                     seed=None) -> tf.data.Dataset:
    return image_dataset_from_directory(
        directory=directory,
        batch_size=batch_size,
        image_size=img_size,
        seed=seed
    )


def load_partial_dataset(directory: str,
                         batch_size=16,
                         img_size=(640, 640),
                         seed=666,
                         validation_split=0.2) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    load_dataset = partial(
        image_dataset_from_directory,
        directory=directory,
        validation_split=validation_split,
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )
    return load_dataset(subset="training"), load_dataset(subset="validation")


def main():
    img_path = "../images/train"
    train_dataset, validation_dataset = load_partial_dataset(img_path)
    dataset = load_all_dataset(img_path)


if __name__ == "__main__":
    main()
