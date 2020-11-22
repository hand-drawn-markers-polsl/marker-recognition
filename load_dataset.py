from typing import Tuple
from functools import partial
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


def _img_as_float(image: tf.float32,
                  labels: tf.int32):
    return tf.dtypes.cast(image, tf.float32) / 255, labels


def load_all_dataset(directory: str,
                     batch_size=16,
                     img_size=(640, 640),
                     seed=42) -> tf.data.Dataset:
    return image_dataset_from_directory(
        directory=directory,
        batch_size=batch_size,
        image_size=img_size,
        seed=seed
    ).map(_img_as_float)


def load_partial_dataset(directory: str,
                         batch_size=16,
                         img_size=(640, 640),
                         seed=42,
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
    train_ds = load_dataset(subset="training").map(_img_as_float)
    val_ds = load_dataset(subset="validation").map(_img_as_float)
    return train_ds, val_ds


def main():
    img_path = "../images/train"
    train_dataset, validation_dataset = load_partial_dataset(img_path)
    dataset = load_all_dataset(img_path)


if __name__ == "__main__":
    main()
