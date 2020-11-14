from tensorflow.keras.preprocessing import image_dataset_from_directory
from functools import partial


def main():
    img_path = "../images/train"
    img_size = (640, 640)
    batch_size = 16

    load_dataset = partial(
        image_dataset_from_directory,
        directory=img_path,
        validation_split=0.2,
        seed=666,
        image_size=img_size,
        batch_size=batch_size
    )
    train_dataset = load_dataset(subset="training")
    validation_dataset = load_dataset(subset="validation")


if __name__ == "__main__":
    main()
